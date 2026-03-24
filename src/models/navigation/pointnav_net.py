# Adapted from:
# - ResNet: https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet.py
# - RNN encoder: https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/models/rnn_state_encoder.py
# - PointNav net: VLFM's nh_pointnav_policy.py (Boston Dynamics AI Institute)
# Filtered to ResNet-18 + LSTM only. Added PointNavResNetPolicyDiscrete for DD-PPO.

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.conv import Conv2d
from torch.nn.utils.rnn import PackedSequence


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.convs(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_planes: int,
        ngroups: int,
        block: Type[BasicBlock],
        layers: List[int],
        cardinality: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality
        self.inplanes = base_planes

        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(block, ngroups, base_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, ngroups, base_planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, ngroups, base_planes * 8, layers[3], stride=2)

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2**5)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, ngroups, stride, downsample, cardinality=self.cardinality)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18(in_channels: int, base_planes: int, ngroups: int) -> ResNet:
    return ResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])


class RNNStateEncoder(nn.Module):
    """RNN state encoder base class for RL. Wraps nn.LSTM; subclass LSTMStateEncoder handles (h, c) packing."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.num_recurrent_layers = num_layers * 2
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.layer_init()

    def layer_init(self) -> None:
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def pack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def unpack_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.contiguous()

    def single_forward(
        self, x: torch.Tensor, hidden_states: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.where(masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(()))
        x, hidden_states = self.rnn(x.unsqueeze(0), self.unpack_hidden(hidden_states))
        hidden_states = self.pack_hidden(hidden_states)
        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq, hidden_states = build_rnn_inputs(x, hidden_states, masks, rnn_build_seq_info)
        rnn_ret = self.rnn(x_seq, self.unpack_hidden(hidden_states))
        x_seq = rnn_ret[0]
        hidden_states = rnn_ret[1]
        hidden_states = self.pack_hidden(hidden_states)
        x, hidden_states = build_rnn_out_from_seq(x_seq, hidden_states, rnn_build_seq_info)
        return x, hidden_states

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) == hidden_states.size(1):
            assert rnn_build_seq_info is None
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            assert rnn_build_seq_info is not None
            x, hidden_states = self.seq_forward(x, hidden_states, masks, rnn_build_seq_info)
        hidden_states = hidden_states.permute(1, 0, 2)
        return x, hidden_states


class LSTMStateEncoder(RNNStateEncoder):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__(input_size, hidden_size, num_layers)

    def pack_hidden(self, hidden_states: Any) -> torch.Tensor:
        return torch.cat(hidden_states, 0)

    def unpack_hidden(self, hidden_states: torch.Tensor) -> Any:
        lstm_states = torch.chunk(hidden_states.contiguous(), 2, 0)
        return (lstm_states[0], lstm_states[1])


def build_rnn_inputs(x, rnn_states, not_dones, rnn_build_seq_info):
    select_inds = rnn_build_seq_info["select_inds"]
    num_seqs_at_step = rnn_build_seq_info["cpu_num_seqs_at_step"]
    x_seq = PackedSequence(x.index_select(0, select_inds), num_seqs_at_step, None, None)
    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    sequence_starts = rnn_build_seq_info["sequence_starts"]
    rnn_states = rnn_states.index_select(1, rnn_state_batch_inds)
    rnn_states.masked_fill_(
        torch.logical_not(not_dones.view(1, -1, 1).index_select(1, sequence_starts)), 0
    )
    return x_seq, rnn_states


def build_rnn_out_from_seq(x_seq, hidden_states, rnn_build_seq_info):
    select_inds = rnn_build_seq_info["select_inds"]
    x = x_seq.data.index_select(0, _invert_permutation(select_inds))
    last_sequence_in_batch_inds = rnn_build_seq_info["last_sequence_in_batch_inds"]
    rnn_state_batch_inds = rnn_build_seq_info["rnn_state_batch_inds"]
    output_hidden_states = hidden_states.index_select(
        1,
        last_sequence_in_batch_inds[_invert_permutation(rnn_state_batch_inds[last_sequence_in_batch_inds])],
    )
    return x, output_hidden_states


def _invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    orig_size = permutation.size()
    permutation = permutation.view(-1)
    output = torch.empty_like(permutation)
    output.scatter_(0, permutation, torch.arange(0, permutation.numel(), device=permutation.device))
    return output.view(orig_size)


class ResNetEncoder(nn.Module):
    visual_keys = ["depth"]

    def __init__(self) -> None:
        super().__init__()
        self.running_mean_and_var = nn.Sequential()
        self.backbone = resnet18(1, 32, 16)
        self.compression = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.GroupNorm(1, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            obs_k = obs_k.permute(0, 3, 1, 2)
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(nn.Module):
    def __init__(self, discrete_actions: bool = False, no_fwd_dict: bool = False):
        super().__init__()
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(4 + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(in_features=2, out_features=32, bias=True)
        self.tgt_embeding = nn.Linear(in_features=3, out_features=32, bias=True)
        self.visual_encoder = ResNetEncoder()
        self.visual_fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = LSTMStateEncoder(576, 512, 2)
        self.num_recurrent_layers = self.state_encoder.num_recurrent_layers
        self.discrete_actions = discrete_actions
        self.no_fwd_dict = no_fwd_dict

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        visual_feats = self.visual_encoder(observations)
        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        goal_observations = observations["pointgoal_with_gps_compass"]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )
        x.append(self.tgt_embeding(goal_observations))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks, rnn_build_seq_info)

        if self.no_fwd_dict:
            return out, rnn_hidden_states

        return out, rnn_hidden_states, {}


class PointNavResNetPolicyDiscrete(nn.Module):
    """Discrete action PointNav policy matching Habitat DD-PPO checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.net = PointNavResNetNet(discrete_actions=True)
        self.action_distribution = nn.Linear(512, 4)

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        logits = self.action_distribution(features)

        if deterministic:
            action = logits.argmax(dim=-1, keepdim=True)
        else:
            action = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)

        return action, rnn_hidden_states


