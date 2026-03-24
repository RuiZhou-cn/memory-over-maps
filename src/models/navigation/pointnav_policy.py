"""DD-PPO PointNav policy wrapper + controller for habitat.Env."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
import types

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .pointnav_net import PointNavResNetPolicyDiscrete

logger = logging.getLogger(__name__)


class _StubModule(types.ModuleType):
    """Stub module for unpickling checkpoints that reference habitat-baselines."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = [f"<stub:{name}>"]
        self.__file__ = f"<stub:{name}>"
        self.__package__ = name

    def __getattr__(self, attr_name):
        if attr_name.startswith("__") and attr_name.endswith("__"):
            raise AttributeError(attr_name)
        full_name = f"{self.__name__}.{attr_name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        cls = type(attr_name, (), {
            "__init__": lambda self, *a, **kw: None,
            "__setstate__": lambda self, st: (
                self.__dict__.update(st) if isinstance(st, dict) else None
            ),
        })
        return cls


class _StubFinder(importlib.abc.MetaPathFinder):
    PREFIXES = ("habitat", "habitat_baselines", "vlfm")

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in self.PREFIXES:
            return importlib.machinery.ModuleSpec(
                fullname, _StubLoader(), is_package=True
            )
        return None


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


class WrappedPointNavResNetPolicy:
    """Wrapper for DD-PPO PointNav policy for single-environment inference.

    Manages hidden state / previous action bookkeeping.
    """

    def __init__(self, ckpt_path: str, device="cuda"):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.policy = load_pointnav_policy(ckpt_path)
        self.policy.to(device)

        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1, self.policy.net.num_recurrent_layers, 512, device=device
        )
        self.pointnav_prev_actions = torch.zeros(1, 1, device=device, dtype=torch.long)

    def act(
        self,
        observations: dict[str, Any],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Infer action towards (rho, theta) goal based on depth.

        Args:
            observations: Must contain:
                - "depth" (float32): Depth image (N, H, W, 1).
                - "pointgoal_with_gps_compass" (float32): (rho, theta) tensor (N, 2).
            masks: (N, 1) bool tensor. 1 after first step, 0 at episode start.
            deterministic: Whether to pick greedy action.

        Returns:
            Action tensor.
        """
        observations = _move_obs_to_device(observations, self.device)
        action, rnn_hidden_states = self.policy.act(
            observations,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )
        self.pointnav_prev_actions = action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states
        return action

    def reset(self) -> None:
        """Reset hidden state and previous action for new episode."""
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(
            self.pointnav_test_recurrent_hidden_states
        )
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def load_pointnav_policy(file_path: str) -> PointNavResNetPolicyDiscrete:
    """Load a discrete DD-PPO PointNav policy from a .pth checkpoint."""
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    try:
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)

    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    policy = PointNavResNetPolicyDiscrete()

    model_state = policy.state_dict()
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("critic."):
            continue
        if k in model_state:
            mapped[k] = v
        elif k == "action_distribution.linear.weight" and "action_distribution.weight" in model_state:
            mapped["action_distribution.weight"] = v
        elif k == "action_distribution.linear.bias" and "action_distribution.bias" in model_state:
            mapped["action_distribution.bias"] = v

    policy.load_state_dict(mapped, strict=False)

    missing = set(model_state.keys()) - set(mapped.keys())
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)

    return policy


def _move_obs_to_device(observations: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=dtype)
    return observations


class PointNavController:
    """Wraps WrappedPointNavResNetPolicy for use with habitat.Env observations. Paper Step 4 (Sec III-E)."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

    def __init__(self, weights_path: str, device: str = "cuda",
                 policy_input_size: int = 224):
        self.policy = WrappedPointNavResNetPolicy(weights_path, device=device)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.policy_input_size = policy_input_size

    def reset(self):
        self.policy.reset()

    def act(self, depth_obs: np.ndarray, rho: float, theta: float,
            step: int, deterministic: bool = True) -> int:
        """Get action from depth observation + polar goal coordinates.

        Args:
            depth_obs: Depth from habitat.Env, shape (H, W, 1) or (H, W),
                already normalized to [0, 1] by habitat's depth sensor.
            rho: Distance to goal (XZ plane, meters).
            theta: Angle to goal (radians).
            step: Current step number (0-indexed).
            deterministic: Use greedy action selection.

        Returns:
            Action index: 0=stop, 1=forward, 2=left, 3=right.
        """
        depth = depth_obs.squeeze(-1) if depth_obs.ndim == 3 else depth_obs

        sz = self.policy_input_size
        depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth_t = torch.nn.functional.interpolate(
            depth_t, size=(sz, sz), mode="area",
        )
        depth = depth_t.squeeze(0).permute(1, 2, 0).numpy()  # (sz, sz, 1)
        depth = depth[np.newaxis, :, :, :]  # (1, sz, sz, 1)

        policy_obs = {
            "depth": depth,
            "pointgoal_with_gps_compass": np.array([[rho, theta]], dtype=np.float32),
        }
        mask = torch.tensor([[step > 0]], dtype=torch.bool, device=self.device)
        action_tensor = self.policy.act(policy_obs, mask, deterministic=deterministic)
        return int(action_tensor.item())
