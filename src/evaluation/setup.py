"""Evaluation setup: device validation, model loading, run directory creation."""

from pathlib import Path


def validate_device(args) -> str:
    """Require CUDA, set args.device='cuda', print device name. Raises RuntimeError if unavailable."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    args.device = "cuda"
    return args.device


def load_sam3(args):
    """Load SAM3 segmenter."""
    from src.models.segmentation import SAM3Segmenter

    return SAM3Segmenter(
        device=args.device, confidence_threshold=args.sam3_confidence,
        batch_size=getattr(args, "sam3_batch_size", 5),
    )


def create_run_dir(eval_type: str, run_tags: str) -> Path:
    """Create and return a timestamped run directory under ``results/{eval_type}/``.

    Example::

        run_dir = create_run_dir("hm3d_objectnav", "all_val")
        # -> Path("results/hm3d_objectnav/20260223_120000_all_val")
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{run_tags}" if run_tags else timestamp
    run_dir = Path("results") / eval_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_and_merge_config(cli_args, load_config_fn, merge_fn):
    """Load YAML config and merge with CLI overrides.

    Shared across all eval scripts.

    Returns:
        Tuple of (merged args namespace, resolved config Path).
    """
    config_path = Path(cli_args.config)
    if config_path.exists():
        print(f"Loading config: {config_path}")
        cfg = load_config_fn(str(config_path))
    else:
        print(f"Config not found: {config_path}, using CLI defaults")
        cfg = {}
    return merge_fn(cfg, cli_args), config_path


def get_scene_tag(args) -> str:
    """Return scene tag for run directory naming."""
    return args.scene if args.scene else "all"


def apply_keyframing(scene_loader, args) -> list:
    """Apply keyframe selection if enabled, returning filtered frame IDs."""
    all_frame_ids = scene_loader.frame_ids
    if args.keyframing:
        from src.utils.keyframe import KeyframeManager
        kf_mgr = KeyframeManager(
            rotation_threshold_deg=args.keyframe_rotation,
            translation_threshold_m=args.keyframe_translation,
            min_frames_between=1,
        )
        poses, fids = scene_loader.get_all_poses()
        all_frame_ids = kf_mgr.select_keyframes(poses, fids)
    return all_frame_ids


