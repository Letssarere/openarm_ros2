import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

_EXPECTED_OBS_DIM = 28
_ACTION_DIM = 7


def _extract_running_stats(ckpt: Dict[str, Any]) -> Dict[str, np.ndarray]:
    mean = None
    var = None

    if not isinstance(ckpt, dict):
        return {}

    state_preproc = ckpt.get("state_preprocessor")
    if isinstance(state_preproc, dict):
        mean = state_preproc.get("running_mean")
        var = state_preproc.get("running_variance")
    if mean is None:
        mean = ckpt.get("state_preprocessor.running_mean")
    if var is None:
        var = ckpt.get("state_preprocessor.running_variance")

    if mean is None or var is None:
        return {}

    mean_arr = np.asarray(mean, dtype=np.float32)
    var_arr = np.asarray(var, dtype=np.float32)
    if mean_arr.shape[0] != _EXPECTED_OBS_DIM or var_arr.shape[0] != _EXPECTED_OBS_DIM:
        raise ValueError(
            f"Expected running stats of shape ({_EXPECTED_OBS_DIM},) but got "
            f"mean={mean_arr.shape}, var={var_arr.shape}"
        )
    std_arr = np.sqrt(np.maximum(var_arr, 1e-8)).astype(np.float32)
    return {"mean": mean_arr, "std": std_arr}


def _restore_mlp(policy_state: Dict[str, Any]) -> torch.nn.Module:
    from torch import nn

    class PolicyMLP(nn.Module):
        def __init__(self, in_dim: int = _EXPECTED_OBS_DIM, hidden=(64, 64), out_dim: int = _ACTION_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden[0]),
                nn.ELU(),
                nn.Linear(hidden[0], hidden[1]),
                nn.ELU(),
                nn.Linear(hidden[1], out_dim),
            )

        def forward(self, x):  # pragma: no cover - simple forward pass
            return self.net(x)

    policy = PolicyMLP()
    cleaned_state = policy_state
    if any(k.startswith("policy.") for k in policy_state.keys()):
        cleaned_state = {k.split("policy.", 1)[1]: v for k, v in policy_state.items() if k.startswith("policy.")}
    missing, unexpected = policy.load_state_dict(cleaned_state, strict=False)
    if missing:
        _LOGGER.warning("Missing keys while loading policy: %s", missing)
    if unexpected:
        _LOGGER.warning("Unexpected keys while loading policy: %s", unexpected)
    policy.eval()
    return policy


def load_ckpt(path: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = None
    stats: Dict[str, np.ndarray] = {}
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        stats = _extract_running_stats(ckpt)
    except Exception as exc:
        _LOGGER.warning("Failed to load checkpoint via torch.load: %s", exc)

    scripted_module = None
    try:
        scripted_module = torch.jit.load(ckpt_path.as_posix(), map_location=device)
        scripted_module.eval()
    except Exception:
        scripted_module = None

    if scripted_module is not None and not stats:
        raise RuntimeError(
            "Loaded TorchScript policy but no running stats were found in checkpoint. "
            "Provide scaler fallback (obs_scaler.npz)."
        )

    if scripted_module is not None:
        result = {"policy": scripted_module}
    elif ckpt is not None:
        if "policy" not in ckpt:
            raise KeyError("Checkpoint does not contain 'policy' state dict")
        result = {"policy": _restore_mlp(ckpt["policy"])}
    else:
        raise RuntimeError(f"Unable to load checkpoint from {ckpt_path}")

    if not stats:
        raise KeyError("Checkpoint does not include state_preprocessor running stats")

    result.update(stats)
    return result
