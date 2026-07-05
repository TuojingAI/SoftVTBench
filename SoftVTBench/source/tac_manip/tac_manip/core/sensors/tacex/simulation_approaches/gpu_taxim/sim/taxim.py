from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from .calibration import CALIB_GELSIGHT

if TYPE_CHECKING:
    try:
        from .taxim_torch import TaximTorch
    except ImportError:
        TaximTorch = None
    try:
        from .taxim_jax import TaximJax
    except ImportError:
        TaximJax = None


def _mk_taxim_torch(
    calib_folder: Path,
    params: dict[str, dict[str, Any]] | None,
    device: str | None,
) -> TaximTorch:
    try:
        import torch.cuda

        from .taxim_torch import TaximTorch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return TaximTorch(device=device, calib_folder=calib_folder, params=params)
    except ImportError:
        raise ImportError("Could not import torch or torch-scatter. Please install both to use the torch backend.")


def _mk_taxim_jax(
    calib_folder: Path,
    params: dict[str, dict[str, Any]] | None,
    device: str | None,
) -> TaximJax:
    try:
        import jax

        from .taxim_jax import TaximJax

        return TaximJax(calib_folder=calib_folder, params=params, device=jax.devices(device)[0])
    except ImportError:
        raise ImportError("Could not import jax. Please install jax to use the jax backend.")


@overload
def Taxim(  # noqa: E704
    calib_folder: Path = ...,
    params: dict[str, dict[str, Any]] | None = ...,
    backend: Literal["torch"] = ...,
) -> TaximTorch: ...


@overload
def Taxim(  # noqa: E704
    calib_folder: Path = ...,
    params: dict[str, dict[str, Any]] | None = ...,
    backend: Literal["jax"] = ...,
) -> TaximJax: ...


@overload
def Taxim(  # noqa: E704
    calib_folder: Path = ...,
    params: dict[str, dict[str, Any]] | None = ...,
    backend: Literal["auto"] = ...,
) -> TaximTorch | TaximJax: ...


def Taxim(
    calib_folder: Path = CALIB_GELSIGHT,
    params: dict[str, dict[str, Any]] | None = None,
    backend: Literal["torch", "jax", "auto"] = "auto",
    device: str | None = None,
) -> TaximTorch | TaximJax:
    """
    Create a Taxim simulator.

    :param calib_folder: Path to the folder with the calibration files.
    :param params:       Simulator parameters. Values set in this dictionary override values set in params.json
                         in the calib_folder.
    :param backend:      Backend to use. Either "torch" or "jax" or "auto", which will try to load torch and default to
                         jax otherwise.
    :param device:       Device to use. Only relevant for the torch backend. If None, the device is chosen
                         automatically.
    """
    if backend == "torch":
        return _mk_taxim_torch(calib_folder, params, device)
    elif backend == "jax":
        return _mk_taxim_jax(calib_folder, params, device)
    elif backend == "auto":
        # Prefer torch by default; fall back to jax if torch (or torch-scatter) isn't available.
        with contextlib.suppress(ImportError):
            return _mk_taxim_torch(calib_folder, params, device)
        try:
            return _mk_taxim_jax(calib_folder, params, device)
        except ImportError:
            raise ImportError(
                "Could load neither PyTorch nor JAX backend. Please install either to use the torch or jax backend."
            )
    else:
        raise ValueError(f"Unknown backend {backend}")
