"""
ZWM — 天地人三才世界模型规划器
Trinity World Model Planner based on I Ching mathematics.
"""

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__ = _pkg_version("zwm")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:  # pragma: no cover — defensive fallback
    __version__ = "0.1.0"
