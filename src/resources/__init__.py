"""Resource helper package.

Keep this package import lightweight.

Historically this module eagerly imported every exogenous pipeline. That made
innocent imports such as ``from resources.utils import ...`` pull in DB writers
and optional heavy dependencies during dashboard startup. The dashboard only
needs small utility helpers at import time, so optional pipeline symbols are now
loaded lazily via ``__getattr__``.
"""

from __future__ import annotations

from .config import ResourcesConfig

_LAZY_EXPORTS = {
    "ResourceRun": (".context", "ResourceRun"),
    "start_run": (".context", "start_run"),
    "resource_span": (".span", "resource_span"),
    "ExogBuildSpec": (".exog_pipeline", "ExogBuildSpec"),
    "build_exog_dataframe": (".exog_pipeline", "build_exog_dataframe"),
    "run_exog_build": (".exog_pipeline", "run_exog_build"),
    "ChronosExogSpec": (".chronos_exog_pipeline", "ChronosExogSpec"),
    "build_chronos_exog_dataframe": (".chronos_exog_pipeline", "build_chronos_exog_dataframe"),
    "run_chronos_exog_build": (".chronos_exog_pipeline", "run_chronos_exog_build"),
    "TimesFMExogSpec": (".timesfm_exog_pipeline", "TimesFMExogSpec"),
    "build_timesfm_exog_dataframe": (".timesfm_exog_pipeline", "build_timesfm_exog_dataframe"),
    "run_timesfm_exog_build": (".timesfm_exog_pipeline", "run_timesfm_exog_build"),
    "Uni2TSExogSpec": (".uni2ts_exog_pipeline", "Uni2TSExogSpec"),
    "build_uni2ts_exog_dataframe": (".uni2ts_exog_pipeline", "build_uni2ts_exog_dataframe"),
    "run_uni2ts_exog_build": (".uni2ts_exog_pipeline", "run_uni2ts_exog_build"),
}

__all__ = ["ResourcesConfig", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
