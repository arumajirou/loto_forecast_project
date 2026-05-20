from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class _Context:
    root: FakeStreamlit

    def __enter__(self) -> FakeStreamlit:
        return self.root

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


@dataclass
class FakeStreamlit:
    select_values: dict[str, Any] = field(default_factory=dict)
    button_values: dict[str, bool] = field(default_factory=dict)
    toggle_values: dict[str, bool] = field(default_factory=dict)
    text_values: dict[str, Any] = field(default_factory=dict)
    multi_values: dict[str, list[Any]] = field(default_factory=dict)
    slider_values: dict[str, Any] = field(default_factory=dict)
    number_values: dict[str, Any] = field(default_factory=dict)
    captured: list[tuple[str, Any]] = field(default_factory=list)

    def _capture(self, kind: str, payload: Any) -> None:
        self.captured.append((kind, payload))

    def __enter__(self) -> FakeStreamlit:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def caption(self, value: Any) -> None:
        self._capture("caption", value)

    def subheader(self, value: Any) -> None:
        self._capture("subheader", value)

    def markdown(self, value: Any) -> None:
        self._capture("markdown", value)

    def info(self, value: Any) -> None:
        self._capture("info", value)

    def warning(self, value: Any) -> None:
        self._capture("warning", value)

    def error(self, value: Any) -> None:
        self._capture("error", value)

    def success(self, value: Any) -> None:
        self._capture("success", value)

    def json(self, value: Any) -> None:
        self._capture("json", value)

    def code(self, value: Any, language: str | None = None) -> None:
        self._capture("code", (value, language))

    def metric(self, label: str, value: Any) -> None:
        self._capture("metric", (label, value))

    def plotly_chart(self, *args: Any, **kwargs: Any) -> None:
        self._capture("plotly_chart", (args, kwargs))

    def line_chart(self, *args: Any, **kwargs: Any) -> None:
        self._capture("line_chart", (args, kwargs))

    def graphviz_chart(self, *args: Any, **kwargs: Any) -> None:
        self._capture("graphviz_chart", (args, kwargs))

    def rerun(self) -> None:
        self._capture("rerun", None)

    def tabs(self, labels: list[str]) -> list[_Context]:
        self._capture("tabs", labels)
        return [_Context(self) for _ in labels]

    def columns(self, count: int) -> list[FakeStreamlit]:
        return [self for _ in range(count)]

    def expander(self, label: str, expanded: bool = False) -> _Context:
        self._capture("expander", (label, expanded))
        return _Context(self)

    def selectbox(self, label: str, options: list[Any], index: int = 0, key: str | None = None) -> Any:
        if key is not None and key in self.select_values:
            return self.select_values[key]
        if label in self.select_values:
            return self.select_values[label]
        return options[index] if options else None

    def multiselect(
        self,
        label: str,
        options: list[Any],
        default: list[Any] | None = None,
        key: str | None = None,
    ) -> list[Any]:
        if key is not None and key in self.multi_values:
            return self.multi_values[key]
        if label in self.multi_values:
            return self.multi_values[label]
        return list(default or [])

    def text_input(self, label: str, value: str = "", key: str | None = None, **_: Any) -> str:
        if key is not None and key in self.text_values:
            return str(self.text_values[key])
        if label in self.text_values:
            return str(self.text_values[label])
        return value

    def text_area(self, label: str, value: str = "", key: str | None = None, **_: Any) -> str:
        return self.text_input(label, value=value, key=key)

    def date_input(self, label: str, value: Any, key: str | None = None, **_: Any) -> Any:
        if key is not None and key in self.text_values:
            return self.text_values[key]
        return value

    def slider(self, label: str, *args: Any, value: Any = None, key: str | None = None, **_: Any) -> Any:
        if key is not None and key in self.slider_values:
            return self.slider_values[key]
        if label in self.slider_values:
            return self.slider_values[label]
        return value if value is not None else args[0]

    def number_input(self, label: str, value: Any = None, key: str | None = None, **_: Any) -> Any:
        if key is not None and key in self.number_values:
            return self.number_values[key]
        if label in self.number_values:
            return self.number_values[label]
        return value

    def toggle(self, label: str, value: bool = False, key: str | None = None, **_: Any) -> bool:
        if key is not None and key in self.toggle_values:
            return self.toggle_values[key]
        if label in self.toggle_values:
            return self.toggle_values[label]
        return value

    def button(self, label: str, key: str | None = None, **_: Any) -> bool:
        if key is not None and key in self.button_values:
            return self.button_values[key]
        if label in self.button_values:
            return self.button_values[label]
        return False

    def data_editor(self, df: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return df

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        self._capture("download_button", (args, kwargs))

    class column_config:
        class SelectboxColumn:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        class CheckboxColumn:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
