from __future__ import annotations

from pathlib import Path


class OperationsDashboardPage:
    def __init__(self, page: object, base_url: str) -> None:
        self.page = page
        self.base_url = base_url

    def goto(self) -> None:
        self.page.goto(self.base_url, wait_until="networkidle")

    def expect_shell_loaded(self) -> None:
        self.page.get_by_role("heading", name="ロト予測 運用ダッシュボード").wait_for()
        self.page.get_by_text("運用・実行結果・分析・検定・可視化・成果物管理を統合表示します。").wait_for()
        self.page.get_by_label("表示パネル(高速モード)").wait_for()

    def select_sidebar_panel(self, panel_name: str) -> None:
        self.page.get_by_label("表示パネル(高速モード)").select_option(label=panel_name)

    def expect_db_feedback(self) -> None:
        self.page.get_by_text("DB接続失敗:", exact=False).wait_for()

    def open_nf_lab_train(self) -> None:
        self.select_sidebar_panel("NeuralForecast 実行・検証ラボ")
        self.page.get_by_label("NeuralForecast 実行・検証ラボ メニュー").select_option(label="学習(train)")
        self.page.get_by_label("model").wait_for()
        self.page.get_by_label("backend").wait_for()
        self.page.get_by_role("button", name="Run train").wait_for()

    def open_operations_fallback(self) -> None:
        self.select_sidebar_panel("運用")
        self.page.get_by_text("DB未接続ですが", exact=False).wait_for()
        self.page.get_by_role("tab", name="機能動作確認").wait_for()
        self.page.get_by_role("tab", name="モデル解析ラボ").wait_for()
        self.page.get_by_role("tab", name="実測vs予測").wait_for()
        self.page.get_by_role("tab", name="Runner").wait_for()

    def save_screenshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=True)

    def accessibility_snapshot(self) -> dict | None:
        return self.page.accessibility.snapshot()

