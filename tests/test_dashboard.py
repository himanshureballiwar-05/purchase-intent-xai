"""Smoke-test every dashboard page using Streamlit's headless AppTest."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.config.settings import PROJECT_ROOT  # noqa: E402

APP = str(PROJECT_ROOT / "dashboard" / "app.py")
PAGES = [
    "1. Overview",
    "2. Live prediction",
    "3. Behaviour analytics",
    "4. Model comparison",
    "5. Explainability",
    "6. Model & system",
]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "model_pipeline.joblib").exists(),
    reason="committed model artifacts not present",
)
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
