"""Smoke-Test der Streamlit-App: startet sie headless und prüft, dass keine Ausnahme auftritt."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from churn.config import METADATA_PATH, MODEL_PATH

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and METADATA_PATH.exists()),
    reason="Modellartefakte fehlen – zuerst `python -m churn.train` ausführen",
)


def test_app_starts_without_errors():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("📉 Churn")


def test_what_if_form_returns_score():
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.button[0].click().run()  # Formular mit Standardwerten absenden
    assert not at.exception
    scores = [m for m in at.metric if m.label == "Churn-Score"]
    assert scores and 0.0 <= float(scores[0].value) <= 1.0
