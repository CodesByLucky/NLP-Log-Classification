import io

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

client = TestClient(server.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_classify_returns_labeled_csv(monkeypatch):
    def fake_classify(df):
        df = df.copy()
        df["target_label"] = "System Notification"
        return df

    monkeypatch.setattr(server, "classify_dataframe", fake_classify)
    csv = "source,log_message\nModernCRM,Backup completed successfully.\n"
    response = client.post("/classify/", files={"file": ("logs.csv", csv, "text/csv")})

    assert response.status_code == 200
    result = pd.read_csv(io.StringIO(response.text))
    assert list(result["target_label"]) == ["System Notification"]


def test_rejects_non_csv_upload():
    response = client.post("/classify/", files={"file": ("logs.txt", "hello", "text/plain")})
    assert response.status_code == 400


def test_rejects_missing_columns():
    response = client.post("/classify/", files={"file": ("logs.csv", "foo,bar\n1,2\n", "text/csv")})
    assert response.status_code == 400
    assert "log_message" in response.json()["detail"]
