import pytest
from fastapi.testclient import TestClient

from app import predict as predict_module


@pytest.fixture
def client():
    return TestClient(predict_module.app)


def test_upload_rejects_unsupported_format(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported video format"


def test_inference_endpoint_returns_prediction(client, monkeypatch):
    expected = {
        "timestamp": "2026-01-01T00:00:00",
        "label": "Violence",
        "confidence": 0.91,
        "alert": True,
    }

    monkeypatch.setattr(
        predict_module,
        "predict_video",
        lambda _: expected,
    )

    response = client.post(
        "/predict",
        files={"file": ("sample.mp4", b"fake video content", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == expected
