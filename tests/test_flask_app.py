
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import flask_app


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True

    with flask_app.app.test_client() as client:
        yield client


class FakePredictPipeline:
    def __init__(self, digit=3, confidence=0.95, error=None):
        self.digit = digit
        self.confidence = confidence
        self.error = error
        self.received_image = None

    def predict(self, image):
        self.received_image = image

        if self.error:
            raise self.error

        return self.digit, self.confidence


def create_test_image(color="white"):
    image = Image.new("RGB", (50, 50), color=color)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_prediction_page(client):
    response = client.get("/predict-page")
    assert response.status_code == 200


def test_api_predict_success(client, monkeypatch):
    fake_pipeline = FakePredictPipeline()

    monkeypatch.setattr(
        flask_app,
        "PredictPipeline",
        lambda: fake_pipeline
    )

    response = client.post(
        "/api/predict",
        data={"image": (create_test_image(), "digit.png")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    result = response.get_json()
    assert result["prediction"] == 3
    assert result["confidence"] == pytest.approx(0.95)


def test_api_predict_without_image(client):
    response = client.post(
        "/api/predict",
        data={},
        content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_predict_invalid_image(client):
    response = client.post(
        "/api/predict",
        data={
            "image": (BytesIO(b"not an image"), "invalid.txt")
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert "Invalid image" in response.get_json()["error"]


def test_api_predict_pipeline_error(client, monkeypatch):
    fake_pipeline = FakePredictPipeline(
        error=RuntimeError("Model prediction failed")
    )

    monkeypatch.setattr(
        flask_app,
        "PredictPipeline",
        lambda: fake_pipeline
    )

    response = client.post(
        "/api/predict",
        data={"image": (create_test_image(), "digit.png")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 500
    assert "error" in response.get_json()


def test_api_preprocesses_image_before_prediction(client, monkeypatch):
    fake_pipeline = FakePredictPipeline()

    monkeypatch.setattr(
        flask_app,
        "PredictPipeline",
        lambda: fake_pipeline
    )

    response = client.post(
        "/api/predict",
        data={
            "image": (
                create_test_image(color="white"),
                "white.png"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    processed_image = fake_pipeline.received_image

    assert processed_image.shape == (28, 28)
    assert processed_image.dtype == np.float32

    # A white image is inverted by the API preprocessing.
    assert np.allclose(processed_image, 0.0)
