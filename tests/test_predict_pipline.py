
import numpy as np
import pytest

from src.exception import CustomException
from src.pipline import predict_pipline as predict_module


class FakeModel:
    """Mock model to test the prediction pipeline without loading TensorFlow."""

    def __init__(self):
        self.received_image = None

    def predict(self, image, verbose=0):
        self.received_image = image

        probabilities = np.zeros((1, 10), dtype=np.float32)
        probabilities[0, 3] = 0.8
        probabilities[0, 2] = 0.2

        return probabilities


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda path: model
    )

    return model


def test_predict_returns_digit_and_confidence(fake_model):
    pipeline = predict_module.PredictPipeline()
    image = np.full((28, 28), 255, dtype=np.uint8)

    digit, confidence = pipeline.predict(image)

    assert digit == 3
    assert confidence == pytest.approx(0.8)


def test_predict_normalizes_image(fake_model):
    pipeline = predict_module.PredictPipeline()
    image = np.full((28, 28), 255, dtype=np.uint8)

    pipeline.predict(image)

    assert fake_model.received_image.shape == (1, 28, 28)
    assert fake_model.received_image.dtype == np.float32
    assert np.allclose(fake_model.received_image, 1.0)


def test_predict_rejects_invalid_image_shape(fake_model):
    pipeline = predict_module.PredictPipeline()
    image = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(CustomException):
        pipeline.predict(image)
