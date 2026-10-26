
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model

from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, image):
        try:
            model_path = os.path.join(
                "artifacts", "model.keras"
            )

            logging.info("Before Loading Model")

            model = load_model(model_path)

            logging.info("After Loading Model")

            image = np.asarray(image, dtype=np.float32)

            if image.shape != (28, 28):
                raise ValueError(
                    f"Expected image shape (28, 28), got {image.shape}"
                )

            # Normalize pixel values
            if image.max() > 1.0:
                image = image / 255.0

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            probabilities = model.predict(image, verbose=0)

            predicted_digit = int(
                np.argmax(probabilities[0])
            )
            confidence = float(
                np.max(probabilities[0])
            )

            return predicted_digit, confidence

        except Exception as e:
            raise CustomException(e, sys)
