
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    # Data Ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = (
        data_ingestion.initiate_data_ingestion()
    )

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    # Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

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

            logging.info("Loading trained MNIST model")

            model = load_model(model_path)

            # Convert input to a NumPy array
            image = np.array(image, dtype=np.float32)

            # Validate image dimensions
            if image.shape != (28, 28):
                raise ValueError(
                    "Expected an image with shape (28, 28)"
                )

            # Normalize pixel values if needed
            if image.max() > 1.0:
                image = image / 255.0

            # Add batch dimension: (28, 28) -> (1, 28, 28)
            image = np.expand_dims(image, axis=0)

            logging.info("Making digit prediction")

            probabilities = model.predict(image, verbose=0)

            predicted_digit = int(np.argmax(probabilities[0]))
            confidence = float(np.max(probabilities[0]))

            return predicted_digit, confidence

        except Exception as e:
            raise CustomException(e, sys)
