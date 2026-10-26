
import os
import sys
from dataclasses import dataclass

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.activations import relu, softmax

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.keras"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_path, test_path):
        try:
            logging.info("Loading transformed MNIST data")

            train_data = np.load(train_path)
            test_data = np.load(test_path)

            X_train = train_data["X"]
            y_train = train_data["y"]

            X_test = test_data["X"]
            y_test = test_data["y"]

            logging.info("Building the ANN model")

            model = Sequential()

            model.add(Flatten(input_shape=(28, 28)))
            model.add(Dense(128, activation=relu))
            model.add(Dense(128, activation=relu))
            model.add(Dense(10, activation=softmax))

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            logging.info("Starting model training")

            model.fit(
                X_train,
                y_train,
                epochs=5,
                batch_size=32,
                verbose=1,
            )

            logging.info("Evaluating model on test data")

            test_loss, test_accuracy = model.evaluate(
                X_test,
                y_test,
                verbose=1,
            )

            logging.info(f"Test loss: {test_loss}")
            logging.info(f"Test accuracy: {test_accuracy}")

            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True,
            )

            model.save(
                self.model_trainer_config.trained_model_file_path
            )

            logging.info("Model saved successfully")

            print(f"Test loss: {test_loss:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(
                "Trained model saved to:",
                self.model_trainer_config.trained_model_file_path,
            )

            return test_accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        train_path = os.path.join(
            "artifacts", "train_transformed.npz"
        )
        test_path = os.path.join(
            "artifacts", "test_transformed.npz"
        )

        trainer = ModelTrainer()

        trainer.initiate_model_trainer(
            train_path=train_path,
            test_path=test_path,
        )

    except Exception as e:
        print(e)
