
import os
import sys
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.datasets import mnist

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.npz")
    test_data_path: str = os.path.join("artifacts", "test.npz")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            logging.info("Loaded the MNIST dataset successfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            np.savez_compressed(
                self.ingestion_config.train_data_path,
                X=X_train,
                y=y_train
            )

            np.savez_compressed(
                self.ingestion_config.test_data_path,
                X=X_test,
                y=y_test
            )

            logging.info("MNIST data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print("Train data path:", train_data)
    print("Test data path:", test_data)
