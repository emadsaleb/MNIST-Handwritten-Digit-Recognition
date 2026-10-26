
import os
import sys
from dataclasses import dataclass

import numpy as np

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join(
        "artifacts", "train_transformed.npz"
    )
    test_data_path: str = os.path.join(
        "artifacts", "test_transformed.npz"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading MNIST train and test data")

            train_data = np.load(train_path)
            test_data = np.load(test_path)

            X_train = train_data["X"]
            y_train = train_data["y"]

            X_test = test_data["X"]
            y_test = test_data["y"]

            logging.info("MNIST data loaded successfully")

            logging.info("Applying normalization to image data")

            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.train_data_path
                ),
                exist_ok=True,
            )

            logging.info("Saving transformed training data")

            np.savez_compressed(
                self.data_transformation_config.train_data_path,
                X=X_train,
                y=y_train,
            )

            logging.info("Saving transformed testing data")

            np.savez_compressed(
                self.data_transformation_config.test_data_path,
                X=X_test,
                y=y_test,
            )

            logging.info("Data transformation completed successfully")

            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        train_path = os.path.join("artifacts", "train.npz")
        test_path = os.path.join("artifacts", "test.npz")

        obj = DataTransformation()

        transformed_train_path, transformed_test_path = (
            obj.initiate_data_transformation(
                train_path,
                test_path,
            )
        )

        print("Transformed train data:", transformed_train_path)
        print("Transformed test data:", transformed_test_path)

    except Exception as e:
        print(e)
