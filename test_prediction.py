
import numpy as np

from src.pipline.predict_pipline import PredictPipeline


# Load one image from the transformed test dataset
test_data = np.load("artifacts/test_transformed.npz")

image = test_data["X"][0]
actual_digit = int(test_data["y"][0])

# Make prediction
pipeline = PredictPipeline()
predicted_digit, confidence = pipeline.predict(image)

print("Actual digit:", actual_digit)
print("Predicted digit:", predicted_digit)
print(f"Confidence: {confidence:.2%}")
