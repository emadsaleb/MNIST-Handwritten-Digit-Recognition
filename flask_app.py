
from flask import Flask, request, render_template
from PIL import Image, UnidentifiedImageError
import numpy as np

from src.pipline.predict_pipline import PredictPipeline

application = Flask(__name__)
app = application


# Home page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction page
@app.route("/predict-page")
def index():
    return render_template("index.html")


# Prediction backend
@app.route("/predict", methods=["POST"])
def predict_data_point():
    try:
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            return render_template(
                "index.html",
                error="Please upload an image."
            ), 400

        # Open the uploaded image and convert it to grayscale
        with Image.open(uploaded_file).convert("L") as image:
            # Resize to MNIST dimensions
            image = image.resize((28, 28))

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        # Invert images with a predominantly white background
        if image_array.mean() > 127:
            image_array = 255.0 - image_array

        print("Image preprocessing completed.")
        print("Calling prediction pipeline...")

        # Run the prediction pipeline
        predict_pipeline = PredictPipeline()

        predicted_digit, confidence = predict_pipeline.predict(
            image_array
        )

        print("Prediction Result:", predicted_digit)
        print("Confidence:", confidence)

        # Return the prediction to the HTML page
        return render_template(
            "index.html",
            prediction=predicted_digit,
            confidence=confidence
        )

    except (UnidentifiedImageError, OSError):
        return render_template(
            "index.html",
            error="Invalid image. Please upload a valid image."
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error="Unable to process the image. Please try again."
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
