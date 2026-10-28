
from flask import Flask, request, render_template, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np

from src.pipline.predict_pipline import PredictPipeline

application = Flask(__name__)
app = application


# Home page — existing route preserved
@app.route("/")
def home():
    return render_template("home.html")


# Prediction page — existing route preserved
@app.route("/predict-page")
def index():
    return render_template("index.html")


# Existing HTML prediction backend — preserved
@app.route("/predict", methods=["POST"])
def predict_data_point():
    try:
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            return render_template(
                "index.html",
                error="Please upload an image."
            ), 400

        with Image.open(uploaded_file).convert("L") as image:
            image = image.resize((28, 28))
            image_array = np.asarray(image, dtype=np.float32)

        if image_array.mean() > 127:
            image_array = 255.0 - image_array

        print("Image preprocessing completed.")
        print("Calling prediction pipeline...")

        predict_pipeline = PredictPipeline()
        predicted_digit, confidence = predict_pipeline.predict(
            image_array
        )

        print("Prediction Result:", predicted_digit)
        print("Confidence:", confidence)

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


# NEW API route — for the Streamlit UI
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({
                "error": "Please upload an image."
            }), 400

        with Image.open(uploaded_file.stream) as img:
            image = img.convert("L").resize((28, 28))
            image_array = np.asarray(image, dtype=np.float32)

        if image_array.mean() > 127:
            image_array = 255.0 - image_array

        predict_pipeline = PredictPipeline()
        predicted_digit, confidence = predict_pipeline.predict(
            image_array
        )

        return jsonify({
            "prediction": int(predicted_digit),
            "confidence": float(confidence)
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid image. Please upload a valid image."
        }), 400

    except Exception:
        app.logger.exception("API prediction failed")
        return jsonify({
            "error": "Unable to process the image. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
