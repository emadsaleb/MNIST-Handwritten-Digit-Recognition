
import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:5000/api/predict"

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Upload a handwritten digit from 0 to 9 "
    "and let the Deep Learning model predict it."
)

st.info(
    "For best results, upload a clear image containing "
    "one handwritten digit."
)

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        width=250
    )

    if st.button("Predict Digit", type="primary"):
        try:
            with st.spinner("Running prediction..."):
                uploaded_file.seek(0)

                response = requests.post(
                    API_URL,
                    files={
                        "image": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

            if response.ok:
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.success("Prediction completed!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Digit",
                        prediction
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            else:
                try:
                    error_message = response.json().get(
                        "error",
                        "Prediction failed."
                    )
                except ValueError:
                    error_message = "Prediction failed."

                st.error(error_message)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the Flask backend. "
                "Start flask_app.py first."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. "
                "Please try again."
            )

        except (ValueError, KeyError):
            st.error(
                "The backend returned an unexpected response."
            )
