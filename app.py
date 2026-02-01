import streamlit as st
from PIL import Image, ImageDraw
import base64
import io
import os
import requests

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Smart Parking Detection",
    layout="centered"
)

st.title("🚗 Smart Parking Detection")
st.write("Upload an image and detect cars using Roboflow")

# ---------- ENVIRONMENT VARIABLES ----------
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = "aswin-gdjej"
WORKFLOW_ID = "find-cars"

if not API_KEY:
    st.error("❌ ROBOFLOW_API_KEY not found in Render Environment Variables")
    st.stop()

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting cars..."):

            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Roboflow HTTP API
            url = f"https://serverless.roboflow.com/workflows/{WORKSPACE}/{WORKFLOW_ID}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            payload = {
                "inputs": {
                    "image": {
                        "type": "base64",
                        "value": img_base64
                    }
                }
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                st.error("❌ Roboflow API request failed")
                st.text(response.text)
                st.stop()

            result = response.json()

        # ---------- DRAW DETECTIONS ----------
        draw = ImageDraw.Draw(image)
        predictions = result["outputs"][0]["predictions"]

        car_count = 0

        for pred in predictions:
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            conf = pred["confidence"]

            # Convert center → box
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"Car {conf:.2f}", fill="red")

            car_count += 1

        st.image(image, caption="Detected Cars", use_container_width=True)
        st.success(f"🚗 Cars Detected: {car_count}")
