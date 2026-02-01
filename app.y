import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from inference_sdk import InferenceHTTPClient


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Smart Parking Detection",
    layout="centered"
)

st.title("🚗 Smart Parking Detection")
st.write("Upload an image and detect cars using Roboflow")

# ---------- ROBOFLOW CONFIG ----------
API_KEY = "VXketd8rJBvbWwf5E8EE"
WORKSPACE = "aswin-gdjej"
WORKFLOW_ID = "find-cars"

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Detecting..."):

            # Convert image to Base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Roboflow client
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=API_KEY
            )

            # Run workflow
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": img_base64}
            )

        # ----------- DRAW DETECTIONS ----------
        draw = ImageDraw.Draw(image)
        predictions = result[0]["predictions"]["predictions"]

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

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Label
            label = f"Car {conf:.2f}"
            draw.text((x1, y1 - 10), label, fill="red")

            car_count += 1

        st.image(image, caption="Detected Cars", use_container_width=True)
        st.success(f"🚗 Cars Detected: {car_count}")
