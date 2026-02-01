import streamlit as st
from PIL import Image, ImageDraw
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Parking Detection",
    layout="centered"
)

st.title("🚗 Smart Parking Detection")
st.write("Car detection using Roboflow Detect API")

# ---------------- ROB0FLOW CONFIG ----------------
API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# 🔴 MUST MATCH YOUR ROB0FLOW MODEL EXACTLY
MODEL_NAME = "find-cars"
MODEL_VERSION = 1   # change ONLY if your model version is different

if not API_KEY:
    st.error("❌ ROBOFLOW_API_KEY not found in Environment Variables")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and st.button("🔍 Run Detection"):

    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # ✅ IMPORTANT: SEND RAW IMAGE BYTES (NO BASE64, NO RESIZE)
    image_bytes = uploaded_file.getvalue()

    # Roboflow Detect API (RAPID)
    url = f"https://detect.roboflow.com/{MODEL_NAME}/{MODEL_VERSION}?api_key={API_KEY}"

    response = requests.post(
        url,
        files={"file": image_bytes},
        timeout=60
    )

    if response.status_code != 200:
        st.error("❌ Roboflow API request failed")
        st.write("Status Code:", response.status_code)
        st.code(response.text)
        st.stop()

    result = response.json()

    predictions = result.get("predictions", [])

    draw = ImageDraw.Draw(image)
    car_count = 0

    for pred in predictions:
        # Safety check
        if not isinstance(pred, dict):
            continue

        if not all(k in pred for k in ("x", "y", "width", "height")):
            continue

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        conf = pred.get("confidence", 0)

        # Convert center → bounding box
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Car {conf:.2f}", fill="red")

        car_count += 1

    if car_count == 0:
        st.warning("⚠️ No cars detected in this image")
    else:
        st.image(image, caption="Detected Cars", use_container_width=True)
        st.success(f"🚗 Cars Detected: {car_count}")
