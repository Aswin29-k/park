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
st.write("Upload an image and detect cars using Roboflow")

# ---------------- ROB0FLOW CONFIG ----------------
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
MODEL = "find-cars"
VERSION = 1   # 🔴 change only if your Roboflow model version is different

if not API_KEY:
    st.error("❌ ROBOFLOW_API_KEY not found in Environment Variables")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and st.button("🔍 Run Detection"):

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Roboflow Detect API (RAPID)
    url = f"https://detect.roboflow.com/{MODEL}/{VERSION}?api_key={API_KEY}"

    response = requests.post(
        url,
        files={"file": uploaded_file.getvalue()},
        timeout=60
    )

    # ---- DEBUG (keep this until it works once) ----
    st.write("Status Code:", response.status_code)

    if response.status_code != 200:
        st.error("❌ Roboflow API request failed")
        st.code(response.text)
        st.stop()

    result = response.json()

    # ---------------- DRAW DETECTIONS ----------------
    draw = ImageDraw.Draw(image)
    predictions = result["predictions"]

    car_count = 0

    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        conf = pred["confidence"]

        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Car {conf:.2f}", fill="red")

        car_count += 1

    st.image(image, caption="Detected Cars", use_container_width=True)
    st.success(f"🚗 Cars Detected: {car_count}")
