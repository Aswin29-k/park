import streamlit as st
from PIL import Image, ImageDraw
import base64
import io
import os
import requests

st.set_page_config(page_title="Smart Parking Detection")
st.title("🚗 Smart Parking Detection")

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = "aswin-gdjej"
WORKFLOW_ID = "find-cars"

if not API_KEY:
    st.error("❌ ROBOFLOW_API_KEY not found")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and st.button("🔍 Run Detection"):

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    url = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"

    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": img_base64
            }
        }
    }

    response = requests.post(url, json=payload, timeout=60)

    if response.status_code != 200:
        st.error("❌ Roboflow API request failed")
        st.code(response.text)
        st.stop()

    result = response.json()

    # ---------- SAFE EXTRACTION ----------
    outputs = result.get("outputs", [])
    if not outputs:
        st.warning("No outputs from workflow")
        st.json(result)
        st.stop()

    predictions = outputs[0].get("predictions", [])

    # Force predictions into list
    if isinstance(predictions, dict):
        predictions = [predictions]
    elif isinstance(predictions, str):
        st.warning("Workflow returned label only, not bounding boxes")
        st.write(predictions)
        st.stop()
    elif not isinstance(predictions, list):
        st.warning("Unknown prediction format")
        st.json(predictions)
        st.stop()

    draw = ImageDraw.Draw(image)
    car_count = 0

    for pred in predictions:

        # 🔒 HARD SAFETY CHECK
        if not isinstance(pred, dict):
            continue

        if not all(k in pred for k in ("x", "y", "width", "height")):
            continue

        x = float(pred["x"])
        y = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
        conf = float(pred.get("confidence", 0))

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
