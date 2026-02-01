import streamlit as st
from PIL import Image, ImageDraw
import base64
import io
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Parking Detection")

st.title("🚗 Smart Parking Detection")
st.write("Upload an image and detect cars using Roboflow Workflow")

# ---------------- ROB0FLOW CONFIG ----------------
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = "aswin-gdjej"
WORKFLOW_ID = "find-cars"

if not API_KEY:
    st.error("❌ ROBOFLOW_API_KEY not found in Environment Variables")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and st.button("🔍 Run Detection"):

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Workflow API URL
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

    # ---------------- SAFE PARSING ----------------
    outputs = result.get("outputs", [])

    if not outputs:
        st.warning("⚠️ No outputs returned from workflow")
        st.json(result)
        st.stop()

    output_block = outputs[0].get("output", {})
    predictions = output_block.get("predictions", [])

    draw = ImageDraw.Draw(image)
    car_count = 0
    boxes_found = False

    # If predictions is a STRING (classification)
    if isinstance(predictions, str):
        st.warning("⚠️ Workflow returned label only (no bounding boxes)")
        st.write("Prediction:", predictions)
        st.stop()

    # If predictions is NOT list
    if not isinstance(predictions, list):
        st.warning("⚠️ Unexpected prediction format")
        st.json(predictions)
        st.stop()

    # Iterate safely
    for pred in predictions:

        # Skip if not dict
        if not isinstance(pred, dict):
            continue

        x = pred.get("x")
        y = pred.get("y")
        w = pred.get("width")
        h = pred.get("height")
        conf = pred.get("confidence", 0)

        # Skip invalid detections
        if None in (x, y, w, h):
            continue

        boxes_found = True

        # Convert center → box
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Car {conf:.2f}", fill="red")

        car_count += 1

    # ---------------- DISPLAY RESULT ----------------
    if boxes_found:
        st.image(image, caption="Detected Cars", use_container_width=True)
        st.success(f"🚗 Cars Detected: {car_count}")
    else:
        st.warning("⚠️ No bounding boxes found in workflow output")
        st.json(predictions)
