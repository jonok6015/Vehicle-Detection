import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Vehicle Detection",
    layout="centered"
)

st.title("🚗 Vehicle Detection (YOLOv11)")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure best.pt is in same folder

model = load_model()

# ---------------- Upload Image ----------------
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Vehicles"):

        with st.spinner("Running detection..."):

            results = model.predict(
                source=img_array,
                conf=0.25,
                device="cpu"   # Use "cuda" if you have GPU
            )

        st.success("Detection complete!")

        # Show detection result
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Result", use_container_width=True)

        # Count detected vehicles
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:

            class_ids = boxes.cls.tolist()
            class_names = [model.names[int(cls)] for cls in class_ids]
            counts = Counter(class_names)

            st.subheader("📦 Detected Vehicles")
            st.write(f"🔢 **Total vehicles detected:** {sum(counts.values())}")

            for name, count in counts.items():
                st.write(f"- **{name}**: {count}")
        else:
            st.warning("No vehicles detected.")
