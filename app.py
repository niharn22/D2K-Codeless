import streamlit as st
import requests
from PIL import Image
import io

# Function to make a request to the Flask endpoint for object detection
def detect_objects(image):
    files = {'image': image}
    response = requests.post('http://localhost:5000/detect_objects', files=files)
    return response.content

def main():
    st.title('Object Detection with TensorFlow - Team Codeless')

    # Upload image
    uploaded_image = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection and get the image with bounding boxes
        image_bytes = detect_objects(uploaded_image)
        detected_image = Image.open(io.BytesIO(image_bytes))
        st.image(detected_image, caption='Detected Objects', use_column_width=True)

if __name__ == '__main__':
    main()
