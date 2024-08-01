import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from copy import deepcopy

from src.core import process_inpaint

def image_download_button(pil_image, filename: str, fmt: str, label="Download"):
    if fmt not in ["jpg", "png"]:
        raise Exception(f"Unknown image format (Available: {fmt} - case sensitive)")
    
    pil_format = "JPEG" if fmt == "jpg" else "PNG"
    file_format = "jpg" if fmt == "jpg" else "png"
    mime = "image/jpeg" if fmt == "jpg" else "image/png"
    
    buf = BytesIO()
    pil_image.save(buf, format=pil_format)
    
    return st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=f'{filename}.{file_format}',
        mime=mime,
    )

def resize_image(image, max_size):
    img_width, img_height = image.size
    if img_width > max_size or img_height > max_size:
        if img_width > img_height:
            new_width = max_size
            new_height = int((max_size / img_width) * img_height)
        else:
            new_height = max_size
            new_width = int((max_size / img_height) * img_width)
        return image.resize((new_width, new_height))
    return image

# Initialize session state variables
if "button_id" not in st.session_state:
    st.session_state["button_id"] = ""
if "color_to_label" not in st.session_state:
    st.session_state["color_to_label"] = {}
if 'reuse_image' not in st.session_state:
    st.session_state.reuse_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

def set_image(img):
    st.session_state.reuse_image = img

st.title("TEXT REMOVAL APP")

st.markdown(
    """
    Welcome to the Text Removal App! This application uses AI to remove text or other unwanted elements from your images.
    
    How to use:
    1. Upload an image
    2. Adjust the image size and brush settings
    3. Draw over the text or elements you want to remove
    4. Click 'Remove Text' to process the image
    """
)

uploaded_file = st.file_uploader("Choose image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        if st.session_state.reuse_image is not None:
            img_input = Image.fromarray(st.session_state.reuse_image)
        else:
            bytes_data = uploaded_file.getvalue()
            img_input = Image.open(BytesIO(bytes_data)).convert("RGBA")

        # Image resize options
        st.subheader("Image Size Settings")
        max_size = st.slider("Max image size", 500, 2000, 2000, 100, 
                             help="Larger sizes may increase processing time but can improve quality.")
        img_input = resize_image(img_input, max_size)

        # Brush settings
        st.subheader("Brush Settings")
        stroke_width = st.slider("Brush size", 1, 100, 50)
        stroke_color = st.color_picker("Brush color", "#FF00FF")

        st.write("**Now draw (brush) over the text or elements you want to remove.**")
        
        # Canvas size logic
        canvas_bg = deepcopy(img_input)
        aspect_ratio = canvas_bg.width / canvas_bg.height
        streamlit_width = 720
        
        # Max width is 720. Resize the height to maintain its aspect ratio.
        if canvas_bg.width > streamlit_width:
            canvas_bg = canvas_bg.resize((streamlit_width, int(streamlit_width / aspect_ratio)))
        
        canvas_result = st_canvas(
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            background_image=canvas_bg,
            width=canvas_bg.width,
            height=canvas_bg.height,
            drawing_mode="freedraw",
            key="canvas", 
        )
        
        if canvas_result.image_data is not None:
            im = np.array(Image.fromarray(canvas_result.image_data.astype(np.uint8)).resize(img_input.size))
            background = np.where(
                (im[:, :, 0] == 0) & 
                (im[:, :, 1] == 0) & 
                (im[:, :, 2] == 0)
            )
            drawing = np.where(
                (im[:, :, 0] == int(stroke_color[1:3], 16)) & 
                (im[:, :, 1] == int(stroke_color[3:5], 16)) & 
                (im[:, :, 2] == int(stroke_color[5:7], 16))
            )
            im[background] = [0,0,0,255]
            im[drawing] = [0,0,0,0]  # RGBA
            
            if st.button('Remove Text'):
                with st.spinner("AI is doing the magic!"):
                    output = process_inpaint(np.array(img_input), np.array(im))
                    img_output = Image.fromarray(output).convert("RGB")
                    st.session_state.processed_image = img_output
                
                st.success("AI has finished the job!")
                st.image(img_output, caption="Processed Image", use_column_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('Edit Again', on_click=set_image, args=(output,)):
                        st.experimental_rerun()
                
                with col2:
                    uploaded_name = os.path.splitext(uploaded_file.name)[0]
                    image_download_button(
                        pil_image=img_output,
                        filename=f"{uploaded_name}_processed",
                        fmt="png",
                        label="Download Processed Image"
                    )
                
                st.info("**TIP**: If the result is not perfect, you can edit again or download the image and reupload it to remove any remaining artifacts.")

        # Compare original and processed images
        if st.session_state.processed_image is not None:
            st.subheader("Compare Original and Processed Images")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_input, caption="Original Image", use_column_width=True)
            with col2:
                st.image(st.session_state.processed_image, caption="Processed Image", use_column_width=True)

    except UnidentifiedImageError:
        st.error("Error: The uploaded file could not be identified as an image. Please make sure you're uploading a valid image file (PNG, JPG, or JPEG).")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}. Please try uploading a different image or contact support if the problem persists.")
