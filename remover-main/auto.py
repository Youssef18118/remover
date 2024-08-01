import numpy as np
import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from io import BytesIO
import easyocr
from src.core import process_inpaint
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import mode


reader = easyocr.Reader(['en', 'ko'])  

FONTS = {
    'Thin': 'Fonts/NotoSansKR-Thin.otf',
    'ExtraLight': 'Fonts/NotoSansKR-ExtraLight.otf',
    'Light': 'Fonts/NotoSansKR-Light.otf',
    'Regular': 'Fonts/NotoSansKR-Regular.otf',
    'Medium': 'Fonts/NotoSansKR-Medium.otf',
    'SemiBold': 'Fonts/NotoSansKR-SemiBold.otf',
    'Bold': 'Fonts/NotoSansKR-Bold.otf',
    'ExtraBold': 'Fonts/NotoSansKR-ExtraBold.otf',
    'Black': 'Fonts/NotoSansKR-Black.otf'
}

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


def get_text_color(image, bbox):
    x, y, w, h = bbox
    region = np.array(image.crop((x, y, x+w, y+h)))
    
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    
    average_brightness = np.mean(gray)
    
    return (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

def estimate_font_weight(image, bbox):
    x, y, w, h = bbox
    region = image.crop((x, y, x+w, y+h))
    region_gray = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    black_pixel_ratio = np.sum(binary == 255) / binary.size
    
    if black_pixel_ratio < 0.1:
        return 'Thin'
    elif black_pixel_ratio < 0.15:
        return 'ExtraLight'
    elif black_pixel_ratio < 0.2:
        return 'Light'
    elif black_pixel_ratio < 0.25:
        return 'Regular'
    elif black_pixel_ratio < 0.3:
        return 'Medium'
    elif black_pixel_ratio < 0.35:
        return 'SemiBold'
    elif black_pixel_ratio < 0.4:
        return 'Bold'
    elif black_pixel_ratio < 0.45:
        return 'ExtraBold'
    else:
        return 'Black'
    

def improved_text_detection(image, min_confidence=0.5):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image)
    results = reader.readtext(img_np)
    
    text_areas = []
    for (bbox, text, prob) in results:
        if prob > min_confidence:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x = int(min(top_left[0], bottom_left[0]))
            y = int(min(top_left[1], top_right[1]))
            w = int(max(top_right[0], bottom_right[0]) - x)
            h = int(max(bottom_left[1], bottom_right[1]) - y)
            
            color = get_text_color(image, (x, y, w, h))
            font_weight = estimate_font_weight(image, (x, y, w, h))
            text_areas.append((x, y, x+w, y+h, text, color, font_weight))
    
    return text_areas


def create_text_mask(image, text_areas):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    for box in text_areas:
        draw.rectangle(box[:4], fill=255)
    return mask


def replace_text(image, text_areas, new_texts, colors):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    draw = ImageDraw.Draw(image)
    
    for (x, y, x2, y2, original_text, _, font_weight), new_text, color in zip(text_areas, new_texts, colors):
        font_size = int((y2 - y) * 0.8) 
        font_path = FONTS.get(font_weight, FONTS['Regular'])
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            st.error(f"Error loading font {font_path}: {str(e)}. Using default font.")
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x, y), new_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        while text_width > (x2 - x) or text_height > (y2 - y):
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((x, y), new_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        
        draw.text((x, y), new_text, font=font, fill=color)
    
    return image

def check_font_files():
    missing_fonts = []
    for weight, path in FONTS.items():
        if not os.path.exists(path):
            missing_fonts.append(f"{weight}: {path}")
    
    if missing_fonts:
        st.warning("The following font files are missing:")
        for font in missing_fonts:
            st.write(font)
        st.write("Please ensure all font files are in the correct location.")
    else:
        st.success("All font files are present.")

st.title("SIMPLIFIED TEXT REPLACEMENT APP (EasyOCR)")

st.markdown(
    """
    Welcome to the Simplified Text Replacement App! This application uses EasyOCR to accurately detect and remove text from your images, then replace it with editable text while maintaining the original position and color.
    
    How to use:
    1. Upload an image
    2. Adjust the confidence threshold if needed
    3. Click 'Detect and Remove Text' to process the image
    4. Edit the detected text as needed
    5. Click 'Replace Text' to add the new text to the image
    6. Download the result
    """
)

check_font_files()

uploaded_file = st.file_uploader("Choose image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        img_input = Image.open(BytesIO(bytes_data)).convert("RGBA")

        st.subheader("Image Size Settings")
        max_size = st.slider("Max image size", 500, 2000, 2000, 100, 
                             help="Larger sizes may increase processing time but can improve quality.")
        img_input = resize_image(img_input, max_size)

        st.image(img_input, caption="Original Image", use_column_width=True)

        min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.5, 0.1,
                                   help="Increase to detect only high-confidence text areas.")

        if st.button('Detect and Remove Text'):
            with st.spinner("AI is detecting and removing text..."):
                text_areas = improved_text_detection(img_input, min_confidence)
                
                img_with_boxes = img_input.copy()
                draw = ImageDraw.Draw(img_with_boxes)
                for box in text_areas:
                    draw.rectangle(box[:4], outline="red", width=2)
                st.image(img_with_boxes, caption="Detected Text Areas", use_column_width=True)
                
                mask = create_text_mask(img_input, text_areas)
                
                img_np = np.array(img_input)
                mask_np = np.array(mask)
                
                mask_rgba = np.zeros(img_np.shape, dtype=np.uint8)
                mask_rgba[:,:,3] = 255 - mask_np
                
                img_output = process_inpaint(img_np, mask_rgba)
                img_output = Image.fromarray(img_output)
                
            st.success("Text removal complete!")
            st.image(img_output, caption="Processed Image", use_column_width=True)
            
            st.session_state.text_areas = text_areas
            st.session_state.img_output = img_output

    except UnidentifiedImageError:
        st.error("Error: The uploaded file could not be identified as an image. Please make sure you're uploading a valid image file (PNG, JPG, or JPEG).")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}. Please try uploading a different image or contact support if the problem persists.")
        st.exception(e)

if 'text_areas' in st.session_state and 'img_output' in st.session_state:
    st.subheader("Text Replacement")
    
    new_texts = []
    colors = []
    for i, (x, y, x2, y2, text, auto_color, font_weight) in enumerate(st.session_state.text_areas):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_text = st.text_input(f"Edit text {i+1}", value=text, key=f"text_{i}")
        with col2:
            color = st.selectbox(f"Color for text {i+1}", ["Black", "White"], key=f"color_{i}")
        
        new_texts.append(new_text)
        colors.append((0, 0, 0) if color == "Black" else (255, 255, 255))
    
    if st.button('Replace Text'):
        img_with_new_text = replace_text(st.session_state.img_output.copy(), st.session_state.text_areas, new_texts, colors)
        st.image(img_with_new_text, caption="Image with Replaced Text", use_column_width=True)
        
        uploaded_name = os.path.splitext(uploaded_file.name)[0]
        image_download_button(
            pil_image=img_with_new_text,
            filename=f"{uploaded_name}_text_replaced",
            fmt="png",
            label="Download Processed Image"
        )

st.markdown("---")
st.info("This app uses EasyOCR for text detection, an AI-based inpainting model for text removal, and allows you to choose between black and white text for each detected text area individually.")