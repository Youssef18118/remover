from flask import Flask, render_template, request, send_file, jsonify
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from io import BytesIO
import easyocr
import numpy as np
import cv2
import base64
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
from src.core import process_inpaint

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'ko'])

# Define available fonts
FONTS = {
    'Thin': 'Fonts/NotoSansKR-Thin.ttf',
    'ExtraLight': 'Fonts/NotoSansKR-ExtraLight.ttf',
    'Light': 'Fonts/NotoSansKR-Light.ttf',
    'Regular': 'Fonts/NotoSansKR-Regular.ttf',
    'Medium': 'Fonts/NotoSansKR-Medium.ttf',
    'SemiBold': 'Fonts/NotoSansKR-SemiBold.ttf',
    'Bold': 'Fonts/NotoSansKR-Bold.ttf',
    'ExtraBold': 'Fonts/NotoSansKR-ExtraBold.ttf',
    'Black': 'Fonts/NotoSansKR-Black.ttf'
}

def setup_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=860,10000")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def fetch_product_info(url):
    driver = setup_chrome_driver()
    try:
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(5)
        
        # Scroll to load all content
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Get the entire page content
        html_content = driver.page_source
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Process all images in the page
        for img in soup.find_all('img'):
            img_url = img.get('src', '')
            if img_url and not img_url.startswith('data:'):
                full_img_url = requests.compat.urljoin(url, img_url)
                try:
                    response = requests.get(full_img_url, timeout=10)
                    if response.status_code == 200:
                        img_data = base64.b64encode(response.content).decode('utf-8')
                        img['src'] = f"data:image/png;base64,{img_data}"
                except requests.exceptions.RequestException:
                    img['src'] = full_img_url
        
        return str(soup)
    finally:
        driver.quit()

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
    
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    
    # Calculate average brightness
    average_brightness = np.mean(gray)
    
    # Choose black or white based on background brightness
    return (0, 0, 0) if average_brightness > 127 else (255, 255, 255)

def estimate_font_weight(image, bbox):
    x, y, w, h = bbox
    region = image.crop((x, y, x+w, y+h))
    region_gray = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to separate text from background
    _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate the ratio of black (text) pixels to total pixels
    black_pixel_ratio = np.sum(binary == 255) / binary.size
    
    # Determine font weight based on the ratio
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
        font_size = int((y2 - y) * 0.8)  # Estimate font size based on box height
        font_path = FONTS.get(font_weight, FONTS['Regular'])
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font {font_path}: {str(e)}. Using default font.")
            font = ImageFont.load_default()
        
        # Calculate text position to maintain original coordinates
        text_bbox = draw.textbbox((x, y), new_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Adjust font size if text is too large
        while text_width > (x2 - x) or text_height > (y2 - y):
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((x, y), new_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        
        # Draw new text directly on the image at the original position
        draw.text((x, y), new_text, font=font, fill=color)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_info', methods=['POST'])
def fetch_info():
    url = request.form['url']
    html_content = fetch_product_info(url)
    return jsonify({'html_content': html_content})

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGBA")
    
    max_size = int(request.form.get('max_size', 2000))
    min_confidence = float(request.form.get('min_confidence', 0.5))
    
    image = resize_image(image, max_size)
    
    text_areas = improved_text_detection(image, min_confidence)
    
    mask = create_text_mask(image, text_areas)
    
    img_np = np.array(image)
    mask_np = np.array(mask)
    
    mask_rgba = np.zeros(img_np.shape, dtype=np.uint8)
    mask_rgba[:,:,3] = 255 - mask_np
    
    img_output = process_inpaint(img_np, mask_rgba)
    img_output = Image.fromarray(img_output)
    
    # Convert image to base64 for displaying in HTML
    buffered = BytesIO()
    img_output.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        'image': img_str,
        'text_areas': text_areas,
        'original_image': base64.b64encode(image.tobytes()).decode()
    })

@app.route('/update_image', methods=['POST'])
def update_image():
    try:
        if 'image' not in request.form:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = request.form['image']
        if not image_data.startswith('data:image'):
            return jsonify({'error': 'Invalid image data format'}), 400
        
        image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))
        
        if 'text_areas' not in request.form:
            return jsonify({'error': 'No text areas provided'}), 400
        text_areas = json.loads(request.form['text_areas'])
        
        if 'new_texts' not in request.form:
            return jsonify({'error': 'No new texts provided'}), 400
        new_texts = json.loads(request.form['new_texts'])
        
        if 'colors' not in request.form:
            return jsonify({'error': 'No colors provided'}), 400
        colors = json.loads(request.form['colors'])
        
        updated_image = replace_text(image, text_areas, new_texts, colors)
        
        buffered = BytesIO()
        updated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'updated_image': img_str})
    except json.JSONDecodeError as e:
        return jsonify({'error': f'JSON decoding error: {str(e)}'}), 400
    except UnidentifiedImageError as e:
        return jsonify({'error': f'Image identification error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)