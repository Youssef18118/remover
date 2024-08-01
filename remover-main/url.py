from flask import Flask, render_template, request, send_file, url_for
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import os
import time
import shutil

app = Flask(__name__)

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch', methods=['POST'])
def fetch():
    url = request.form['url']
    selected_size = request.form['size']

    # Clear and create directories
    clear_folder('static/screenshots')
    clear_folder('static/cropped')
    clear_folder('static/enlarged')

    # Remove specific files if they exist
    remove_file('static/export.html')
    remove_file('static/stitched_image.png')

    # Selenium settings for mobile view
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=375,812")  # Mobile size
    options.add_argument("--headless")
    options.add_argument("user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(3)

    # List of query selectors to capture sections, excluding specific styles and classes
    sections = [
        'section.component:not(.cart-slider-wrap):not(.KRC0025.stage-medium.bg-white):not(.KRC0004.stage-wide.txt-img.bg-white):not(.KRC0025.stage-full.bg-white):not(.KRC1200.stage-medium.full-size):not(.KRC0022.bg-white):not(.KRP1109)',
    ]

    screenshots = []
    section_count = 0

    for selector in sections:
        elements = driver.execute_script(f"return document.querySelectorAll('{selector}')")
        print(f"Found {len(elements)} elements for selector: {selector}")

        for element in elements:
            try:
                if element:
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    time.sleep(2)  # Wait for the section to come into view
                    screenshot_path = os.path.join("static/screenshots", f"screenshot_section_{section_count}.png")
                    element.screenshot(screenshot_path)
                    screenshots.append(screenshot_path)
                    section_count += 1
                    print(f"Captured screenshot for element with selector: {selector}")
                else:
                    print(f"Element not found for selector: {selector}")
            except Exception as e:
                print(f"Error capturing element with selector {selector}: {e}")

    driver.quit()

    if not screenshots:
        print("No screenshots captured.")
        return "No screenshots captured.", 500

    # Crop and resize images
    cropped_screenshots = []
    enlarged_screenshots = []
    crop_width = 350  # Adjusted width to ensure scrollbar is hidden

    for screenshot in screenshots:
        with Image.open(screenshot) as img:
            cropped_img = img.crop((0, 0, crop_width, img.height))
            cropped_screenshot_path = os.path.join("static/cropped", f"cropped_{os.path.basename(screenshot)}")
            cropped_img.save(cropped_screenshot_path)
            cropped_screenshots.append(f"cropped_{os.path.basename(screenshot)}")

            # Calculate the new height while maintaining the aspect ratio
            new_height = int(860 * img.height / crop_width)

            # Perform the resize operation and ensure it is saved correctly
            enlarged_img = cropped_img.resize((860, new_height), Image.LANCZOS)
            enlarged_screenshot_path = os.path.join("static/enlarged", f"enlarged_{os.path.basename(screenshot)}")
            enlarged_img.save(enlarged_screenshot_path)

            # Re-open the saved image to verify the size
            with Image.open(enlarged_screenshot_path) as check_img:
                enlarged_screenshots.append(f"enlarged_{os.path.basename(screenshot)}")

    # Stitch enlarged screenshots together
    images = [Image.open(os.path.join("static/enlarged", screenshot)) for screenshot in enlarged_screenshots]
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    total_height = sum(heights)

    stitched_image = Image.new('RGB', (total_width, total_height))
    y_offset = 0
    for img in images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height

    stitched_image_path = os.path.join("static", "stitched_image.png")
    stitched_image.save(stitched_image_path)

    return render_template('result.html', screenshots=enlarged_screenshots, stitched_image=stitched_image_path)

@app.route('/export_image/<filename>')
def export_image(filename):
    path = os.path.join("static", filename)
    return send_file(path, as_attachment=True)

@app.route('/export_html')
def export_html():
    screenshots = os.listdir('static/enlarged')
    absolute_screenshot_paths = [os.path.abspath(os.path.join('static/enlarged', screenshot)) for screenshot in screenshots]
    html_content = render_template('export.html', screenshots=absolute_screenshot_paths)
    export_path = os.path.abspath("static/export.html")
    with open(export_path, "w") as file:
        file.write(html_content)
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
