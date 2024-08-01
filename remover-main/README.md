
# Text Removal App

This Streamlit application uses LaMa inpainting model to remove text or unwanted elements from your images. 

## Features

* **Image Upload:** Upload an image in PNG, JPG, or JPEG format.
* **Interactive Canvas:** Draw over the text you want to remove using a customizable brush.
* **AI Processing:** Process the image using LaMa inpainting model
* **Download Processed Image:** Save the cleaned image in PNG format.
* **Image Comparison:** View a side-by-side comparison of the original and processed images.
* **Edit Again:** Re-edit the processed image to further refine the results.

## How to Use

1. **Upload an Image:** Click the "Choose image" button and select an image file.
2. **Adjust Image Size:** Use the "Max image size" slider to resize the image for optimal processing.
3. **Configure Brush Settings:** Customize the brush size and color using the sliders and picker.
4. **Draw Over Text:** Draw over the text or elements you want to remove on the canvas.
5. **Process the Image:** Click the "Remove Text" button to initiate the AI processing.
6. **Download Processed Image:** Click the "Download Processed Image" button to save the cleaned image.
7. **Edit Again:** Click the "Edit Again" button to continue editing the processed image.

## Installation

1. **Install Required Libraries:** 
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App:**
   ```bash
   streamlit run main.py
   ```

## Notes

* The processing time may vary depending on the size and complexity of the image.
* For optimal results, try drawing a clear and defined outline around the text or unwanted elements.
* If the initial result isn't satisfactory, you can edit the image again or re-upload it to further refine the results.

## Dependencies

* Streamlit
* NumPy
* Pandas
* Pillow
* Streamlit-drawable-canvas

## Contributing

Contributions are welcome! Please feel free to fork the repository and submit pull requests.
