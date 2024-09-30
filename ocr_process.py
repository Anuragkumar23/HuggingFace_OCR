import cv2
import pytesseract
import numpy as np

# Save the image for debugging purposes
def save_debug_image(image, filename):
    cv2.imwrite(filename, image)

# Preprocess the image for segmentation
def preprocess_image_for_segmentation(img):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    
    # Apply adaptive thresholding to create a binary image
    _, binary_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary_img

# Extract text from the OCR image (directly from an image object, no paths)
def extract_text_from_image(img):
    # Optionally resize the image for better OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Preprocess image
    preprocessed_img = preprocess_image_for_segmentation(img)

    # Save the preprocessed image for debugging
    save_debug_image(preprocessed_img, "full_preprocessed_image.jpg")

    # Run OCR on the full preprocessed image (no block segmentation)
    extracted_text = pytesseract.image_to_string(preprocessed_img, lang='eng+hin')

    return extracted_text
