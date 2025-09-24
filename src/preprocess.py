import cv2
import numpy as np

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance contrast
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Save temporary image
    temp_path = "temp_preprocessed.jpg"
    cv2.imwrite(temp_path, thresh)
    return temp_path