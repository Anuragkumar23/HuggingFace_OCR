# OCR Segmentation and Text Extraction (English + Hindi)
This project is designed to perform Optical Character Recognition (OCR) on images containing text in both English and Hindi. The script processes the image by segmenting it into blocks, enhancing the quality for better text extraction, and finally extracting the text using Tesseract OCR.



---
title: Ocr Image File Processing[huggingface_app]
emoji: 💻
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---


## Features
- **Image Loading**: Loads images from the specified path.
- **Preprocessing**: Converts images to grayscale, applies CLAHE for contrast enhancement, and performs adaptive thresholding.
- **Segmentation**: Segments the image into blocks based on contours.
- **OCR Extraction**: Uses Tesseract to extract text from the segmented blocks for both English and Hindi languages.

## Requirements
- Python 3.x
- OpenCV
- Pytesseract
- NumPy
- stremlit
- streamlit
- PyPDF2
- streamlit_pdf_viewer
- langchain
- transformers
- ctransformers
- langchain==0.3.0
- langchain-community==0.3.0
- langchain-core==0.3.0
- langchain-text-splitters==0.3.0
- langsmith==0.1.121
- Pillow
- PyMuPDF
- sentence-transformers
- faiss-cpu
- numpy

  # APP sample at Huggingface_hub[https://huggingface.co/spaces/kumarAnurag/ocr_image_file_processing]
  ![Screenshot 2024-09-30 204441](https://github.com/user-attachments/assets/726f762a-342c-4bf8-af18-ad9cde9be204)


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
