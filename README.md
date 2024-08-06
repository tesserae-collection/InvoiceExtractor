# Invoice/Receipt Extraction App

This application is designed to extract structured data from invoices and receipts. It uses a combination of OCR (Optical Character Recognition) and machine learning models to process the input files.

### `Note`: I have not retrained these models yet, they are taken as barebone models. I need to fine tune these models with a dataset(which I don't have yet). I will also write some adapters from google invoice parser so anyone can gather some data and will try to provide some training scripts so anyone can finetune their own models. 

## Features

- Upload an image or PDF file
- Process the file with or without OCR
- Display the input file and the extracted data in a user-friendly interface

## Technologies Used

- Python
- Streamlit for the web interface
- PyTorch for machine learning
- PaddleOCR for OCR
- PyMuPDF for PDF processing
- PIL for image processing

## Installation

1. Clone the repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash 
streamlit run app.py

Then, open your web browser to http://localhost:8501 to use the app.
