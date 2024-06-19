import base64

from donut.donutvllm import DonutVLLM

import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes

from ocr.paddleocr import PaddleOCRWrapper
from ocr_extraction_models.MIstral7b_4bit import FineTunedMistral7B


class App:
    def __init__(self, file):
        self.file = file

    def process_file(self):
        if self.file is not None:
            self.process_image()

    def display_image(self):
        if hasattr(self.file, "read"):
            image = Image.open(self.file)
            self.file = image
        st.image(self.file)

    def process_image(self, process_type: str = "ocr"):
        if process_type == "no_ocr":
            model = DonutVLLM()
            result = model.generate_output_json(self.file)
        elif process_type == "ocr":
            model = FineTunedMistral7B()
            ocr = PaddleOCRWrapper()
            ocr_results = ocr.get_text_and_bounding_boxes_per_page(self.file)
            result = model.generate_output_json(ocr_results)
        elif process_type == "both":
            model1 = FineTunedMistral7B()
            ocr = PaddleOCRWrapper()
            ocr_results = ocr.get_text_and_bounding_boxes_per_page(self.file)
            result1 = model1.generate_output_json(ocr_results)
            model = DonutVLLM()
            result2 = model.generate_output_json(self.file)
            result = {"ocr": result1, "no_ocr": result2}
        else:
            raise ValueError("Invalid type")
        return result


def display_pdf(file):
    b64 = base64.b64encode(file).decode()  # some strings <-> bytes conversions necessary here
    href = f'<left><iframe src="data:application/pdf;base64,{b64}" width="500" height="800"></iframe></left>'
    st.markdown(href, unsafe_allow_html=True)


def convert_pdf(file):
    pages = convert_from_bytes(file, 500, fmt="jpeg")
    print(type(pages[0]))
    return pages[0]


def main():
    st.set_page_config(layout="wide")
    st.title("Invoice/Receipt Extraction App")

    file = st.sidebar.file_uploader("Upload an image or PDF", type=['png', 'jpg', 'pdf'])

    if file is None:
        st.info("Please upload an image or PDF file")
        st.stop()
    if file.type == "application/pdf":
        # file = open(file, 'rb')
        file = file.read()
        file = convert_pdf(file)
    ap = App(file)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.header("Input File")
        ap.display_image()
    option = st.sidebar.selectbox(
        "How would you like to have your document processes?",
        ("With OCR", "Without OCR", "Both"))

    process_button = st.sidebar.button("process", key="process_button")
    with col2:
        st.header("Output JSON")
        if process_button and option == "Without OCR":
            result = ap.process_image("no_ocr")
            st.json(result)
        elif process_button and option == "With OCR":
            result = ap.process_image("ocr")
            st.json(result)
        elif process_button and option == "Both":
            result = ap.process_image("both")
            st.json(result)


if __name__ == '__main__':
    main()
