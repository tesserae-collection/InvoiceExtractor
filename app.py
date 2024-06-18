import base64
from builtins import enumerate

from donut.donutvllm import DonutVLLM

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pymupdf

class App:
    def __init__(self):
        self.file = None

    def get_file(self):
        self.file = st.sidebar.file_uploader("Upload an image or PDF", type=['png', 'jpg', 'pdf'])

    def process_file(self):
        if self.file is not None:
           self.process_image()

    def convert_pdf(self):
        doc = pymupdf.open(self.file)
        text = ""
        for page_no in range(doc.page_count):
            page = doc.load_page(page_no)
            text = page.get_text("blocks")
        st.text_area(text)

    def display_pdf(self):
        pdf_file = self.file.read()
        b64 = base64.b64encode(pdf_file).decode()  # some strings <-> bytes conversions necessary here
        href = f'<left><iframe src="data:application/pdf;base64,{b64}" width="500" height="800"></iframe></left>'
        st.markdown(href, unsafe_allow_html=True)

    def display_image(self):
        image = Image.open(self.file)
        st.image(image)

    def process_image(self):
        noocr_model = DonutVLLM()
        result = noocr_model.generate_output_json(self.file)
        return result

    def run(self):
        self.get_file()
        self.display_pdf()
        self.process_file()


# def main():


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    col1, col2 = st.columns(2)
    app = App()
    app.get_file()
    # app.file = open("/home/extravolatile/Downloads/01_Gradient+Boosting+Techniques+for+Credit+card+Fraud+detection-final+(2).pdf", "rb")
    with col1:
        app.display_image()
    with col2:
        result = app.process_image()
        st.json(result)


