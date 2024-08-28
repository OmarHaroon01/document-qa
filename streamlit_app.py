import streamlit as st
import cohere
import pytesseract
import os

from tempfile import NamedTemporaryFile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def apply_ocr_to_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if len(text) < 100:
        text = apply_ocr_to_pdf(pdf_path)
    return text

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – Cohere will answer! "
    "To use this app, you need to provide an Cohere API key, which you can get [here](https://dashboard.cohere.com/api-keys). "
)


cohere_api_key = st.text_input("Cohere API Key", type="password")
if not cohere_api_key:
    st.info("Please add your Cohere API key to continue.", icon="🗝️")
else:

    co = cohere.Client(api_key=cohere_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf)", type=("pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Give me all passenger detail along with the flight details.",
        disabled=not uploaded_file,
    )
    

    if uploaded_file and question:
        context = ""
        bytes_data = uploaded_file.read()
        with NamedTemporaryFile(delete=False) as tmp:  
            tmp.write(bytes_data)                      
            context = process_pdf(tmp.name)   
        os.remove(tmp.name)               

        
        # messages = [
        #     {
        #         "role": "user",
        #         "content": f"Here's a document: {document} \n\n---\n\n {question}",
        #     }
        # ]

        # # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     stream=True,
        # )
        response = co.chat(
          model="command-r-plus",
          message=
          """
          Given the following context:
    
          {context}
    
          Please answer the following question based on the above context:
    
          {question}
    
          Your answer should be concise and focus on the most important details. Make sure the answer is in JSON format. If you don't know a specific value, keep it empty. The dates should be specified like this: 25th January 2011.
          """.format(context=context, question=question)
        )

        # Stream the response to the app using `st.write_stream`.
        st.write(response.text)
