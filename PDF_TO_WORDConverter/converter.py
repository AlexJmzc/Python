# Código que convierte un archivo PDF a Word
# Code that converts an PDF file to a Word file

# Instalar convertidor con el comando -> pip install pdf2docx
# Install the converter with the command -> pip install pdf2docx

from pdf2docx import Converter

def converter(pdf_path, docx_path):
    cv = Converter(pdf_path)

    cv.convert(docx_path, start=0, end=None)

    cv.close()

pdf_path = 'text.pdf'

docx_path = 'text.docx'

converter(pdf_path, docx_path)

