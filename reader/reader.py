import os
import PyPDF2
import docx


class DocumentReader:

    @staticmethod
    def read_pdf(file_path):
        try:
            with open(file_path, "rb") as f:
                text = ""
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error while reading pdf file: {e}")
            return ""

    @staticmethod
    def read_docx(file_path):
        try:
            with open(file_path, "rb") as f:
                text = ""
                doc = docx.Document(f)
                for para in doc.paragraphs:
                    for run in para.runs:
                        text += run.text
                return text
        except Exception as e:
            print(f"Error while reading docx file: {e}")
            return ""

    @staticmethod
    def read_txt(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                return text
        except Exception as e:
            print(f"Error while reading txt file: {e}")
            return ""

    @staticmethod
    def read_file(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            return DocumentReader.read_pdf(file_path)
        elif file_extension == ".docx":
            return DocumentReader.read_docx(file_path)
        elif file_extension == ".txt":
            return DocumentReader.read_txt(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return ""
