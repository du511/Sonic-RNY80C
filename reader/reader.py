import os
import fitz
import docx


class DocumentReader:

    @staticmethod
    def read_pdf(file_path):
        try:
            """
            从指定的PDF文件中提取所有页面的文本，并将其存储在TXT文件中。
            参数:
            file_path: str, PDF文件的路径。
            output_path: str, 输出的TXT文件的路径。
            """
            with fitz.open(file_path) as doc:
                # 初始化一个空字符串来收集文本
                full_text = ""
                # 遍历每一页
                for page in doc:
                    # 提取当前页面的文本并追加到full_text字符串
                    full_text += page.get_text()
                return full_text
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
