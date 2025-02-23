import os
import fitz  # PyMuPDF
import docx


class DocumentReader:

    @staticmethod
    def read_pdf(file_path):
        try:
            doc = fitz.open(file_path)  # 打开PDF文件
            text = ""
            for page_num in range(len(doc)):  # 遍历每一页
                page = doc.load_page(page_num)  # 加载当前页面
                text += page.get_text()  # 提取文本内容
            doc.close()  # 关闭文档
            return text
        except Exception as e:
            print(f"Error while reading pdf file: {e}")
            return ""

    @staticmethod
    def read_docx(file_path):
        try:
            doc = docx.Document(file_path)  # 打开Word文档
            text = ""
            for para in doc.paragraphs:  # 遍历每个段落
                text += para.text + "\n"  # 添加段落内容并换行
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