import jieba
import re

class KeywordDetector:
    def __init__(self, file_chinese_stopwords, file_key_words):
        self.file_chinese_stopwords = file_chinese_stopwords
        self.file_key_words = file_key_words

    def keyword_detection(self):
        try:
            #筛选出中文停用词
            with open(self.file_chinese_stopwords, 'r', encoding='utf-8') as f:
                chinese_stop_words = [line.strip() for line in f.readlines()]
            #读取关键词
            with open(self.file_key_words, 'r', encoding='utf-8') as f:
                text = re.sub(r'\s+','', f.read().strip())
                words = jieba.lcut(text)
            filter_words = []
            for word in words:
                if word not in chinese_stop_words:
                    filter_words.append(word)
            #转为集合
            security_words = set(filter_words)
            return security_words
        except FileNotFoundError:
                print("文件未找到，请检查文件路径。")
                return None
        except UnicodeDecodeError:
            print("文件编码可能存在问题，请检查文件编码。")
            return None
        except Exception as e:
            print(f"发生未知错误: {e}")
            return None





               