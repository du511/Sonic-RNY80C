import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
import re
import jieba

class NaiveBayesClassifier:
    
    # 初始化贝叶斯以及SMOTE过采样模型
    def __init__(self):
        #引入停止词,便于处理文本
        try:
            with open('./naive_bayes_model/stop_words/hit_stopwords.txt', 'r', encoding='utf-8') as f:
                stop_words = f.read()
                stop_words_list = []
                for word in jieba.lcut(re.sub(r'[^\w\s]', '', stop_words).lower().strip()):
                    stop_words_list.append(word)
                stop_words_list = list(set(stop_words_list))
                self.vectorizer = TfidfVectorizer(stop_words = stop_words_list)
        except FileNotFoundError:
            print("停止词文件不存在，请检查路径")
        self.clf = MultinomialNB()
        self.sm = SMOTE()

    # 训练模型以及使用SMOTE过采样
    def train(self, train_data):
        clean_text = []
        texts, labels = zip(*train_data)
        for text in texts:
            text = jieba.lcut(re.sub(r'[^\w\s]', '', text).lower().strip())
            text = [element for element in text if element != '\t']#筛除\t
            # print(f"这是刚分词后的数据:{text}")
            clean_text.append(text)
        new_list = [" ".join(sublist) for sublist in clean_text]
        X = self.vectorizer.fit_transform(new_list)
        Y =labels = np.array(labels)
        X_res, Y_res = self.sm.fit_resample(X, Y)
        self.clf.fit(X_res, Y_res)

    # 保存模型
    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.clf, model_path)

    # 加载模型
    def load_model(self, model_path, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(model_path)
    
    # 预测标签
    def predict(self, input_text):
        if self.clf is None or self.vectorizer is None:
            print("模型或特征提取器未加载，请先调用 load_model 方法")
        try:
            input_text = list(jieba.lcut(re.sub(r'[^\w\s]', '', input_text).lower().strip()))
            question_vector = self.vectorizer.transform(input_text)
            predicted_label = self.clf.predict(question_vector)[0]
            return predicted_label
        
        except Exception as e:
            print("预测出错：", e)
            return None
