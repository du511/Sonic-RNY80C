import os
import sys
import json
import toml
import jieba
import datetime
import sqlite3

from  reader.reader import DocumentReader#引入文档读取器
from RAG.faiss_indexer import FaissIndexer#引入Faiss索引器
from generator.embedding import Embedding#引入文本向量化器
from generator.response_generator import ResponseGenerator#引入回答生成器
from generator.chat_history_control import ControlChatHistoryData #引入对话历史记录控制系统
from filter.keyword_detection import KeywordDetector#引入关键词检测分类器
from naive_bayes_model.naive_bayes_classifier import NaiveBayesClassifier#引入朴素贝叶斯分类器
from naive_bayes_model.train_data.train_data import data#引入朴素贝叶斯训练数据
from generator.MyStreamingHandler import MyStreamingHandler#引入流式输出系统

#读取配置文件
config = toml.load("config/parameter.toml")
bert_uncased_model_name = config["bert"]["model_name"]
base_url = config["ChatOllama"]["base_url"]
local_model_name = config["ChatOllama"]["model_name"]
temperature = config["ChatOllama"]["temperature"]
top_p = config["ChatOllama"]["top_p"]
top_k = config["ChatOllama"]["top_k"]
bot_name = config["ChatOllama"]["project_name"]
logo = config["pattern"]["logo"]
welcome = config["pattern"]["welcome"]
help = config["help"]["help"]


#初始化langchain本地模型部署
from langchain_ollama import ChatOllama
model = ChatOllama(base_url = base_url, model = local_model_name,temperature = temperature, top_p = top_p, top_k = top_k, 
                   callbacks = [MyStreamingHandler()], streaming = True )

#确保日志存在,且创建日志目录以及日志文件
if not os.path.exists("./logs"):
    os.mkdir("logs")

def save_last_filename(filename):
        with open("logs/last_filename.json", "w") as f:
            json.dump({"filename": filename}, f)

def load_last_filename():
     if os.path.exists("logs/last_filename.json"):
         with open("logs/last_filename.json", "r") as f:
             data = json.load(f)
             return data.get("filename") 
     return None       

def main():
     #是否为调试模式
     if len(sys.argv) > 1 and sys.argv[1] == "-d":
          debug_mode = True
     else:
          debug_mode = False
     
      #读取上次使用的文档名
     last_filename = load_last_filename()

     if last_filename:
          use_last = input(f"是否需要使用上次的文档:{last_filename}? (y/n)")
          if use_last.lower() == "y":
               file_path = last_filename
          else:
               file_path = input("请输入文档路径: ")
               save_last_filename(file_path)
     else:
        file_path = input("请输入文档路径: ")
        save_last_filename(file_path)            
  
     #读取文档
     document_reader = DocumentReader()
     raw_text = document_reader.read_file(file_path)
     if not raw_text:
          return
    #  print(f"文档内容:\n{raw_text}")
    
     #分为段落
     paragraphs = raw_text.split("\n\n")

     #建立Faiss索引
     embedding_generator = Embedding(bert_uncased_model_name)
     faiss_indexer = FaissIndexer()
     index = faiss_indexer.build_index(paragraphs, embedding_generator)
     if not index:
          return
     
     #初始化对话历史记录
     answer_count = 0
     tick_count = 0

     #初始化关键词检测分类器
     keyword_set = KeywordDetector(file_chinese_stopwords = "docs/scu_stopwords.txt", file_key_words = "docs/key_words.txt")
     answer_generator = ResponseGenerator(model)

     #开始记录日志
     if debug_mode:
          log_file = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

     #朴素贝叶斯分类
     classifier = NaiveBayesClassifier()
     model_path = "./naive_bayes_model/saved_model/naive_bayes_model.pkl"
     vectorizer_path = "./naive_bayes_model/saved_vectorizer/naive_bayes_vectorizer.pkl"
     train_data_path = data

     #加载或训练模型
     if not os.path.exists(model_path) and not os.path.exists(vectorizer_path):
          classifier.train(train_data_path)
          classifier.save_model(model_path=model_path, vectorizer_path=vectorizer_path)
     else:
      classifier.load_model(model_path, vectorizer_path)

     #实例化对话历史记录控制系统
     history_control = ControlChatHistoryData()

     print(logo)

     while True:  # 主程序循环
          order = input(f"欢迎使用{bot_name}!，请输入指令: ")
          order_parts = list(order.split())  # 先将分割结果存储在变量中

          # 帮助指令
          if order.lower() == "-h":
               print("指令列表:\n")
               print(help)

          # 退出指令
          elif order.lower() == "q":
               break

          # 用户列表指令
          elif order.lower() == "-ls":
               users = history_control.list_user_ids()
               if users:
                    for user in users:
                         print("-" * 50)
                         print(user)
                    print("-" * 50)
               else:
                    print("当前没有用户,请先创建用户")

          # 用户删除指令
          elif len(order_parts) == 2 and order_parts[0] == "-d":
               user_id = order_parts[1]
               users = history_control.list_user_ids()
               if user_id in users:
                    history_control.delete_user_history(user_id)
                    print(f"用户{user_id}的历史记录已删除")
               else:
                    print(f"用户{user_id}不存在,请先创建用户")

          # 用户创建指令
          elif len(order_parts) == 2 and order_parts[0] == "-n":
               user_id = order_parts[1]
               users = history_control.list_user_ids()
               if user_id not in users:
                    history_control.create_new_user(user_id)
                    print(f"用户{user_id}已创建")
               else:
                    print(f"用户{user_id}已存在,请勿重复创建")

          # 会话列表指令
          elif len(order_parts) == 3 and order_parts[0] == "-l" and order_parts[2] == "-ls":
               user_id = order_parts[1]
               sessions = history_control.list_session_ids(user_id)
               if sessions:
                    for session in sessions:
                         print("-" * 50 + "\n")
                         print(session)
                    print("-" * 50 + "\n")
               else:
                    print(f"用户{user_id}没有会话记录,请先创建新会话")

          # 会话删除指令
          elif len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-d":
               user_id = order_parts[1]
               session_id = order_parts[3]
               sessions = history_control.list_session_ids(user_id)
               if session_id in sessions:
                    history_control.delete_session_history(user_id, session_id)
                    print(f"用户{user_id}的会话{session_id}已删除")
               else:
                    print(f"用户{user_id}的会话{session_id}不存在,请先创建会话")

          # 会话创建指令
          elif len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-n":
               user_id = order_parts[1]
               session_id = order_parts[3]
               sessions = history_control.list_session_ids(user_id)
               if session_id not in sessions:
                    history_control.create_new_session(user_id, session_id)
                    print(f"用户{user_id}的会话{session_id}已创建")
               else:
                    print(f"用户{user_id}的会话{session_id}已存在,请勿重复创建")

          # 进入会话指令
          elif len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-l":
               user_id = order_parts[1]
               session_id = order_parts[3]
               sessions = history_control.list_session_ids(user_id)
               if session_id in sessions:
                    print(welcome)
                    print(f"欢迎进入用户{user_id}的会话{session_id}，请输入指令: ")
                    while True:
                         # 读取用户输入
                         user_input = input("请输入你的问题(输入'q'退出): ")
                         if user_input.lower() == "q":
                              break

                         # 关键词分类问题类型
                         user_input_word = set(jieba.lcut(user_input))
                         judgment_outcome = bool(user_input_word & keyword_set.keyword_detection())

                         # 朴素贝叶斯分类预测问题类型
                         predicted_label = classifier.predict(user_input)

                         # 生成输入的向量
                         input_embedding = embedding_generator.get_embedding(user_input)
                         # 搜索索引
                         if input_embedding is None:
                              continue
                         if index is None:
                              print("索引未建立,无法搜索")
                              continue
                         unique_id = faiss_indexer.search_index(input_embedding, top_k=5)
                         # 对rag使用进行判断
                         if judgment_outcome:
                              if predicted_label == 1:
                               relevant_context = list(set([paragraphs[i] for i in unique_id]))
                              elif predicted_label == 0:
                               relevant_context = []
                         elif not judgment_outcome:
                              if predicted_label == 1:
                               relevant_context = list(set([paragraphs[i] for i in unique_id]))
                              elif predicted_label == 0:
                               relevant_context = []

                         if debug_mode:
                              print(f"预测标签:{predicted_label}")

                         # 获取回答以及提示词模板,回答生成处理在response_generator.py中
                         answer, template = answer_generator.generate_response(user_id, session_id, user_input, relevant_context, return_template=True)

                         # 更新对话历史

                         # 调试日志
                         if debug_mode:
                              tick_count += 1
                              with open(log_file, "a", encoding="utf-8") as f:
                                   f.write(f"[{tick_count}]\n")
                                   f.write("-" * 50 + "\n")
                                   f.write(f"用户输入: {user_input}\n")
                                   f.write("-" * 50 + "\n")
                                   f.write(f"第{answer_count}次回答: {answer}\n")
                                   f.write("-" * 50 + "\n")
                                   f.write(f"关键词判断: {judgment_outcome}, 预测标签:{predicted_label}\n")
                                   f.write("-" * 50 + "\n")
                                   f.write(f"提示词模板:{template}\n")
                                   f.write("-" * 50 + "\n")
               else:
                    print(f"用户{user_id}的会话{session_id}不存在,请先创建会话")

if __name__ == "__main__":
 try:
     main()
 except Exception as e:
     print(f"程序执行过程中出现异常: {e}")