import os
import sys
import json
import toml
import jieba
import datetime

from  reader.reader import DocumentReader#引入文档读取器
from RAG.faiss_indexer import FaissIndexer#引入Faiss索引器
from generator.embedding import Embedding#引入文本向量化器
from generator.daily_net_response_generator import DNetResponseGenerator#引入日常/网安技术回答生成器
#引入法律及其案例分类器
from generator.chat_history_control import ControlChatHistoryData #引入对话历史记录控制系统
from naive_bayes_model.naive_bayes_classifier import NaiveBayesClassifier#引入朴素贝叶斯分类器
from naive_bayes_model.train_data.train_data import data#引入朴素贝叶斯训练数据
from generator.MyStreamingHandler import MyStreamingHandler#引入流式输出系统

#读取配置文件
config = toml.load("config/parameter.toml")
bert_uncased_model_name = config["bert"]["model_name"]

#读取提示参数
bot_name = config["model_A"]["project_name"]
logo = config["pattern"]["logo"]
welcome = config["pattern"]["welcome"]
help = config["help"]["help"]

#模型A参数
base_url_A = config["model_A"]["base_url"]
local_model_name_A = config["model_A"]["model_name"]
temperature_A = config["model_A"]["temperature"]
top_p_A = config["model_A"]["top_p"]
top_k_A = config["model_A"]["top_k"]

#模型B参数
base_url_B = config["model_B"]["base_url"]
local_model_name_B = config["model_B"]["model_name"]
temperature_B = config["model_B"]["temperature"]
top_p_B = config["model_B"]["top_p"]
top_k_B = config["model_B"]["top_k"]




#初始化langchain本地模型部署
from langchain_ollama import ChatOllama
model_A = ChatOllama(base_url = base_url_A, model = local_model_name_A,temperature = temperature_A, top_p = top_p_A, top_k = top_k_A, 
                   callbacks = [MyStreamingHandler()], streaming = True )
model_B = ChatOllama(base_url = base_url_B, model = local_model_name_B,temperature = temperature_B, top_p = top_p_B, top_k = top_k_B, )

#确保日志存在,且创建日志目录以及日志文件
if not os.path.exists("./logs"):
    os.mkdir("logs")

def save_last_filename(net_filename, law_filename, case_filename):#保存文件,要分类修改 2.19
     data = {
          "net_filename" : net_filename,
          "law_filename" : law_filename,
          "case_filename" : case_filename
     }
     with open("logs/last_filename.json", "w") as f:
          json.dump(data, f)

def load_last_filename():
     if os.path.exists("logs/filename.json"):
         with open("logs/filename.json", "r") as f:
             data = json.load(f)
             return data.get("net_filename"), data.get("law_filename"), data.get("case_filename") #改了,三个字符都要返回 2.19
     return None       

def main():
     #是否为调试模式
     if len(sys.argv) > 1 and sys.argv[1] == "-d":
          debug_mode = True
     else:
          debug_mode = False
     
     net_filename, law_filename, case_filename = load_last_filename()
     if net_filename:
          use_last = input(f"(库1)是否需要使用上次的文档:{net_filename}? (y/n)")
          if use_last.lower() == "y":
               net_file_path = net_filename
          else:
               net_file_path = input("(库一)请输入文档路径: ")
     else:
          net_file_path = input("(库一)请输入文档路径: ")

     if law_filename:
          use_last = input(f"(库2)是否需要使用上次的文档:{law_filename}? (y/n)")
          if use_last.lower() == "y":
               laws_file_path = law_filename
          else:
               laws_file_path = input("(库二)请输入文档路径: ")
     else:
          laws_file_path = input("(库二)请输入文档路径: ")

     if case_filename:
          use_last = input(f"(库3)是否需要使用上次的文档:{case_filename}? (y/n)")
          if use_last.lower() == "y":
               cases_file_path = case_filename
          else:
               cases_file_path = input("(库三)请输入文档路径: ")
     else:
          cases_file_path = input("(库三)请输入文档路径: ")
     #先保存文件
     save_last_filename(net_file_path, laws_file_path, cases_file_path)
                 
  #这里需要修改:对三个文件都进行读取
     #读取文档
     document_reader = DocumentReader()#读取器
     #建立文档列表,再一个个读取
     file_list = [net_file_path, laws_file_path, cases_file_path]
     raw_text = [document_reader.read_file(file_path) for file_path in file_list]
     raw_paragraphs = []
     for text in raw_text:
          paragraphs = text.split("\n\n")
          raw_paragraphs.append(paragraphs)#对段落进行分段处理
    
     #建立Faiss索引
     embedding_generator = Embedding(bert_uncased_model_name)#向量生成器
     faiss_indexer_net = FaissIndexer()#网安知识索引器
     faiss_indexer_laws = FaissIndexer()#法律索引器
     faiss_indexer_cases = FaissIndexer()#案例索引器

     i = 0
     while i < len(raw_paragraphs):
          if i == 0:
               net_indexs = faiss_indexer_net.build_index(raw_paragraphs[i], embedding_generator)#一个index里面包含一种数据的很多段
               i += 1
          elif i == 1:
               laws_indexs = faiss_indexer_laws.build_index(raw_paragraphs[i], embedding_generator)
               i += 1
          else:
               cases_indexs = faiss_indexer_cases.build_index(raw_paragraphs[i], embedding_generator)
               i += 1

     #初始化对话历史记录
     answer_count = 0
     tick_count = 0

     #初始化回答器
     answer_generator = DNetResponseGenerator(model_A)

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

                         # 朴素贝叶斯分类预测问题类型
                         predicted_label = classifier.predict(user_input)

                         # 生成输入的向量
                         input_embedding = embedding_generator.get_embedding(user_input)
                         # *搜索索引这一部分还会施工*
                         if input_embedding is None:
                              continue
                         if index is None:
                              print("索引未建立,无法搜索")
                              continue
                         unique_id = faiss_indexer.search_index(input_embedding, top_k=5)
                         # 对rag使用进行判断
  
                         if predicted_label == 0:
                           relevant_context = []
                         elif predicted_label == 1:
                           relevant_context = list(set([paragraphs[i] for i in unique_id]))

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