import os
import sys
import json
import toml
import datetime
from  reader.reader import DocumentReader
from RAG.faiss_indexer import FaissIndexer
from generator.embedding import Embedding
from generator.response_generator import ResponseGenerator
from filter.context_filter import ContextFilter

#读取配置文件
config = toml.load("config/parameter.toml")
api_base = config["openai"]["api_base"]
api_key = config["openai"]["api_key"]
openai_model_name = config["openai"]["model_name"]
temperature = config["openai"]["temperature"]
top_p = config["openai"]["top_p"]
bert_uncased_model_name = config["bert"]["model_name"]

#初始化openai
from openai import OpenAI
client = OpenAI(
    base_url = api_base,
    api_key = api_key, # required, but unused
)


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
     
    #初始化对话历史
     conversation_history = []
     answer_count = 0
     tick_count = 0

     filter_context = ContextFilter(openai_model_name, client)
     answer_generator = ResponseGenerator(openai_model_name, temperature, top_p, client)

     #开始记录日志
     if debug_mode:
          log_file = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

     while True:
          #读取用户输入
          user_input = input("请输入你的问题(输入'q'退出): ")
          if user_input.lower() == "q":
                 break
          
          #生成输入的向量
          input_embedding = embedding_generator.get_embedding(user_input)
          #搜索索引
          if input_embedding is None:
               continue
          if index is None:
            print("索引未建立,无法搜索")
            continue
          unique_id = faiss_indexer.search_index(input_embedding)
        #   if unique_id is None:
        #     relevant_context = []
        #   else:
          relevant_context = list(set([paragraphs[i] for i in unique_id]))           
          #生成回答
          answer, prompt = answer_generator.generate_response(user_input, relevant_context, conversation_history, return_prompt=True)
          #打印回答
          print(answer)
          #更新对话历史
          conversation_history.append((user_input, answer))
          #每十次筛选一次上下文
          answer_count += 1
          if answer_count % 10 == 0:
             conversation_history = filter_context.filter_context(conversation_history)
          #调试日志
          if debug_mode:
               tick_count += 1
               with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{tick_count}]\n")
                    f.write("-"*50 + "\n")
                    f.write(f"用户输入: {user_input}\n")
                    f.write("-"*50 + "\n")
                    f.write(f"第{answer_count}次回答: {answer}\n")
                    f.write("-"*50 + "\n")
                    f.write(f"提示词:{prompt}\n")
                    f.write("-"*50 + "\n")
                    f.write(f"对话历史: {conversation_history}\n")
                    f.write("-"*50 + "\n")

if __name__ == "__main__":
     main()