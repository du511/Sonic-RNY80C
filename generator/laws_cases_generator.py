from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from generator.chat_history_control import ControlChatHistoryData
from langchain_core.runnables.base import RunnableSequence
from generator.embedding import Embedding#直接在def里面包含了输入数据的向量化,便于faiss数据库的查询
import toml

config = toml.load("config/parameters.toml")
bot_name = config["model_A"]["project_name"]

class LCResponseGenerator:

    def __init__(self, model_A, model_B):
        self.model_A = model_A
        self.model_B = model_B
#前面两个参数还是为了对实现账户管理系统,我需要对faiss数据库进行传入,
#这里我先做法律回答生成系统
    def generate_response_laws(self, input_user_id, input_session_id, question, 
                               laws_faiss_indexer, cases_faiss_indexer,
                               laws_index, cases_index,
                               return_template = True ):

        embedding_generator = Embedding("bert-base-uncased")#初始化向量生成器
        input_embedding = embedding_generator.get_embedding(question)#对输出向量化生成

        #faiss数据库查询,返回最相似的法律条文
        unique_id_laws = laws_index.search_index(input_embedding)

        #生成最相关的法律条文
        relevant_laws = list(set([laws_index[i] for i in unique_id_laws[0]]))

        #第一次输入的提示词模板并处理
        template_for_model_laws = f"""
        你{bot_name}是一位专业的法律专家，专门解答法律相关问题。请根据以下相关法律文本回答问题：
        相关法律文本：{relevant_laws}
        问题：{question}
        """ 
        prompt = ChatPromptTemplate(template_for_model_laws)

        #开始链式处理
        chain = RunnableSequence([lambda x: prompt,
                                  self.model_A,
                                  ])


        