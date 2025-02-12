from operator import itemgetter
from typing import List

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

import sqlite3
import json



#定义会话类

class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):#由于是一个类,在实例化的过程中,作为一个值自动传入且处理对话记录
    messages: List[BaseMessage] = Field(default_factory=list) #库中已经封装,自动进行数据规范,且却表每个实例都有一个空列表
    #messages属性为列表,且列表中的每个元素被BaseMessage类实例化,即每个元素都是一个消息对象,消息对象包含消息内容,消息类型,消息时间等属性
    #每个被baseMessage处理过后的消息都被添加到列表中,会有相应方法自动将其处理具有为各个属性的字典形式,方便储存和读取
    #添加信息
    def add_session_history(self, message: List[BaseMessage]) -> None:
        self.messages.extend(message)

    #清空列表
    def clear(self) -> None:
        self.messages = []

#创造并连接数据库文件
conn = sqlite3.connect('./database/chat_history.db')
cursor = conn.cursor()

#创建表
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history(
                user_id text,
                session_id text,
                PRIMARY KEY (user_id, session_id),
                history text
                )''')
conn.commit()

#储存/读取会话记录系统(存疑)
def add_session_history(user_id: str, session_id: str, history: InMemoryMessageHistory):
    history_json = json.dumps([history.dict() for message in history.messages])
    cursor.execute('''
                   INSERT OR REPLACE INTO chat_history (user_id, session_id, history)
                   VALUES (?,?,?)
                   ''', (user_id, session_id, history_json))


    


###############################
#获取/创建会话列表,使用两个参数session_id和user_id
def get_session_history(user_id: str, session_id: str, history: InMemoryMessageHistory) -> InMemoryMessageHistory:
    if (user_id, session_id) not in store:
        store[(user_id, session_id)] = InMemoryMessageHistory() #双元素元组作为参数传入
    return store[(user_id, session_id)]

#略,获取历史记录

#初始化RWTH系统

chain_with_history = RunnableWithMessageHistory(
    chain,#传入可运行链
    get_session_history = get_session_history, #获得获取对应id的配置
    input_messages_key = "question",           #获取处理的输入信息对应的key
    output_messages_key = "history",           #历史记录嵌入位置的对应key
    history_factory_config = [                 #配置历史记录工厂,用于定义获取聊天历史时所需的可配置参数
        ConfigurableFieldSpec(                 #特别注意:并不只是以元组的id,session配置方式才让ConfigurableFieldSpec用如下方式呈现配置
            id = "user_id",                    #配置参数id，这里的id就对应get_session_history函数中的参数名
            annotation = str,                  #RunnableWithMessageHistory相当方便,它实例化中的参数可以传入你自定义的会话用户储存方法以及其对应参数
            name = "用户ID",                    #并且自动进行适配
            description = "每个user的唯一标识符",
            default = "",          
            is_shared = True,                  #总的来说,RunableWithMessageHistory的配置参数是非常灵活的,你可以根据自己的需求进行配置
            ),                                 #它分为传入运行链,以及用户传入和历史记录消息参数对应位置两个基础配置部分
        ConfigurableFieldSpec(                 #和获取对话记录的自定义部分
            id = "session_id",
            annotation = str,
            name = "会话ID",
            description = "每个session的唯一标识符",
            default = "",
            is_shared = True,
            ),     
    ],
)

chain_with_history.invoke(
    {"question":question},
    config={"user_id":user_id, "session_id":session_id}#这里就用来传入参数了,外部输入的参数
)






