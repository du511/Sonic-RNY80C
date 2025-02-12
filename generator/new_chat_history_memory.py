import json
import sqlite3
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from langchain_core.runnables import ConfigurableFieldSpec

#先写InMemoryMessageHistory,对获取的消息进行处理!
class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages : List[BaseMessage] = Field(default_factory=list)

    #添加并处理对话记录
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []
    #每次生成一次对话记录,RWMH就会使用这个模块自动处理对话记录:
    #将对话记录储存在message列表中,这个过程就包括对话记录的格式化处理

#构建数据库并连接
conn = sqlite3.connect('./database/chat_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
               user_id TEXT,
               session_id TEXT,
               history TEXT,
               PRIMARY KEY (user_id, session_id)
               )''')
conn.commit()

#增添历史记录
def add_history(user_id: str, session_id: str, history: InMemoryMessageHistory) -> None:
    history_json = json.dumps([history.dict() for message in history.messages])
    cursor.execute("INSERT INTO chat_history (user_id, session_id, history) VALUES (?,?,?)", 
                   (user_id, session_id, history_json))
    #将history(即对话)转换为json格式,然后传入数据库对应id的列中
    conn.commit()

#从指令获取指定用户历史记录(关键函数,为配置RWMH提供数据)
def get_session_history(user_id: str, session_id: str) -> InMemoryMessageHistory:
    cursor.execute("SELECT history FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
    result = cursor.fetchone()#这一行的所有列对应的数据,即history_json(我们的数据库只有"history"这一列)
    if result:
        history_json = result[0]#load解析json文件为一个py列表,列表中包含多个字典,每个字典对一次人的问或机器的回答
        messages = [BaseMessage(**msg) for msg in json.loads(history_json)]#解包列表中每个元素,即字典,并用BaseMessage类实例化,处理问答过程
        history = InMemoryMessageHistory(messages = messages)#BaseMessage处理后的消息列表才可传入IMMH
        return history #返回IMMH对象,给RWMH使用
    return InMemoryMessageHistory()#返回空IMMH对象,此时没有历史记录

#删除指定用户和其指定的聊天历史记录
def delete_session_history(user_id: str, session_id: str) -> None:
    cursor.execute("DELETE FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
    conn.commit()

#列出所有用户id
def list_user_ids() -> List[str]:
    cursor.execute("SELECT user_id FROM chat_history")
    rows = cursor.fetchall()#返回的是这列的所有行,就要每行的第一个元素,即user_id
    return [row[0] for row in rows]
        
#列出指定用户的会话列表
def list_session_ids(user_id: str) -> List[str]:
    cursor.execute("SELECT session_id FROM chat_history WHERE user_id=?", (user_id,))
    rows = cursor.fetchall()
    return [row[0] for row in rows]




