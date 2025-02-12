from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import sqlite3
import json

class ControlChatHistoryData():
    #先写InMemoryMessageHistory,对获取的消息进行处理!
    class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):
        messages : List[BaseMessage] = Field(default_factory=list)

        #添加并处理对话记录
        def add_message(self, message: BaseMessage) -> None:
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []


    def __init__(self):
        """构建并连接数据库"""
        self.conn = sqlite3.connect('./database/chat_history.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
               user_id TEXT,
               session_id TEXT,
               history TEXT,
               PRIMARY KEY (user_id, session_id)
               )''')
        self.conn.commit()#保存数据库

    def add_history(self, user_id: str, session_id: str, history: InMemoryMessageHistory) -> None:#这里传入的history是InMemoryMessageHistory对象,不止是数据属性,更有模型中的方法.
        """将对话记录存入数据库"""
        history_json = json.dumps([message.model_dump() for message in history.messages])#这里只调用了其中的数据属性
        self.cursor.execute("INSERT INTO chat_history (user_id, session_id, history) VALUES (?,?,?)", 
                    (user_id, session_id, history_json))
        #将history(即对话)转换为json格式,然后传入数据库对应id的列中
        self.conn.commit()

    def get_session_history(self,user_id: str, session_id: str) -> InMemoryMessageHistory:
        """从指令获取指定用户历史记录(关键函数,为配置RWMH提供数据)"""
        self.cursor.execute("SELECT history FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
        result = self.cursor.fetchone()#这一行的所有列对应的数据,即history_json(我们的数据库只有"history"这一列)
        if result:
            history_json = result[0]#load解析json文件为一个py列表,列表中包含多个字典,每个字典对一次人的问或机器的回答
            messages = [BaseMessage(**msg) for msg in json.loads(history_json)]#解包列表中每个元素,即字典,并用BaseMessage类实例化,处理问答过程
            history = ControlChatHistoryData.InMemoryMessageHistory(messages = messages)#BaseMessage处理后的消息列表才可传入IMMH
            return history #返回IMMH对象,给RWMH使用
        return ControlChatHistoryData.InMemoryMessageHistory()#返回空IMMH对象,此时没有历史记录
    
    def delete_session_history(self, user_id: str, session_id: str) -> None:
        """删除指定用户和其指定的聊天历史记录"""
        self.cursor.execute("DELETE FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
        self.conn.commit()

    def list_user_ids(self) -> List[str]:
        """ 列出所有用户id"""
        self.cursor.execute("SELECT user_id FROM chat_history")
        rows = self.cursor.fetchall()#返回的是这列的所有行,就要每行的第一个元素,即user_id
        return [row[0] for row in rows]
            
    def list_session_ids(self,user_id: str) -> List[str]:
        """列出指定用户的会话列表"""
        self.cursor.execute("SELECT session_id FROM chat_history WHERE user_id=?", (user_id))
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]
    
    def create_new_user(self, user_id: str) -> None:
        """创建新用户"""
        self.cursor.execute("INSERT INTO chat_history (user_id) VALUES (?)",(user_id))
        self.conn.commit() 

    def create_new_session(self, user_id: str, session_id: str) -> None:
        """创建新会话"""
        self.cursor.execute("INSERT INTO chat_history (user_id, session_id) VALUES (?,?)",(user_id, session_id))
        self.conn.commit()

    




    




