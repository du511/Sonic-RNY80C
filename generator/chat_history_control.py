from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import sqlite3
import queue

class SQLiteConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path)
            self.pool.put(conn)

    def get_connection(self):
        return self.pool.get()

    def release_connection(self, conn):
        self.pool.put(conn)

# 基本控制系统
class ControlChatHistoryData:
    # 先写 InMemoryMessageHistory，对获取的消息进行处理并获取历史记录列表！
    class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):
        messages: List[BaseMessage] = Field(default_factory=list)

        # 添加并处理对话记录
        def add_message(self, message: BaseMessage) -> None:
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []

    def __init__(self):
        """构建并连接数据库"""
        self.connection_pool = SQLiteConnectionPool('./generator/database/chat_history.db')
        self.conn = sqlite3.connect('./generator/database/chat_history.db')
        self.cursor = self.conn.cursor()
        # 修改表结构，添加消息内容、类型和序号列
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
               user_id TEXT,
               session_id TEXT,
               message_index INTEGER,
               message_content TEXT,
               message_type TEXT,
               PRIMARY KEY (user_id, session_id, message_index)
               )''')
        self.conn.commit()  # 保存数据库

    def add_history(self, user_id: str, session_id: str, history: InMemoryMessageHistory) -> None:
        """将对话记录存入数据库"""
        for index, message in enumerate(history.messages):#就指的是历史记录的消息(人机的对话记录)
            self.cursor.execute("INSERT INTO chat_history (user_id, session_id, message_index, message_content, message_type) VALUES (?,?,?,?,?)",
                                (user_id, session_id, index, message.content, message.type))
        self.conn.commit()

    def get_session_history(self, input_user_id: str, input_session_id: str) -> InMemoryMessageHistory:
        try:
            self.cursor.execute("SELECT message_content, message_type FROM chat_history WHERE user_id=? AND session_id=? ORDER BY message_index",
                                (input_user_id, input_session_id))
            rows = self.cursor.fetchall()
            messages = []
            for content, msg_type in rows:
                if content is not None and msg_type is not None:
                    messages.append(BaseMessage(content=content, type=msg_type))
            if messages:
                history = self.InMemoryMessageHistory(messages=messages)
                return history
            else:
                placeholder = self.InMemoryMessageHistory()
                placeholder_u_message = BaseMessage(content="No history found for this session.", type='user')
                placeholder_ai_message = BaseMessage(content="No history found for this session.", type='assistant')
                placeholder.add_message(placeholder_u_message)
                placeholder.add_message(placeholder_ai_message)
                self.add_history(input_user_id, input_session_id, placeholder)
        except Exception as e:
            print(f"历史记录获取失败！{e}")
            

    def update_session_history(self, user_id: str, session_id: str, user_message: str, assistant_message: str):
        """更新指定用户和其指定的聊天历史记录"""
        if user_message is None:
          user_message = ""
        if assistant_message is None:
            assistant_message = ""
        updated_history = self.InMemoryMessageHistory()
        new_user_message = BaseMessage(content=user_message, type='user')
        new_assistant_message = BaseMessage(content=assistant_message, type='assistant')
        updated_history.add_message(new_user_message)
        updated_history.add_message(new_assistant_message)
        try:
          self.add_history(user_id, session_id, updated_history)
          print("历史记录更新成功！")
        except Exception as e:
          print(f"历史记录更新失败！{e}")

    def delete_session_history(self, user_id: str, session_id: str) -> None:
        """删除指定用户和其指定的聊天历史记录"""
        self.cursor.execute("DELETE FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
        self.conn.commit()

    def delete_user_history(self, user_id: str) -> None:
        """删除指定用户的所有聊天历史记录"""
        self.cursor.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
        self.conn.commit()

    def list_user_ids(self) -> List[str]:
        """ 列出所有用户 id"""
        self.cursor.execute("SELECT DISTINCT user_id FROM chat_history")
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def list_session_ids(self, user_id: str) -> List[str]:
        """列出指定用户的会话列表"""
        self.cursor.execute("SELECT DISTINCT session_id FROM chat_history WHERE user_id=?", (user_id,))
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def create_new_user(self, user_id: str) -> None:
        """创建新用户"""
        self.cursor.execute("INSERT INTO chat_history (user_id) VALUES (?)", (user_id,))
        self.conn.commit()

    def create_new_session(self, user_id: str, session_id: str) -> None:
        """创建新会话"""
        self.cursor.execute("INSERT INTO chat_history (user_id, session_id) VALUES (?,?)", (user_id, session_id))
        self.conn.commit()




    




    




