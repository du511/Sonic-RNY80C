from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage  # 修改导入
from pydantic import BaseModel, Field
import sqlite3
import queue

class SQLiteConnectionPool:
    # 保持原有实现不变
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

class ControlChatHistoryData:
    class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_message(self, message: BaseMessage) -> None:
            self.messages.append(message)

        def clear(self) -> None:
            self.messages = []

    def __init__(self):
        # 保持原有数据库连接逻辑不变
        self.connection_pool = SQLiteConnectionPool('./generator/database/chat_history.db')
        conn = self.connection_pool.get_connection()
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            user_id TEXT,
            session_id TEXT,
            message_index INTEGER,
            message_content TEXT,
            message_type TEXT,
            PRIMARY KEY (user_id, session_id, message_index)
        )''')
        conn.commit()
        self.connection_pool.release_connection(conn)

    # 修改关键方法：get_session_history
    def get_session_history(self, input_user_id: str, input_session_id: str) -> InMemoryMessageHistory:
        conn = self.connection_pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_content, message_type FROM chat_history "
                "WHERE user_id=? AND session_id=? ORDER BY message_index",
                (input_user_id, input_session_id)
            )
            messages = []
            for content, msg_type in cursor.fetchall():
                # 关键修改：区分消息类型
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    messages.append(AIMessage(content=content))
            return self.InMemoryMessageHistory(messages=messages)
        finally:
            self.connection_pool.release_connection(conn)

    # 修改关键方法：update_session_history
    def update_session_history(self, user_id: str, session_id: str, user_message: str, assistant_message: str):
        """更新历史记录（追加模式）"""
        # 获取现有历史记录
        existing_history = self.get_session_history(user_id, session_id)
        
        # 追加新消息
        existing_history.add_message(HumanMessage(content=user_message))
        existing_history.add_message(AIMessage(content=assistant_message))
        
        # 原子化更新数据库
        conn = self.connection_pool.get_connection()
        try:
            # 先删除旧记录
            conn.execute("DELETE FROM chat_history WHERE user_id=? AND session_id=?", (user_id, session_id))
            # 插入所有消息（包括新增）
            for index, message in enumerate(existing_history.messages):
                conn.execute(
                    "INSERT INTO chat_history VALUES (?,?,?,?,?)",
                    (user_id, session_id, index, message.content, message.type)
                )
            conn.commit()
        finally:
            self.connection_pool.release_connection(conn)

    # 以下方法保持原样
    def add_history(self, user_id: str, session_id: str, history: InMemoryMessageHistory) -> None:
        """保持原有实现不变"""
        # ...（原有代码不变）

    def delete_session_history(self, user_id: str, session_id: str) -> None:
        """保持原有实现不变"""
        # ...（原有代码不变）

    # 其他辅助方法保持原样
    # ...（原有代码不变）