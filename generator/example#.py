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

# 定义 InMemoryMessageHistory 类
class InMemoryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: List[BaseMessage]) -> None:
        self.messages.extend(message)

    def clear(self) -> None:
        self.messages = []

# 连接到 SQLite 数据库
conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_histories (
        user_id TEXT,
        session_id TEXT,
        history TEXT,
        PRIMARY KEY (user_id, session_id)
    )
''')
conn.commit()

# 新增会话历史记录
def add_session_history(user_id: str, session_id: str, history: InMemoryMessageHistory):
    import json
    history_json = json.dumps([message.dict() for message in history.messages])
    cursor.execute('''
        INSERT OR REPLACE INTO chat_histories (user_id, session_id, history)
        VALUES (?,?,?)
    ''', (user_id, session_id, history_json))
    conn.commit()

# 查看会话历史记录
def get_session_history(user_id: str, session_id: str) -> InMemoryMessageHistory:
    cursor.execute('''
        SELECT history FROM chat_histories
        WHERE user_id =? AND session_id =?
    ''', (user_id, session_id))
    result = cursor.fetchone()
    if result:
        history_json = result[0]
        messages = [BaseMessage(**msg) for msg in json.loads(history_json)]
        history = InMemoryMessageHistory(messages=messages)
        return history
    return InMemoryMessageHistory()

# 删除会话历史记录
def delete_session_history(user_id: str, session_id: str):
    cursor.execute('''
        DELETE FROM chat_histories
        WHERE user_id =? AND session_id =?
    ''', (user_id, session_id))
    conn.commit()

# 查看所有用户
def get_all_users():
    cursor.execute('SELECT DISTINCT user_id FROM chat_histories')
    rows = cursor.fetchall()
    return [row[0] for row in rows]

# 查看用户的所有会话
def get_user_sessions(user_id: str):
    cursor.execute('SELECT DISTINCT session_id FROM chat_histories WHERE user_id =?', (user_id,))
    rows = cursor.fetchall()
    return [row[0] for row in rows]

# 删除用户及其所有会话
def delete_user(user_id: str):
    cursor.execute('DELETE FROM chat_histories WHERE user_id =?', (user_id,))
    conn.commit()

# 初始化 LangChain 链
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at answering questions."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | ChatOpenAI()

# 初始化 RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    output_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="用户 ID",
            description="每个 user 的唯一标识符",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="会话 ID",
            description="每个 session 的唯一标识符",
            default="",
            is_shared=True,
        ),
    ],
)

def main():
    while True:
        users = get_all_users()
        if not users:
            print("无用户")
        else:
            print("用户列表:")
            for user in users:
                print(user)

        command = input("请输入指令 (-l 读取用户名, -d 删除用户, -n 新建用户, q 退出): ")
        if command == 'q':
            break
        elif command == '-l':
            user_id = input("请输入要读取的用户名: ")
            if user_id not in users:
                print("用户不存在")
                continue
            while True:
                sessions = get_user_sessions(user_id)
                if not sessions:
                    print("无会话")
                else:
                    print("会话列表:")
                    for session in sessions:
                        print(session)

                session_command = input("请输入指令 (-l 读取会话名, -d 删除会话, -n 新建会话, q 退出): ")
                if session_command == 'q':
                    break
                elif session_command == '-l':
                    session_id = input("请输入要读取的会话名: ")
                    if session_id not in sessions:
                        print("会话不存在")
                        continue
                    while True:
                        history = get_session_history(user_id, session_id)
                        print("聊天历史:")
                        for message in history.messages:
                            print(message)

                        question = input("请输入问题 (q 退出): ")
                        if question == 'q':
                            break
                        result = chain_with_history.invoke(
                            {"question": question},
                            config={"configurable": {"user_id": user_id, "session_id": session_id}}
                        )
                        new_history = get_session_history(user_id, session_id)
                        add_session_history(user_id, session_id, new_history)
                elif session_command == '-d':
                    session_id = input("请输入要删除的会话名: ")
                    if session_id in sessions:
                        delete_session_history(user_id, session_id)
                        print("会话已删除")
                    else:
                        print("会话不存在")
                elif session_command == '-n':
                    session_id = input("请输入新会话名: ")
                    new_history = InMemoryMessageHistory()
                    add_session_history(user_id, session_id, new_history)
                    print("新会话已创建")
                else:
                    print("无效指令")
        elif command == '-d':
            user_id = input("请输入要删除的用户名: ")
            if user_id in users:
                delete_user(user_id)
                print("用户已删除")
            else:
                print("用户不存在")
        elif command == '-n':
            user_id = input("请输入新用户名: ")
            # 先创建一个空会话，确保用户存在于数据库中
            session_id = "initial_session"
            new_history = InMemoryMessageHistory()
            add_session_history(user_id, session_id, new_history)
            print("新用户已创建")
        else:
            print("无效指令")

    # 关闭数据库连接
    conn.close()

if __name__ == "__main__":
    main()
