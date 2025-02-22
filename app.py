import streamlit as st
import os
import json
import toml
from reader.reader import DocumentReader
from RAG.faiss_indexer import FaissIndexer
from generator.response_generator import ResponseGenerator
from generator.chat_history_control import ControlChatHistoryData
from naive_bayes_model.naive_bayes_classifier import NaiveBayesClassifier
from langchain_ollama import ChatOllama
import re

# 读取配置文件
config = toml.load("config/parameter.toml")
bert_uncased_model_name = config["bert"]["model_name"]

# 模型A参数
base_url_A = config["model_A"]["base_url"]
local_model_name_A = config["model_A"]["model_name"]
temperature_A = config["model_A"]["temperature"]
top_p_A = config["model_A"]["top_p"]
top_k_A = config["model_A"]["top_k"]

# 初始化LangChain本地模型部署
model_A = ChatOllama(
    base_url=base_url_A,
    model=local_model_name_A,
    temperature=temperature_A,
    top_p=top_p_A,
    top_k=top_k_A
)

st.set_page_config(page_title="网安智脑 - AI", page_icon="web/favicon.png")
# 初始化对话历史记录控制系统
history_control = ControlChatHistoryData()

# 初始化朴素贝叶斯分类器
classifier = NaiveBayesClassifier()
model_path = "./naive_bayes_model/saved_model/naive_bayes_model.pkl"
vectorizer_path = "./naive_bayes_model/saved_vectorizer/naive_bayes_vectorizer.pkl"
train_data_path = "./naive_bayes_model/train_data/train_data"

# 加载或训练模型
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    classifier.train(train_data_path)
    classifier.save_model(model_path=model_path, vectorizer_path=vectorizer_path)
else:
    classifier.load_model(model_path, vectorizer_path)

# 初始化回答器
response_generator = ResponseGenerator(model_A)

# 初始化索引器
faiss_indexer_net = FaissIndexer()
faiss_indexer_laws = FaissIndexer()
faiss_indexer_cases = FaissIndexer()

# 确保日志目录存在
if not os.path.exists("./logs"):
    os.mkdir("logs")

def save_last_user_id(user_id):
    with open("logs/last_user_id.txt", "w") as f:
        f.write(user_id)

def load_last_user_id():
    if os.path.exists("logs/last_user_id.txt"):
        with open("logs/last_user_id.txt", "r") as f:
            return f.read().strip()
    return ""

def save_last_library_paths(net_path, law_path, case_path):
    data = {
        "net_path": net_path,
        "law_path": law_path,
        "case_path": case_path
    }
    with open("logs/library_paths.json", "w") as f:
        json.dump(data, f)

def load_last_library_paths():
    if os.path.exists("logs/library_paths.json"):
        with open("logs/library_paths.json", "r") as f:
            return json.load(f)
    return {"net_path": "", "law_path": "", "case_path": ""}

def process_library_folder(folder_path, library_name):
    document_reader = DocumentReader()
    paragraphs = []
    base_folder = "docs"
    output_folder = os.path.join(base_folder, library_name)
    output_file_path = os.path.join(output_folder, f"{library_name}_rag_data.txt")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in [".pdf", ".docx", ".txt"]:
                file_path = os.path.join(root, file)
                if file_path != output_file_path:
                    text = document_reader.read_file(file_path)
                    text = text.replace("\n", "  ")
                    formatted_text = f"{file}: {text}"
                    paragraphs.append(formatted_text)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(paragraph + "\n\n")

    return paragraphs

# 初始化侧边栏
st.sidebar.title("用户和会话管理")

# 用户ID管理
last_user_id = load_last_user_id()
user_id = st.sidebar.selectbox("选择用户ID", ["新建用户"] + history_control.list_user_ids(), index=0 if not last_user_id else history_control.list_user_ids().index(last_user_id) + 1)

if user_id == "新建用户":
    new_user_id = st.sidebar.text_input("请输入新的用户ID")
    if st.sidebar.button("创建用户"):
        if new_user_id and new_user_id not in history_control.list_user_ids():
            history_control.create_new_user(new_user_id)
            user_id = new_user_id
            save_last_user_id(user_id)
            st.sidebar.success(f"用户 {user_id} 已创建")
        else:
            st.sidebar.error("用户ID已存在或无效")
else:
    save_last_user_id(user_id)

# 删除用户
if st.sidebar.button("删除当前用户"):
    if user_id and user_id != "新建用户":
        history_control.delete_user_history(user_id)
        st.sidebar.success(f"用户 {user_id} 的所有会话已删除")
        user_id = "新建用户"  # 重置用户ID
        save_last_user_id(user_id)  # 保存最新的用户ID

# 会话管理
selected_session = st.sidebar.selectbox("选择会话", ["新建会话"] + history_control.list_session_ids(user_id))

if selected_session == "新建会话":
    session_id = st.sidebar.text_input("新建会话 ID", key="new_session_id")
    if st.sidebar.button("创建会话"):
        history_control.create_new_session(user_id, session_id)
        st.sidebar.success(f"会话 {session_id} 已创建")
        selected_session = session_id
else:
    session_id = selected_session

# 删除会话
if st.sidebar.button("删除当前会话"):
    if session_id and session_id != "新建会话":
        history_control.delete_session_history(user_id, session_id)
        st.sidebar.success(f"会话 {session_id} 已删除")
        selected_session = "新建会话"  # 重置会话ID
        st.session_state.chat_history = []  # 清除前端聊天记录

# 文档路径输入
last_paths = load_last_library_paths()
net_file_path = st.sidebar.text_input("网安知识库路径", value=last_paths["net_path"])
law_file_path = st.sidebar.text_input("法律法规库路径", value=last_paths["law_path"])
case_file_path = st.sidebar.text_input("案例库路径", value=last_paths["case_path"])

# 保存当前文档路径
save_last_library_paths(net_file_path, law_file_path, case_file_path)

# 处理库文件夹
net_paragraphs = process_library_folder(net_file_path, "net")
law_paragraphs = process_library_folder(law_file_path, "laws")
case_paragraphs = process_library_folder(case_file_path, "cases")

# 聊天界面
st.title("网安智脑 AI")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 每次切换会话时清除前端聊天记录（不删除数据库中的历史记录）
if selected_session == "新建会话":
    st.session_state.chat_history = []

# 显示最新的对话
for message in st.session_state.chat_history:
    avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
    with st.chat_message(message["role"], avatar=avatar):
        # 直接渲染已处理内容（不再替换换行符）
        st.markdown(message["content"], unsafe_allow_html=False)

def get_response_content(response):
    """深度解析响应对象获取content内容"""
    try:
        # 尝试直接访问content属性
        if hasattr(response, 'content'):
            return response.content
        
        # 处理LangChain的AIMessage类型
        if hasattr(response, 'response_metadata'):
            return response.response_metadata.get('message', {}).get('content', '')
        
        # 处理字典类型的响应
        if isinstance(response, dict):
            return response.get('content', '')
        
        # 处理字符串类型的响应
        if isinstance(response, str):
            content_match = re.search(r"content='(.*?)'", response, re.DOTALL)
            return content_match.group(1) if content_match else response
        
        return str(response)
    except Exception as e:
        print(f"内容解析错误: {str(e)}")
        return "无法解析响应内容"

# 改进后的预处理函数
def preprocess_response(text):
    """增强型内容格式化"""
    # 移除所有HTML标签（关键修复）
    text = re.sub(r'<[^>]+>', '', text)
    
    # 处理转义字符（将双反斜杠转回单反斜杠）
    text = text.replace('\\\\', '\\')
    
    # 统一换行符为标准的 \n
    text = re.sub(r'(\\n)|[\r\n]+', '\n', text)
    
    # 自动修复常见Markdown格式
    text = re.sub(r'^(\s*)```', r'\1```', text, flags=re.MULTILINE)  # 代码块对齐
    text = re.sub(r'(?<!\\)(`{1,2})(?!`)', r'\\\1', text)  # 保护孤立的反引号
    
    # 处理中文标点与特殊符号
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "\\~": "~", "\\*": "*", "\\_": "_"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    return text.strip()

    
user_input = st.chat_input("请输入你的问题 (输入 'q' 退出)")
if user_input:
    if user_input.lower() == "q":
        st.write("会话已结束")
    else:
        predicted_label = classifier.predict(user_input)
        if predicted_label == 0:
            response = response_generator.generate_response_dailys(user_id, session_id, user_input)
        elif predicted_label == 1:
            response = response_generator.generate_response_nets(user_id, session_id, user_input, 
                                                                 net_faisser_indexer=faiss_indexer_net,
                                                                 rag_paragraphs_nets=net_paragraphs)
        elif predicted_label == 2:
            response = response_generator.generate_response_laws(user_id, session_id, user_input,
                                                                 laws_faiss_indexer=faiss_indexer_laws,
                                                                 cases_faiss_indexer=faiss_indexer_cases,
                                                                 rag_paragraphs_laws=law_paragraphs,
                                                                 rag_paragraphs_cases=case_paragraphs)
        elif predicted_label == 3:
            response = response_generator.analyze_case_with_law(user_id, session_id, user_input,
                                                                case_faiss_indexer=faiss_indexer_cases,
                                                                law_faiss_indexer=faiss_indexer_laws,
                                                                rag_paragraphs_cases=case_paragraphs,
                                                                rag_paragraphs_laws=law_paragraphs)

        raw_response = response  # 保留原始响应对象
        
        # 深度解析内容
        raw_content = get_response_content(raw_response)
        
        # 二次验证（确保提取正确）
        if "content='" in str(raw_response):
            content_match = re.search(r"content='(.*?)'", str(raw_response), re.DOTALL)
            fallback_content = content_match.group(1) if content_match else raw_content
            raw_content = fallback_content
        
        # 统一预处理
        processed_content = preprocess_response(raw_content)
        
        # 更新聊天历史记录
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": processed_content})

        # 显示最新消息
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(processed_content, unsafe_allow_html=False)