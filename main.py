import os
import sys
import json
import toml
import jieba
import datetime

from reader.reader import DocumentReader  # 引入库读取器
from RAG.faiss_indexer import FaissIndexer  # 引入Faiss索引器
from generator.embedding import Embedding  # 引入文本向量化器
from generator.response_generator import (
    ResponseGenerator,
)  # 引入日常/网安技术/法律/案例解析回答生成器
from generator.chat_history_control import (
    ControlChatHistoryData,
)  # 引入对话历史记录控制系统
from naive_bayes_model.naive_bayes_classifier import (
    NaiveBayesClassifier,
)  # 引入朴素贝叶斯分类器
from naive_bayes_model.train_data.train_data import data  # 引入朴素贝叶斯训练数据
from generator.MyStreamingHandler import MyStreamingHandler  # 引入流式输出系统

# 读取配置文件
config = toml.load("config/parameter.toml")
bert_uncased_model_name = config["bert"]["model_name"]

# 读取提示参数
bot_name = config["model_A"]["project_name"]
logo = config["pattern"]["logo"]
welcome = config["pattern"]["welcome"]
help = config["help"]["help"]

# 模型A参数
base_url_A = config["model_A"]["base_url"]
local_model_name_A = config["model_A"]["model_name"]
temperature_A = config["model_A"]["temperature"]
top_p_A = config["model_A"]["top_p"]
top_k_A = config["model_A"]["top_k"]

# 初始化langchain本地模型部署
from langchain_ollama import ChatOllama

model_A = ChatOllama(
    base_url=base_url_A,
    model=local_model_name_A,
    temperature=temperature_A,
    top_p=top_p_A,
    top_k=top_k_A,
    callbacks=[MyStreamingHandler()],
    streaming=True,
)

# 确保日志存在,且创建日志目录以及日志文件
if not os.path.exists("./logs"):
    os.mkdir("logs")


def save_last_filename(
    net_filename, law_filename, case_filename
):  # 保存文件,要分类修改 2.19
    data = {
        "net_filename": net_filename,
        "law_filename": law_filename,
        "case_filename": case_filename,
    }
    with open("logs/library_name.json", "w") as f:
        json.dump(data, f)


def load_last_filename():
    if os.path.exists("logs/library_name.json"):
        with open("logs/library_name.json", "r") as f:
            data = json.load(f)
            return (
                data.get("net_filename"),
                data.get("law_filename"),
                data.get("case_filename"),
            )  # 改了,三个字符都要返回 2.19
    return None, None, None


def process_library_folder(folder_path, library_name):
    document_reader = DocumentReader()  # 读取器
    paragraphs = []  # 保持原来的变量名

    # 定义输出文件的完整路径
    base_folder = "docs"
    output_folder = os.path.join(base_folder, library_name)
    output_file_path = os.path.join(output_folder, f"{library_name}_rag_data.txt")

    # 如果输出文件夹不存在，创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 如果输出文件已存在，先删除
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            # 检查文件扩展名是否为支持的格式
            if file_extension in [".pdf", ".docx", ".txt"]:
                # 构造文件的完整路径
                file_path = os.path.join(root, file)
                # 排除目标输出文件本身被处理的情况
                if file_path != output_file_path:
                    text = document_reader.read_file(file_path)
                    # 按换行符分割文本
                    lines = text.split("\n")
                    formatted_lines = []
                    for line in lines:
                        # 为每一行加上文件名
                        formatted_line = f"{file}: {line}"
                        formatted_lines.append(formatted_line)
                    # 将处理后的行重新组合为一个字符串，每行之间用换行符分隔，并在每个文件内容后添加一个空行
                    formatted_text = "\n\n".join(formatted_lines) + "\n\n"
                    paragraphs.append(formatted_text)  # 添加到列表中

    # 将处理后的文本保存到目标文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        for paragraph in paragraphs:
            f.write(paragraph)

    return paragraphs  # 返回处理后的文本列表


def main():
    # 是否为调试模式
    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        debug_mode = True
    else:
        debug_mode = False

    net_filename, law_filename, case_filename = load_last_filename()
    if net_filename:
        use_last = input(f"(网安库)是否需要使用上次的库:{net_filename}? (y/n)")
        if use_last.lower() == "y":
            net_file_path = f"{net_filename}"
        else:
            net_file_path = input("(网安库)请输入库名(默认: net): ")
            net_file_path = f"{net_file_path}"
    else:
        net_file_path = input("(网安库)请输入库名(默认: net): ")
        net_file_path = f"{net_file_path}"

    if law_filename:
        use_last = input(f"(法律库)是否需要使用上次的库:{law_filename}? (y/n)")
        if use_last.lower() == "y":
            laws_file_path = f"{law_filename}"
        else:
            laws_file_path = input("(法律库)请输入库名(默认: laws): ")
            laws_file_path = f"{laws_file_path}"

    else:
        laws_file_path = input("(法律库)请输入库名(默认: laws): ")
        laws_file_path = f"{laws_file_path}"

    if case_filename:
        use_last = input(f"(案例库)是否需要使用上次的库:{case_filename}? (y/n)")
        if use_last.lower() == "y":
            cases_file_path = f"{case_filename}"
        else:
            cases_file_path = input("(案例库)请输入库名(默认: cases): ")
            cases_file_path = f"{cases_file_path}"
    else:
        cases_file_path = input("(案例库)请输入库名(默认: cases): ")
        cases_file_path = f"{cases_file_path}"
    # 先保存文件
    save_last_filename(net_file_path, laws_file_path, cases_file_path)

    # 对每个库的文件夹进行处理:
    net_paragraphs = process_library_folder(net_file_path, "net")
    law_paragraphs = process_library_folder(laws_file_path, "laws")
    case_paragraphs = process_library_folder(cases_file_path, "cases")

    """以下是主程序里的RAG部分:
       1. 读取RAG纯文本材料库
       2.获取RAG纯文本,处理为段,便于输入
       3.对每个库的faiss索引工具进行初始化
       """

    # 1.读取库,并进行处理
    document_reader = DocumentReader()  # 读取器

    # 构建路径
    file_paths = [
        os.path.join("docs", "net", "net_rag_data.txt"),
        os.path.join("docs", "laws", "laws_rag_data.txt"),
        os.path.join("docs", "cases", "cases_rag_data.txt"),
    ]

    # 先建立读取循环
    raw_texts = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                raw_texts.append(raw_text)
        except Exception as e:
            print(f"读取RAG纯文本材料库文件{file_path}失败,请检查文件路径或文件内容")

    # 再建立段落处理循环
    raw_paragraphs = []
    for raw_text in raw_texts:
        paragraphs = raw_text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        raw_paragraphs.append(paragraphs)

    """采用数字编号对不同的RAG纯文本材料(已分段)进行索引:(存在于raw_paragraphs中)
       网安知识: 0
       法律法规: 1
       法律案例: 2
     """
    # 2.初始化索引器
    faiss_indexer_net = FaissIndexer()  # 初始化网安索引器
    faiss_indexer_laws = FaissIndexer()  # 初始化法律索引器
    faiss_indexer_cases = FaissIndexer()  # 初始化案例索引器

    # 初始化对话历史记录
    answer_count = 0
    tick_count = 0

    # 初始化回答器
    response_generator = ResponseGenerator(model_A)

    # 开始记录日志
    if debug_mode:
        log_file = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    # 朴素贝叶斯分类
    classifier = NaiveBayesClassifier()
    model_path = "./naive_bayes_model/saved_model/naive_bayes_model.pkl"
    vectorizer_path = "./naive_bayes_model/saved_vectorizer/naive_bayes_vectorizer.pkl"
    train_data_path = data

    # 加载或训练模型
    if not os.path.exists(model_path) and not os.path.exists(vectorizer_path):
        classifier.train(train_data_path)
        classifier.save_model(model_path=model_path, vectorizer_path=vectorizer_path)
    else:
        classifier.load_model(model_path, vectorizer_path)

    # 实例化对话历史记录控制系统
    history_control = ControlChatHistoryData()

    print(logo)

    while True:  # 主程序循环
        order = input(f"欢迎使用{bot_name}!，请输入指令(-h为帮助,输入q退出): ")
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
        elif (
            len(order_parts) == 3 and order_parts[0] == "-l" and order_parts[2] == "-ls"
        ):
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
        elif (
            len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-d"
        ):
            user_id = order_parts[1]
            session_id = order_parts[3]
            sessions = history_control.list_session_ids(user_id)
            if session_id in sessions:
                history_control.delete_session_history(user_id, session_id)
                print(f"用户{user_id}的会话{session_id}已删除")
            else:
                print(f"用户{user_id}的会话{session_id}不存在,请先创建会话")

        # 会话创建指令
        elif (
            len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-n"
        ):
            user_id = order_parts[1]
            session_id = order_parts[3]
            sessions = history_control.list_session_ids(user_id)
            if session_id not in sessions:
                history_control.create_new_session(user_id, session_id)
                print(f"用户{user_id}的会话{session_id}已创建")
            else:
                print(f"用户{user_id}的会话{session_id}已存在,请勿重复创建")

        # 进入会话指令
        elif (
            len(order_parts) == 4 and order_parts[0] == "-l" and order_parts[2] == "-l"
        ):
            user_id = order_parts[1]
            session_id = order_parts[3]
            sessions = history_control.list_session_ids(user_id)
            if session_id in sessions:
                print(welcome)
                print(f"欢迎进入用户{user_id}的会话{session_id}，请输入指令: ")
                while True:
                    # 读取用户输入
                    user_input = input("\n请输入你的问题(输入'q'退出): ")
                    if user_input.lower() == "q":
                        break

                    # 朴素贝叶斯分类预测问题类型
                    predicted_label = classifier.predict(user_input)

                    if debug_mode:
                        print(f"预测标签:{predicted_label}")

                    # 获取回答以及提示词模板,回答生成处理在response_generator.py中,开始分类:
                    response_generator = ResponseGenerator(model_A)
                    if predicted_label == 0:
                        response, template = (
                            response_generator.generate_response_dailys(
                                user_id, session_id, user_input, return_template=True
                            )
                        )

                    elif predicted_label == 1:
                        response, template = response_generator.generate_response_nets(
                            user_id,
                            session_id,
                            user_input,
                            net_faisser_indexer=faiss_indexer_net,
                            rag_paragraphs_nets=raw_paragraphs[0],
                            return_template=True,
                        )

                    elif predicted_label == 2:
                        response, template = response_generator.generate_response_laws(
                            user_id,
                            session_id,
                            user_input,
                            laws_faiss_indexer=faiss_indexer_laws,
                            cases_faiss_indexer=faiss_indexer_cases,
                            rag_paragraphs_laws=raw_paragraphs[1],
                            rag_paragraphs_cases=raw_paragraphs[2],
                            return_template=True,
                        )

                    elif predicted_label == 3:
                        response, template = response_generator.analyze_case_with_law(
                            user_id,
                            session_id,
                            user_input,
                            case_faiss_indexer=faiss_indexer_cases,
                            law_faiss_indexer=faiss_indexer_laws,
                            rag_paragraphs_cases=raw_paragraphs[2],
                            rag_paragraphs_laws=raw_paragraphs[1],
                            return_template=True,
                        )

                    # 调试日志
                    if debug_mode:
                        tick_count += 1
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"[{tick_count}]\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"用户输入: {user_input}\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"第{answer_count}次回答: {response}\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"朴素贝叶斯预测标签:{predicted_label}\n")
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
