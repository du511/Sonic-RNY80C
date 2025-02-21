from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from generator.chat_history_control import ControlChatHistoryData
from generator.embedding import Embedding#直接在def里面包含了输入数据的向量化,便于faiss数据库的查询,这个东西在主程序只有一个作用,就是向量化输入然后开始索引
import toml

config = toml.load("config/parameter.toml")
bot_name = config["model_A"]["project_name"]

class ResponseGenerator:

    #传入用户相关模型
    def __init__(self, model_A):
        self.model_A = model_A
    
    #直接定义一个生成相关文本的函数
    def get_relevant_texts(self,input_text, embedding_generator, rag_paragraphs, faiss_indexer ):
        """传入的参数有:用户输入,向量生成器,RAG的纯文本,faiss数据库索引工具"""
        input_embedding = embedding_generator.get_embedding(input_text)
        faiss_indexer.build_index(rag_paragraphs, embedding_generator)#建立索引
        unique_id_laws = faiss_indexer.search_index(input_embedding)#搜索并获取索引
        relevant_texts = list(set([rag_paragraphs[i] for i in (unique_id_laws[0] if isinstance(unique_id_laws[0], list) else [unique_id_laws[0]])]))
        return relevant_texts#返回相关文本
    


    #日常寒暄输出
    def generate_response_dailys(self, input_user_id, input_session_id, question, return_template = False):
        role = f"""
               你是{bot_name},一个有一定网络安全与执法知识的智能闲聊科普机器人。
               回答毒舌+傲娇+少量颜文字"""
        template = f"""
            <|system|>{role}<|end|>
            <|human|>当前问题是：{question}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请严格按照以下规则回答问题：
            以傲娇可爱的风格简洁回应。简洁回答即可。
            加上颜文字<|end|>
            """
        prompt = ChatPromptTemplate.from_template(template)
        try:
            history_control = ControlChatHistoryData()
            chain = prompt | self.model_A

            chat_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history = history_control.get_session_history,#传入对应方法
            input_message_key = "question",
            output_message_key = "history",
            history_factory_config = [
                ConfigurableFieldSpec(
                    id = "input_user_id",
                    annotation = str,
                    name = "用户ID",
                    description = "每个用户的唯一标识",
                    default = ""),
                ConfigurableFieldSpec(
                    id = "input_session_id",
                    annotation = str,
                    name = "会话ID",
                    description = "每个用户对应的会话的唯一标识",
                    default = "")
                
            ],
        )
            
            answer = chat_with_history.invoke({"question":question}, config={"input_user_id": input_user_id, "input_session_id": input_session_id})
            history_control.update_session_history(input_user_id, input_session_id, question, answer.content)#更新历史记录
            if return_template:
                return answer, template
            else:
                return answer
        except Exception as e:
            print(f"抱歉无法回答: {e}")
            if return_template:
                return "" ,template
            else:
                return ""
        
    #网络安全问题回答(直接回答)
    def generate_response_nets(self, input_user_id, input_session_id, question,
                                net_faisser_indexer,
                                rag_paragraphs_nets,
                                return_template=True):
        """(主程序传参)传入的参数有:用户id,会话id,用户输入,
        net_faiss工具库,RAG的纯文本,返回template的开关
        """
        embedding_generator = Embedding("bert-base-chinese")#初始化向量生成器
        relevant_net_texts = self.get_relevant_texts(question, embedding_generator, rag_paragraphs_nets, net_faisser_indexer)#获取相关文本,接下传入链式结构
        role = f""" 
        你是{bot_name},一个网络安全AI机器人。
        在回答用户问题前，必须优先参考之前的对话历史记录。若历史记录中有相关信息，要基于此准确作答。
        根据指示使用RAG文档或独立回答,与用户聊天对话,仅使用用户的语言进行对话
        """
        relevant_net_texts = "\n".join(relevant_net_texts)       
        template = f"""
            <|system|>{role}<|end|>
            <|human|>当前问题：{question}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请严格按照以下优先级处理问题：
            1. 优先判断问题类型：若用户进行寒暄/无关提问（如问候、天气等），别管{relevant_net_texts}相关内容。
            2. 网络安全问题处理：当且仅当问题涉及网络安全时：
            确保回答准确简洁，剔除无关技术细节
            不需要做任何解释,直接回答即可
            网络安全问题处理：当且仅当问题涉及网络安全时：
            结合以下RAG文档中的专业信息
            相关RAG文档内容：
            {relevant_net_texts}这些内容只有在问题出现与其中内容相关的时候才使用对应内容
            加上颜文字<|end|>
            """
        prompt = ChatPromptTemplate.from_template(template)
        try:
            history_control = ControlChatHistoryData()
            chain = prompt | self.model

            chat_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history = history_control.get_session_history,#传入对应方法
            input_message_key = "question",
            output_message_key = "history",
            history_factory_config = [
                ConfigurableFieldSpec(
                    id = "input_user_id",
                    annotation = str,
                    name = "用户ID",
                    description = "每个用户的唯一标识",
                    default = ""),
                ConfigurableFieldSpec(
                    id = "input_session_id",
                    annotation = str,
                    name = "会话ID",
                    description = "每个用户对应的会话的唯一标识",
                    default = "")
                
            ],
        )
            
            answer = chat_with_history.invoke({"question":question}, config={"input_user_id": input_user_id, "input_session_id": input_session_id})
            history_control.update_session_history(input_user_id, input_session_id, question, answer.content)#更新历史记录

            if return_template:
                return answer, template
            else:
                return answer
        except Exception as e:
            print(f"抱歉无法回答网络安全类问题: {e}")
            if return_template:
                return "" ,template
            else:
                return ""        

    # 前面两个参数还是为了对实现账户管理系统,我需要对faiss数据库进行传入,
    # 这里我先做法律回答生成系统
    def generate_response_laws(self, input_user_id, input_session_id, question,
                               laws_faiss_indexer, cases_faiss_indexer,
                               rag_paragraphs_laws, rag_paragraphs_cases,
                               return_template=True):
        """(主程序传参)传入的参数有:用户id,会话id,用户输入,
        相关RAG文本的生成:RAG相关纯文本,两种faiss工具库,以及准备生成为索引的纯文本材料。
        以及返回template的开关
        """
        embedding_generator = Embedding("bert-base-chinese")  # 初始化向量生成器

        relevant_laws = self.get_relevant_texts(question, embedding_generator, rag_paragraphs_laws,
                                                laws_faiss_indexer)  # 获取相关文本,接下传入链式结构

        # 直接把中间结果输出封装为函数,便于隐藏
        role = f""" 
        你是{bot_name},一个专业的法律解答助手。
        在回答用户问题前，必须优先参考之前的对话历史记录。若历史记录中有相关信息，要基于此准确作答。
        仅使用用户的语言进行对话，提供准确、简洁的法律解答。
        并且尽力列出法律条文相关案例，以便更准确地让用户理解法律条文。
        """
        rag_laws_context = '\n'.join(relevant_laws)

        # 生成法律解答的内部函数
        def generate_law_answer():
            template_law = f"""
            <|system|>{role}<|end|>
            <|human|>当前问题：{question}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请根据以下相关法律条文解答问题：
            {rag_laws_context}
            回答需准确简洁，剔除无关技术细节。<|end|>
            """
            prompt_law = ChatPromptTemplate.from_template(template_law)
            history_control = ControlChatHistoryData()
            chain_law = prompt_law | self.model_A
            chat_with_history_law = RunnableWithMessageHistory(
                chain_law,
                get_session_history=history_control.get_session_history,
                input_message_key="question",
                output_message_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="input_user_id",
                        annotation=str,
                        name="用户ID",
                        description="每个用户的唯一标识",
                        default=""
                    ),
                    ConfigurableFieldSpec(
                        id="input_session_id",
                        annotation=str,
                        name="会话ID",
                        description="每个用户对应的会话的唯一标识",
                        default=""
                    )
                ]
            )
            try:
                law_answer = chat_with_history_law.invoke({"question": question},
                                                          config={"input_user_id": input_user_id,
                                                                  "input_session_id": input_session_id})
            except Exception as e:
                print(f"在 generate_response_laws 函数的 generate_law_answer 中调用模型时出错: {e}")
                return None
            history_control.update_session_history(input_user_id, input_session_id, question,
                                                   law_answer.content)
            return law_answer

        law_answer = generate_law_answer()
        if law_answer is None:
            if return_template:
                return None, None
            else:
                return None, " "

        # 生成案例信息的内部函数
        def generate_case_info():
            template_case = f"""
            <|system|>{role}<|end|>
            <|human|>根据以下法律解答找出相关案例：{law_answer.content}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>只输出相关案例信息。<|end|>
            """
            prompt_case = ChatPromptTemplate.from_template(template_case)
            history_control = ControlChatHistoryData()
            chain_case = prompt_case | self.model_A
            chat_with_history_case = RunnableWithMessageHistory(
                chain_case,
                get_session_history=history_control.get_session_history,
                input_message_key="question",
                output_message_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="input_user_id",
                        annotation=str,
                        name="用户ID",
                        description="每个用户的唯一标识",
                        default=""
                    ),
                    ConfigurableFieldSpec(
                        id="input_session_id",
                        annotation=str,
                        name="会话ID",
                        description="每个用户对应的会话的唯一标识",
                        default=""
                    )
                ]
            )
            try:
                case_info = chat_with_history_case.invoke({"question": question},
                                                          config={"input_user_id": input_user_id,
                                                                  "input_session_id": input_session_id})
            except Exception as e:
                print(f"在 generate_response_laws 函数的 generate_case_info 中调用模型时出错: {e}")
                return None
            history_control.update_session_history(input_user_id, input_session_id, question,
                                                   case_info.content)
            return case_info

        case_info = generate_case_info()
        if case_info is None:
            if return_template:
                return None, None
            else:
                return None, " "

        relevant_cases = self.get_relevant_texts(case_info.content, embedding_generator, rag_paragraphs_cases,
                                                 cases_faiss_indexer)

        # 最后的回答系统
        rag_cases_context = '\n'.join(relevant_cases)
        template_final = f"""
        <|system|>{role}<|end|>
        <|human|>根据以下法律条文和相关案例进行总结回答：
        法律条文：{rag_laws_context}
        相关案例：{rag_cases_context}
        <|history|>{{history}}<|end|>
        <|assistant|>请按照以下格式输出回答：
        法律 xx 条是什么什么(基本信息), 意义是 xx, 适用于 xx
        相关案例是: xx
        回答必须要根据相关案例解释法律条文使用户更容易理解。<|end|>
        """
        prompt_final = ChatPromptTemplate.from_template(template_final)
        history_control = ControlChatHistoryData()
        chain_final = prompt_final | self.model_A
        chat_with_history_final = RunnableWithMessageHistory(
            chain_final,
            get_session_history=history_control.get_session_history,
            input_message_key="question",
            output_message_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="input_user_id",
                    annotation=str,
                    name="用户ID",
                    description="每个用户的唯一标识",
                    default=""
                ),
                ConfigurableFieldSpec(
                    id="input_session_id",
                    annotation=str,
                    name="会话ID",
                    description="每个用户对应的会话的唯一标识",
                    default=""
                )
            ]
        )
        try:
            final_answer = chat_with_history_final.invoke({"question": question},
                                                          config={"input_user_id": input_user_id,
                                                                  "input_session_id": input_session_id})
        except Exception as e:
            print(f"在 generate_response_laws 函数的最终调用模型时出错: {e}")
            final_answer = None
        if final_answer is not None:
            history_control.update_session_history(input_user_id, input_session_id, question,
                                                   final_answer.content)

        if return_template:
            return final_answer, template_final
        else:
            return final_answer, " "

    # 对案件的分析的回复生成函数系统
    def analyze_case_with_law(self, input_user_id, input_session_id, case_description,
                              case_faiss_indexer, law_faiss_indexer,
                              rag_paragraphs_cases, rag_paragraphs_laws,
                              return_template=True):
        embedding_generator = Embedding("bert-base-chinese")

        # 检索相关案子
        relevant_cases = self.get_relevant_texts(case_description, embedding_generator, rag_paragraphs_cases,
                                                 case_faiss_indexer)
        relevant_cases_text = '\n'.join(relevant_cases)

        role = f"""
        你是{bot_name}, 一位专业的法律分析专家。
        在分析案子前，必须优先参考之前的对话历史记录。若历史记录中有相关信息，要基于此准确作答。
        仅使用用户的语言进行对话，提供准确、专业的法律分析。
        """

        # 生成法律分析的内部函数
        def generate_law_analysis():
            template_law_analysis = f"""
            <|system|>{role}<|end|>
            <|human|>请根据以下相关案子找出适用的法律并分析：{relevant_cases_text}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请详细阐述适用的法律以及分析过程，准确且有条理。<|end|>
            """
            prompt_law_analysis = ChatPromptTemplate.from_template(template_law_analysis)
            history_control = ControlChatHistoryData()
            chain_law_analysis = prompt_law_analysis | self.model_A
            chat_with_history_law_analysis = RunnableWithMessageHistory(
                chain_law_analysis,
                get_session_history=history_control.get_session_history,
                input_message_key="question",
                output_message_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="input_user_id",
                        annotation=str,
                        name="用户ID",
                        description="每个用户的唯一标识",
                        default=""
                    ),
                    ConfigurableFieldSpec(
                        id="input_session_id",
                        annotation=str,
                        name="会话ID",
                        description="每个用户对应的会话的唯一标识",
                        default=""
                    )
                ]
            )
            try:
                law_analysis = chat_with_history_law_analysis.invoke({"question": case_description},
                                                                     config={"input_user_id": input_user_id,
                                                                             "input_session_id": input_session_id})
            except Exception as e:
                print(f"在 analyze_case_with_law 函数的 generate_law_analysis 中调用模型时出错: {e}")
                return None
            history_control.update_session_history(input_user_id, input_session_id, case_description,
                                                   law_analysis.content)
            return law_analysis

        law_analysis = generate_law_analysis()
        if law_analysis is None:
            if return_template:
                return None, None
            else:
                return None, " "

        # 检索相关法律条文
        relevant_laws = self.get_relevant_texts(law_analysis.content, embedding_generator, rag_paragraphs_laws,
                                                law_faiss_indexer)
        relevant_laws_text = '\n'.join(relevant_laws)

        # 生成最终分析的内部函数
        def generate_final_analysis():
            template_final_analysis = f"""
            <|system|>{role}<|end|>
            <|human|>请根据以下相关案子和法律条文对案子进行全面的法律分析：
            相关案子：{relevant_cases_text}
            相关法律条文：{relevant_laws_text}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请综合分析，详细阐述法律条文如何应用于案子，以及案子的法律依据和结论。<|end|>
            """
            prompt_final_analysis = ChatPromptTemplate.from_template(template_final_analysis)
            history_control = ControlChatHistoryData()
            chain_final_analysis = prompt_final_analysis | self.model_A
            chat_with_history_final_analysis = RunnableWithMessageHistory(
                chain_final_analysis,
                get_session_history=history_control.get_session_history,
                input_message_key="question",
                output_message_key="history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="input_user_id",
                        annotation=str,
                        name="用户ID",
                        description="每个用户的唯一标识",
                        default=""
                    ),
                    ConfigurableFieldSpec(
                        id="input_session_id",
                        annotation=str,
                        name="会话ID",
                        description="每个用户对应的会话的唯一标识",
                        default=""
                    )
                ]
            )
            try:
                final_analysis = chat_with_history_final_analysis.invoke({"question": case_description},
                                                                         config={"input_user_id": input_user_id,
                                                                                 "input_session_id": input_session_id})
            except Exception as e:
                print(f"在 analyze_case_with_law 函数的 generate_final_analysis 中调用模型时出错: {e}")
                return None, None
            history_control.update_session_history(input_user_id, input_session_id, case_description,
                                                   final_analysis.content)
            return final_analysis, template_final_analysis

        final_analysis, template_final_analysis = generate_final_analysis()
        if final_analysis is None:
            if return_template:
                return None, None
            else:
                return None, " "

        if return_template:
            return final_analysis, template_final_analysis
        else:
            return final_analysis, " "