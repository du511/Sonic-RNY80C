from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from generator.chat_history_control import ControlChatHistoryData

#该类集成了两个部分:
#1. 问答模板生成器: 该部分负责根据用户输入  的相关信息生成问答模板,(提示词工程)并将其转换为语言模型可读的输入格式。
#2. 语言模型: 对历史记录的调用,最后结合历史记录和问答模板生成语言模型的输出,作为回答。
class DNetResponseGenerator:
    
    def __init__(self, model):
        self.model = model

    def generate_response(self, input_user_id, input_session_id, question, relevant_context,return_template = False):
        role = """ 
        你是Sonic RNY-80C,一个网络安全AI机器人。
        在回答用户问题前，必须优先参考之前的对话历史记录。若历史记录中有相关信息，要基于此准确作答。
        根据指示使用RAG文档或独立回答,与用户聊天对话,仅使用用户的语言进行对话
        用雌小鬼语气回答毒舌+傲娇+少量颜文字
        """
        rag_context = '\n'.join(relevant_context)
        if relevant_context:
            template = f"""
            <|system|>{role}<|end|>
            <|human|>当前问题：{question}<|end|>
            <|history|>{{history}}<|end|>
            <|assistant|>请严格按照以下优先级处理问题：
            1. 优先判断问题类型：若用户进行寒暄/无关提问（如问候、天气等），别管{rag_context}相关内容。
            2. 网络安全问题处理：当且仅当问题涉及网络安全时：
            确保回答准确简洁，剔除无关技术细节
            不需要做任何解释,直接回答即可
            网络安全问题处理：当且仅当问题涉及网络安全时：
            结合以下RAG文档中的专业信息
            相关RAG文档内容：
            {rag_context}这些内容只有在问题出现与其中内容相关的时候才使用对应内容
            加上颜文字<|end|>
            """
            prompt = ChatPromptTemplate.from_template(template)

        else:
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
            print(f"抱歉无法回答: {e}")
            if return_template:
                return "" ,template
            else:
                return ""        