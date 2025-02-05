from openai import OpenAI

class ResponseGenerator:
    def __init__(self, openai_model_name, temperature, top_p, client):
        self.openai_model_name = openai_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.client = client

    def generate_response(self, question, relevant_context, conversation_history, return_prompt = False):
        context = '\n'.join([q + " " + a for q, a in conversation_history])
        rag_context = '\n'.join(relevant_context)
        wrapped_context = f"这是你与用户对话的上下文记录,有必要的话，简洁明了地根据该对话进行回答:{{{context}}},不要回答无关问题。"
        if relevant_context:
            prompt  = f"""请严格按照以下优先级处理问题：
            当前问题类型判断：'{question}'
            1. **优先判断问题类型**：若用户进行寒暄/无关提问（如问候、天气等），别用RAG文档。
            2. **网络安全问题处理**：当且仅当问题涉及网络安全时：
            a) 结合以下RAG文档中的专业信息
            b) 需与当前对话上下文关联：{wrapped_context}
            c) 确保回答准确简洁，剔除无关技术细节
            3. **拒绝机制**：若RAG文档与问题无关，直接忽略文档内容独立回答
            不需要做任何解释,直接回答即可
            相关RAG文档内容：
            {rag_context}
            """
        else:
            prompt = f"""
            请严格按照以下规则回答问题：
            1. 快速判断问题类型。如果是寒暄（如问候、天气等）或者与网络安全毫无关联的问题，以傲娇可爱的风格直接呛回去并简洁回应，禁止使用RAG文档内容。回复时要干脆利落，不要添加多余解释或废话。
            2. 若问题涉及网络安全，需结合当前对话上下文{wrapped_context}，给出准确简洁的回答，剔除所有无关技术细节。
            当前问题是：'{question}'
            你是一个傲娇可爱的宝宝，在回复中展现出这种性格特点。
            不需要做任何解释,直接回答即可。
            """

        try:
            response = self.client.chat.completions.create(
                model = self.openai_model_name,
                messages = [
                    {"role": "system", "content": "你是rabbit,一个网络安全AI，性格直爽，脾气火爆，说话从不藏着掖着，和用户交流时，就要把这种个性展现出来,根据指示使用RAG文档或独立回答,与用户聊天对话,仅使用用户的语言进行对话"},
                    {"role": "user", "content": prompt}
                ],
                temperature = self.temperature,
                top_p = self.top_p,              
            )
            answer = response.choices[0].message.content
            if return_prompt:
                return answer, prompt
            else:
                return answer
        except Exception as e:
            print(f"抱歉无法回答: {e}")
            if return_prompt:
                return prompt
            else:
                return ""
            
    

        