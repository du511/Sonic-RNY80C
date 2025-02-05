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
        prompt = f"根据上下文回复并且自行判断是否需要结合rag文档:\
        \n 优先考虑用户提出的问题:{question}, \
        \n 上下文:{wrapped_context}, \
        \n rag文档:{rag_context}, "
         
        try:
            response = self.client.chat.completions.create(
                model = self.openai_model_name,
                messages = [
                    {"role": "system", "content": "你是rabbit,一个网络安全AI，根据指示使用RAG文档或独立回答,与用户聊天对话,仅使用用户的语言进行对话"},
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

        