from openai import OpenAI

class ContextFilter:
    def __init__(self, openai_model_name, client):
        self.client = client
        self.openai_model_name = openai_model_name
        

    def filter_context(self, conversation_history):
        context = "\n".join(q + "\n" + a for q, a in conversation_history)
        prompt = f"从以下对话筛选重要上下文信息:\n{{{context}}}"
        try:
            response = self.client.chat.completions.create(
                model = self.openai_model_name,
                message = [
                    {"role": "system", "content": "帮助我筛选上下文信息"},
                    {"role": "user", "content": prompt}
                ]
            )
            lines =  response.choices[0].message.content.split("\n")
            important_context = []
            for i in range(0, len(lines), 2):
                if i+1 < len(lines):
                    a = (lines[i], lines[i+1])
                    important_context.append(a)
            return important_context
        except Exception as e:
            print(f"筛选上下文时发生错误:{e}")
            return None
        
        