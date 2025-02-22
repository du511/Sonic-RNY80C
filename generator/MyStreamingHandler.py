from langchain.callbacks.base import BaseCallbackHandler


class MyStreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:  # 返回值为None
        print(token, end="", flush=True)
