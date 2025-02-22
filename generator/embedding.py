import torch
from transformers import AutoTokenizer, AutoModel


class Embedding:
    # 初始化分词器和预训练模型
    def __init__(self, bert_model_name):
        self.bert_model_name = bert_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name)

    # 获取文本的embedding,嵌入向量,并且进一步处理
    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(
                text, truncation=True, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings
        except Exception as e:
            print(f"Error in getting embedding: {e}")
            return None
