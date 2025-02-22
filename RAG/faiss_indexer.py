import faiss
import numpy as np


class FaissIndexer:

    def __init__(self):
        self.index = None

    def build_index(self, texts, embedding_generator):  # 建立索引
        try:
            embeddings = [embedding_generator.get_embedding(text) for text in texts]
            embeddings = np.array(
                [embedding for embedding in embeddings if embedding is not None]
            ).astype(
                "float32"
            )  # faiss要求输入的数据类型为float32且为numpy
            self.index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )  # 该方法需要基于维数进行创建索引，这里创建的是基于维数(即单个向量的维度)的平面扁平索引,这里也是在实例化,该类处理维数是基于第一个维度
            self.index.add(embeddings)
            return self.index
        except Exception as e:
            print("Error in FaissIndexer:", e)
            return None

    def search_index(self, query_embedding, top_k=5):  # 搜索索引
        if self.index is None:
            print("Faiss index is not built yet.")
            return None
        try:
            query_embedding = np.array([query_embedding]).astype("float32")
            distances, indices = self.index.search(query_embedding, top_k)
            unique_indices = list(
                set(indices[0])
            )  # 输出的是和查询向量top_k个最相似的索引的位置编号
            return unique_indices
        except Exception as e:
            print("Error in FaissIndexer:", e)
            return []
