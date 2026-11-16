from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

print("⏳ 正在加载本地 Embedding 模型 (第一次会下载几十MB)...")
# 1. 加载中文向量化模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print("✅ 模型加载成功！")

# 2. 准备真实文档
documents = [
    "FastAPI 是最好的 Python 后端框架，支持异步。",
    "Qdrant 是一个高性能的向量数据库，用 Rust 编写。",
    "今天天气真不错，适合出去吃火锅。",
    "大模型（LLM）可以直接生成文本，但有时候会产生幻觉。"
]

# 3. 把文字变成真实的向量
# 注意：bge 模型输出的是 512 维的向量
print("🔄 正在将文档转化为向量...")
embeddings = model.encode(documents)
print(f"✅ 转化完成！向量的维度是: {len(embeddings[0])} 维")

# 4. 存入 Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "real_documents"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

# 这里的 size 必须是模型输出的真实维度 512
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE), # 真实语义搜索通常用余弦相似度
)

# 组装数据并存入
points = []
for i, (text, vector) in enumerate(zip(documents, embeddings)):
    points.append(
        PointStruct(id=i+1, vector=vector.tolist(), payload={"text": text})
    )
client.upsert(collection_name=collection_name, wait=True, points=points)
print("✅ 真实数据已存入向量库！")

# 5. 模拟用户提问（RAG 的检索阶段）
query = "什么是向量数据库？"
print(f"\n🗣️ 用户提问: {query}")

# 把提问也变成向量
query_vector = model.encode(query).tolist()

# 去 Qdrant 里搜
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2
)

print("\n🔍 找到的最相关文档：")
for hit in search_result.points:
    print(f"匹配得分: {hit.score:.4f} | 文本内容: {hit.payload['text']}")