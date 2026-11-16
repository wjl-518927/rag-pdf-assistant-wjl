from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

# 1. 连接到本地的 Qdrant
client = QdrantClient(host="localhost", port=6333)

collection_name = "my_documents"

# 2. 规范地创建集合（替代已弃用的 recreate_collection）
if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)

# 3. 准备数据并存入
client.upsert(
    collection_name=collection_name,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"text": "FastAPI 是最好的 Python 后端框架"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"text": "Qdrant 是一个高性能的向量数据库"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"text": "今天天气真不错"}),
    ],
)
print("✅ 数据成功存入 Qdrant！")

# 4. 使用新版 query_points 接口进行相似度检索
search_result = client.query_points(
    collection_name=collection_name,
    query=[0.2, 0.8, 0.7, 0.1],  # 注意新接口直接用 query 传入向量
    limit=2
)

print("\n🔍 检索结果：")
for hit in search_result.points:
    print(f"匹配得分: {hit.score:.4f} | 文本内容: {hit.payload['text']}")