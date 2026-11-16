import fitz
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# 1. 初始化模型和数据库
print("⏳ 加载向量模型...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
client = QdrantClient(host="localhost", port=6333)

# 我们建一个全新的集合来存这篇真实文档
collection_name = "paper_docs"

# 清除旧的（如果存在的话），并重新建一个 512 维的集合
if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

# 2. 读取 PDF 并提取全文
pdf_path = "data/UmuGmPJC.pdf" # 确保你的文件名对得上，或者改成 UmuGmPJC.pdf
print(f"📄 正在解析文档: {pdf_path}")
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()
doc.close()

# 3. 文本清洗和智能切块 (Chunking)
# 先去掉多余的空格和不可见字符
full_text = full_text.replace('\n', '').replace(' ', '')

chunk_size = 300
chunks = []
for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i+chunk_size]
    
    # 【新增过滤规则】
    # 1. 过滤掉太短的碎片
    if len(chunk) < 50: 
        continue
    # 2. 过滤掉包含大量网址、doi、参考文献特征的脏数据
    if "http" in chunk or "doi" in chunk.lower() or "References" in chunk or "CVPR" in chunk:
        continue
        
    chunks.append(chunk)

print(f"✂️ 文档切块完成，成功清洗出 {len(chunks)} 个高质量片段。")

# 4. 批量把切块变成向量，并存入数据库
print("🔄 正在将片段转化为向量并入库...")
points = []
# 为了不让内存撑爆，我们分批处理
for i, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()
    points.append(
        PointStruct(
            id=str(uuid.uuid4()), # 生成随机的唯一 ID
            vector=vector,
            payload={"text": chunk} # 把原文存在 payload 里
        )
    )
    # 打印一下进度
    if (i+1) % 10 == 0 or (i+1) == len(chunks):
        print(f"已处理: {i+1}/{len(chunks)}")

# 真正存入 Qdrant
client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)
print("✅ 所有真实文档数据已成功存入 Qdrant！")