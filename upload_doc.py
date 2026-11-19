import fitz
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

print("⏳ 加载向量模型...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
client = QdrantClient(host="localhost", port=6333)

collection_name = "paper_proofread"
if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

pdf_path = "data/UmuGmPJC.pdf" # 替换成你的实际文件名
print(f"📄 正在解析文档: {pdf_path}")
doc = fitz.open(pdf_path)

points = []
for page_num, page in enumerate(doc):
    page_text = page.get_text()
    # 按段落切分（用双换行符分割），保留原始排版
    paragraphs = page_text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        # 太短的段落（比如就几个字）不存
        if len(para) < 20:
            continue
            
        vector = model.encode(para).tolist()
        # 核心：把页码和内容一起存进 payload！
        points.append(
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=vector, 
                payload={"text": para, "page": page_num + 1} # page_num 从0开始，+1才是人看的页码
            )
        )

doc.close()

if points:
    client.upsert(collection_name=collection_name, wait=True, points=points)
    print(f"✅ 成功入库！共处理了 {len(points)} 个段落。")
else:
    print("⚠️ 没有提取到有效文本，请检查 PDF 是否为纯图片扫描版。")