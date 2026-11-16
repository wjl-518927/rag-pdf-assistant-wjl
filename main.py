from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import fitz  # fite
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

app = FastAPI(title="RAG 论文问答系统")

# 初始化全局单例 (服务启动时加载，避免每次请求都重新加载)
print("⏳ 正在加载模型与数据库连接...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
qdrant_client = QdrantClient(host="localhost", port=6333)
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

COLLECTION_NAME = "paper_docs"
# 确保集合存在
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
print("✅ 服务初始化完成！")


# ================= 接口 1：上传并解析文档 =================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. 检查文件类型
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="目前只支持 PDF 文件")
    
    # 2. 读取文件内容并临时保存
    # 注意：在真实的 WSL 环境里，我们用 /tmp 目录存临时文件
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 3. 解析 PDF
        print(f"📄 正在解析: {file.filename}")
        doc = fitz.open(temp_file_path)
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        
        # ================= 核心清洗逻辑更新 =================
        # A. 去掉所有多余的换行符和空格
        full_text = full_text.replace('\n', '').replace(' ', '')
        
        # B. 核心大招：寻找“参考文献”标志并截断
        # 大部分论文的参考文献标题叫做 "References" 或 "参考文献"
        import re
        # 正则表达式：匹配 References 或 参考文献，且不区分大小写
        split_pattern = re.compile(r'(References|参考文献)', re.IGNORECASE)
        
        # 用正则去搜索，如果找到了，就只保留匹配位置前面的文本
        match = split_pattern.search(full_text)
        if match:
            print(f"✂️ 发现参考文献标志，位置在 {match.start()}，进行截断清理。")
            full_text = full_text[:match.start()]
        # ====================================================

        # 3. 切块 (现在的 full_text 已经非常干净了)
        chunk_size = 300
        chunks = []
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i+chunk_size]
            # 此时已经不需要再判断 "References" 了，因为早被切了
            if len(chunk) < 50: continue
            chunks.append(chunk)
        
            
        # 4. 向量化并存入 Qdrant
        print(f"🔄 正在向量化 {len(chunks)} 个片段...")
        points = []
        for chunk in chunks:
            vector = embedding_model.encode(chunk).tolist()
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk}))
            
        qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
        
        return {"message": "文档上传并解析成功", "file_name": file.filename, "chunks_count": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")
    finally:
        # 记得清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ================= 接口 2：对话问答 =================
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_with_paper(request: ChatRequest):
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        # 1. 检索
        query_vector = embedding_model.encode(question).tolist()
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        
        if not search_result.points:
            return {"answer": "知识库为空，请先上传文档。", "contexts": []}
            
        contexts = [hit.payload['text'] for hit in search_result.points]
        context_text = "\n\n---\n\n".join(contexts)
        
        # 2. 拼装 Prompt 并调用大模型
        prompt = f"""
        你是一个智能助手。请严格根据下面提供的背景知识来回答用户的问题。
        如果你在背景知识中找不到答案，请直接说“我不知道”，不要编造。
        
        背景知识：
        {context_text}
        
        用户问题：
        {question}
        """
        
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的文档问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        return {
            "answer": response.choices[0].message.content,
            "contexts": contexts # 把检索到的原文也返回给前端，方便调试
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回答失败: {str(e)}")

# 测试用，运行时可以访问 http://127.0.0.1:8000/docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)