import gradio as gr
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 强制使用本地模型缓存
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= 初始化 =================
print("⏳ 正在加载模型与数据库...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
qdrant_client = QdrantClient(host="localhost", port=6333)

# ================= 大模型配置 =================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL_NAME = "deepseek-chat"

COLLECTION_NAME = "cs_knowledge_base"

if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

# ================= 后端逻辑函数 =================
def process_pdf(file):
    txt_path = "data/test_clean.txt"
    if not os.path.exists(txt_path):
        return "⚠️ 找不到 data/test_clean.txt 文件！"
        
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # 每次解析前，先清空旧集合，防止历史数据污染
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print("🧹 检测到旧数据，已自动清空向量库。")
        
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    
    # ================= 语义切块 =================
    paragraphs = full_text.split('\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current_chunk) + len(p) < 400:
            current_chunk += p + "\n"
        else:
            if len(current_chunk) > 50:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
            
    if len(current_chunk) > 50:
        chunks.append(current_chunk.strip())

    # ================= 正则二次审查（白名单保底） =================
    print(f"🕵️ 正在进行正则过滤，共 {len(chunks)} 个片段...")
    
    # 只要包含这些核心词，就认为是有效数据
    valid_keywords = ["U-Net", "UCTransNet", "CTrans", "CCT", "CCA", "Transformer", "编码器", "解码器", "医学图像", "分割"]
    
    clean_chunks = []
    for chunk in chunks:
        # 只要包含白名单中的任何一个词，就保留
        if any(k in chunk for k in valid_keywords):
            clean_chunks.append(chunk)
        else:
            print(f"🗑️ 已过滤无关块: {chunk[:20]}...")
            
    if not clean_chunks:
        print("⚠️ 警告：过滤太严，所有块都被删光了！请检查切块或关键词。")
        return "⚠️ 解析失败：所有片段都被判定为无效噪音。请调整切块策略或过滤词表。"
        
    print(f"✅ 审查完成，保留了 {len(clean_chunks)} 个有效片段。")

    # ================= 向量化入库 =================
    points = []
    for chunk in clean_chunks: 
        vector = embedding_model.encode(chunk).tolist()
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk}))
        
    qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
    return f"✅ 解析成功！共存入 {len(clean_chunks)} 个纯净片段。现在可以开始提问了！"


def chat_with_paper(question, history):
    if not question:
        return "", history
    
    query_vector = embedding_model.encode(question).tolist()
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )
    
    if not search_result.points:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "知识库为空，请先上传文档。"})
        return "", history
        
    contexts = [hit.payload['text'] for hit in search_result.points]
    context_text = "\n\n---\n\n".join(contexts)
    
    print(f"\n🔍【Qdrant 捞出的真实原文】:\n{context_text}\n")

    prompt = f"""
    【系统指令】
    你是一个严谨的文档问答机器。你唯一的任务是从下方的【背景知识】中提取信息来回答用户。
    
    【严禁事项】
    1. 绝对禁止使用你预训练记忆中的任何知识！哪怕你确信答案，只要【背景知识】中没有，你也必须装作不知道。
    2. 如果【背景知识】中的内容与问题无关，请直接且仅回复这一句话：“对不起，在提供的文档中未找到相关答案。”
    3. 不要向用户解释原因，不要输出文档之外的内容。

    【背景知识】：
    {context_text}

    【用户问题】：
    {question}
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的文档问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ 调用模型失败: {str(e)}"
        
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history

# ================= 前端界面 =================
with gr.Blocks(title="文档智能问答系统") as demo:
    gr.Markdown("# 📚 专属文档问答系统")
    gr.Markdown("请确保 data/test_clean.txt 文件已准备好，点击解析入库后即可提问。")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传文件 (目前代码直接读取 txt)", file_types=[".pdf"])
            upload_btn = gr.Button("开始解析文档", variant="primary")
            status_text = gr.Textbox(label="解析状态", interactive=False, lines=3)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话记录", height=500)
            with gr.Row():
                msg = gr.Textbox(label="输入你的问题", placeholder="例如：CTrans模块由哪两个子模块组成？", scale=4)
                clear = gr.Button("清空对话")
                
    upload_btn.click(process_pdf, inputs=file_input, outputs=status_text)
    msg.submit(chat_with_paper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)