from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载同目录下的 .env 文件
load_dotenv() 

# 从环境变量中读取 Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 1. 初始化模型和数据库
print("⏳ 加载模型中...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
client_qdrant = QdrantClient(host="localhost", port=6333)

# 2. 初始化大模型客户端 (把刚才申请的 Key 填在这里)

llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="http://172.24.160.1:11434/v1")

def ask_rag(question):
    print(f"\n🗣️ 用户提问: {question}")
    
    # 第一步：把问题变成向量
    query_vector = model.encode(question).tolist()
    
    # 第二步：去 Qdrant 检索最相关的 1 条文档
    search_result = client_qdrant.query_points(
        collection_name="paper_docs",
        query=query_vector,
        limit=3
    )
    
    # 提取检索到的背景知识
        # 提取检索到的前 3 条背景知识，拼成一整段长文本
    context = "\n\n---\n\n".join([hit.payload['text'] for hit in search_result.points])
    print(f"🔍 检索到的背景知识: {context}")
    print(f"🔍 本次提取的背景知识片段 (前150字):\n{context[:150]}...\n")
    # 第三步：拼装 Prompt，发给大模型
    prompt = f"""
    你是一个智能助手。请严格根据下面提供的背景知识来回答用户的问题。
    背景知识里可能包含一些参考文献、页脚等无关噪音，请忽略它们，只根据正文内容回答问题。
    
    背景知识：
    {context}
    
    用户问题：
    {question}
    """
    
    print("🤖 大模型正在思考并生成回答...")
    response = llm_client.chat.completions.create(
            model="qwen2.5:7b", # 换成你本地下载的模型名字
            messages=[
                {"role": "system", "content": "你是一个严谨的文档问答助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    
    print("\n✅ 最终回答：")
    print(response.choices[0].message.content)

    for hit in search_result.points:
        print(f"得分: {hit.score:.4f} | 内容: {hit.payload['text'][:100]}...")
# 测试
if __name__ == "__main__":
    # 换成关于论文的问题
    ask_rag("这篇论文提出了什么核心方法？")
    ask_rag("作者使用了哪些技术来解决医学图像分割问题？") 