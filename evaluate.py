import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化
print("⏳ 加载模型和数据库...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
client_qdrant = QdrantClient(host="localhost", port=6333)
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 读取测试集
with open("eval_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

def evaluate():
    total = len(test_data)
    keyword_hits = 0 # 关键词命中数
    retrieval_hits = 0 # 检索命中数
    
    print(f"🚀 开始评估，共 {total} 道测试题...\n")
    
    for idx, item in enumerate(test_data):
        question = item["question"]
        standard_answer = item["standard_answer"]
        keywords = item["ground_truth_keywords"]
        
        print(f"[{idx+1}/{total}] 🗣️ 提问: {question}")
        
        # 1. 检索 (扩大 limit 到 3)
        query_vector = model.encode(question).tolist()
        search_result = client_qdrant.query_points(
            collection_name="paper_docs",
            query=query_vector,
            limit=3
        )
        context = "\n\n".join([hit.payload['text'] for hit in search_result.points])
        
        # 2. 计算检索命中率 (如果捞出来的背景知识里包含了关键词，就算检索命中)
        ret_hit = any(kw in context for kw in keywords)
        if ret_hit:
            retrieval_hits += 1
            
        # 3. 生成回答
        prompt = f"""
        你是一个智能助手。请严格根据下面提供的背景知识来回答用户的问题。
        如果你在背景知识中找不到答案，请直接说“我不知道”，不要编造。
        
        背景知识：
        {context}
        
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
        answer = response.choices[0].message.content
        
        # 4. 计算回答命中率 (如果生成的回答包含了关键词，就算回答命中)
        ans_hit = any(kw in answer for kw in keywords)
        if ans_hit:
            keyword_hits += 1
            
        print(f"   🤖 回答: {answer[:50]}...") # 只打印前50字
        print(f"   [检索命中]: {'✅' if ret_hit else '❌'} | [回答命中]: {'✅' if ans_hit else '❌'}\n")
        
        # 防止 API 请求过快
        import time
        time.sleep(1)

    # 5. 打印最终报告
    print("="*40)
    print("📊 评估结果报告")
    print(f"测试总数: {total}")
    print(f"检索命中率 (Hit Rate): {retrieval_hits / total * 100:.2f}%")
    print(f"回答准确率 (Accuracy): {keyword_hits / total * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate()