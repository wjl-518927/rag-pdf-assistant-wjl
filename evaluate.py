import json
import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from FlagEmbedding import FlagReranker # 👈 引入 Rerank 库

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ================= 配置区 =================
USE_RERANK =  True  # 👈 开关：True 开启重排序，False 关闭
TOP_K_RETRIEVE = 10 # 初筛捞 10 个
TOP_K_FINAL = 3     # 精排后最终给 3 个

print("⏳ 加载模型...")
os.environ["HF_HUB_OFFLINE"] = "1"
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 加载 Rerank 模型 (这是个交叉编码器，专门用来精读打分的)
if USE_RERANK:
    print("⏳ 加载 Rerank 模型 (bge-reranker-base)...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

client_qdrant = QdrantClient(host="localhost", port=6333)
llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
judge_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

COLLECTION_NAME = "cs_knowledge_base"

def get_context(question):
    # 1. 粗排：向量检索捞 Top-10
    query_vector = embedding_model.encode(question).tolist()
    search_result = client_qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE
    )
    
    if not search_result.points:
        return "知识库为空。"
        
    contexts = [hit.payload['text'] for hit in search_result.points]
    
    # 2. 精排：如果开启 Rerank
    if USE_RERANK:
        # 组装成 [问题, 文本] 的配对，让 Rerank 模型打分
        pairs = [[question, text] for text in contexts]
        scores = reranker.compute_score(pairs)
        
        # 将得分和文本重新绑定，并按得分从高到低排序
        scored_contexts = list(zip(scores, contexts))
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        
        # 取 Top-3
        final_contexts = [text for score, text in scored_contexts[:TOP_K_FINAL]]
    else:
        # 如果不开启，直接取向量库返回的前 3 个
        final_contexts = contexts[:TOP_K_FINAL]
        # 如果你开启了 Rerank，在返回前打印一下排名前三的文本
    if USE_RERANK:
        print("\n🔍 [Rerank ON 捞出的Top-1 文本]:")
        print(scored_contexts[0][1][:150]) # 打印前150字
    else:
        print("\n🔍 [Rerank OFF 捞出的Top-1 文本]:")
        print(contexts[0][:150])    
    return "\n\n---\n\n".join(final_contexts)

def evaluate():
    with open("eval_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    total = len(test_data)
    correct_count = 0
    total_time = 0
    
    print(f"🚀 开始评估，共 {total} 道测试题 (Rerank: {'ON' if USE_RERANK else 'OFF'})...\n")
    
    for idx, item in enumerate(test_data):
        question = item["question"]
        standard_answer = item["standard_answer"]
        
        print(f"[{idx+1}/{total}] 🗣️ 提问: {question}")
        start_time = time.time()
        
        # 获取上下文 (用上了 Rerank)
        context_text = get_context(question)
        
        # 生成回答
        prompt = f"""
        你是一个严谨的文档问答机器。你唯一的任务是从下方的【背景知识】中提取信息来回答用户。
        
        【核心指令】
        1. 如果背景知识中找不到答案，请直接回复：“对不起，在提供的文档中未找到相关答案。”
        2. 如果【背景知识】中的不同段落之间存在明显矛盾，请指出矛盾，并拒绝回答。
        3. 不要使用你预训练的知识。
        
        【背景知识】：
        {context_text}
        
        【用户问题】：
        {question}
        """
        try:
            response = llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个严谨的文档问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"调用模型失败: {e}"
            
        cost_time = time.time() - start_time
        total_time += cost_time
        
        print(f"   🤖 AI回答: {answer}")
        
        # 裁判打分（代码不变）
        judge_prompt = f"""
        你是一个严格的 AI 裁判。请对比【标准答案】和【AI回答】，判断 AI 回答是否正确传达了标准答案的语义。
        如果 AI 回答包含了标准答案的核心意思，或者当标准答案表示“不知道”时 AI 也正确拒绝了回答，请输出 1。
        否则，输出 0。不要输出任何多余的解释，只能输出一个数字。
        【标准答案】：{standard_answer}
        【AI回答】：{answer}
        """
        
        try:
            judge_response = judge_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            if "1" in judge_response.choices[0].message.content:
                print("   ✅ [裁判判定]: 正确\n")
                correct_count += 1
            else:
                print("   ❌ [裁判判定]: 错误或产生幻觉\n")
        except Exception as e:
            print(f"   ⚠️ 裁判打分失败: {e}\n")
            
        time.sleep(0.5)

    print("="*40)
    print(f"📊 评估结果报告 (Rerank: {'ON' if USE_RERANK else 'OFF'})")
    print(f"回答准确率: {correct_count / total * 100:.2f}%")
    print(f"平均响应延迟: {total_time / total:.2f} 秒/次")
    print("="*40)

if __name__ == "__main__":
    evaluate()