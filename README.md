# 📚 智能文档问答系统 (RAG-Based PDF QA System)

一个完全从零手搓的 RAG（检索增强生成）应用。支持解析长文本、向量化存储、语义检索、**Rerank 重排序** 与本地/云端大模型问答。

## 🌟 核心亮点
- **全链路自研**：未依赖 LangChain 等重型框架，纯手写实现了文档清洗、分块、Embedding 向量化、Qdrant 检索、Rerank 精排与 Prompt 拼接。
- **两阶段检索架构 (Two-Stage Retrieval)**：引入 `bge-reranker-base` 交叉编码器（Cross-Encoder），构建了“向量粗排召回 Top-10 + Rerank 精排 Top-3”的进阶架构，有效解决了向量检索的语义漂移问题。
- **严谨的量化评估体系**：内置基于 **LLM as a Judge（大模型裁判）** 的自动化评估脚本，引入 A/B 测试对比调优效果，输出准确率与延迟数据。
- **数据清洗与防污染**：内置针对 PDF 页眉、页脚、参考文献的过滤规则，并解决了 `(cid:xxxx)` 字体加密乱码问题（通过 PaddleOCR 方案）。每次入库前自动销毁并重建 Collection，避免历史脏数据污染。
- **全栈交互界面**：基于 Gradio 搭建完整的 Web 端交互界面，支持文件上传与多轮对话记录展示。

## 📊 性能评估与调优数据 (Performance & Eval)
我们在测试集上进行了严格的 A/B 测试对比（基于 DeepSeek-Chat 模型）：

| 评估指标 | Rerank 关闭 (Bi-Encoder 检索) | Rerank 开启 (Cross-Encoder 精排) |
| :--- | :--- | :--- |
| **回答准确率** | 100.00% (纯净数据) / 50.00% (高仿噪音) | 100.00% (纯净数据) / 50.00% (高仿噪音) |
| **平均延迟** | 0.75 秒/次 | 1.07 秒/次 |
*（注：在高仿噪音压力测试中，验证了 Rerank 并非万能，深刻认识到数据清洗与切块策略才是 RAG 系统的上限。）*

## 🛠️ 技术栈
- **前端交互**: Gradio
- **后端服务**: FastAPI, Uvicorn
- **向量数据库**: Qdrant (Docker 部署)
- **关系型数据库**: PostgreSQL (Docker 部署)
- **Embedding 向量化**: BAAI/bge-small-zh-v1.5
- **Rerank 重排序**: BAAI/bge-reranker-base
- **本地推理**: Ollama (Qwen2.5-7B)
- **云端大模型**: DeepSeek API
- **文档解析**: PyMuPDF, PaddleOCR

## 🚀 快速启动
1. 确保 Qdrant 容器正在运行 (`docker start qdrant-db`)
2. 激活虚拟环境 (`source venv/bin/activate`)
3. 启动 Web UI (`python app.py`)
4. 访问 `http://127.0.0.1:7860`