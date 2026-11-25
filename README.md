# 📚 智能文档问答系统 (RAG-Based PDF QA System)

一个完全从零手搓的 RAG（检索增强生成）应用。支持解析长文本、向量化存储、语义检索、**Rerank 重排序**、**数据防污染清洗**，并结合本地/云端大模型实现精准问答。

## 🌟 核心亮点

- **全链路自研**：未依赖 LangChain 等重型框架，纯手写实现了文档清洗、语义切块、Embedding 向量化、Qdrant 检索、Rerank 精排与 Prompt 拼接。
- **两阶段检索架构 (Two-Stage Retrieval)**：引入 `bge-reranker-base` 交叉编码器（Cross-Encoder），构建了“向量粗排召回 Top-10 + Rerank 精排 Top-3”的进阶架构，有效解决了向量检索的语义漂移问题。
- **数据防污染与正则拦截**：每次入库前自动销毁并重建 Collection。在经历“高仿脏数据”压力测试后，升级为**自然段语义切块**，并引入**正则白名单过滤**，成功从物理层面切断了垃圾数据的污染。
- **严谨的量化评估体系**：内置基于 **LLM as a Judge（大模型裁判）** 的自动化评估脚本，引入 A/B 测试对比调优效果，量化输出准确率与延迟数据。
- **复杂文档解析方案**：内置针对 PDF 页眉、页脚、参考文献的过滤规则，并解决了 `(cid:xxxx)` 字体加密乱码问题（通过 PaddleOCR 方案）。
- **全栈交互界面**：基于 Gradio 搭建完整的 Web 端交互界面，支持文件上传与多轮对话记录展示。

## 📊 性能评估与调优数据 (Performance & Eval)

我们在测试集上进行了严格的 A/B 测试对比，并进行了极端的“脏数据压力测试”：

| 评估指标 | Rerank 关闭 (纯向量检索) | Rerank 开启 (交叉编码器精排) |
| :--- | :--- | :--- |
| **回答准确率 (纯净数据)** | 100.00% | 100.00% |
| **回答准确率 (高仿噪音压测)** | 50.00% (被误导) | 50.00% (被字面连贯性欺骗) |
| **平均响应延迟** | 0.75 秒/次 | 1.07 秒/次 |

> **压测复盘**：测试证明 Rerank 并非万能，面对逻辑连贯的“专业伪知识”时也会失效。这促成了后续架构升级：通过**语义切块**将真伪数据物理隔离，并通过**正则白名单**在入库前拦截无专业词汇的噪音。优化后，系统成功在混合垃圾数据中保持了 100% 的准确率。

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

---
*Created with ❤️ by wjl-518927. Focusing on high-performance and reliable AI Engineering.*