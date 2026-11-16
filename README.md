# 📚 RAG-Based PDF QA System

一个基于 FastAPI、Qdrant 向量数据库和 DeepSeek 大模型的私有化文档问答系统。

## ✨ 核心特性
- **全链路手写**：不依赖 LangChain，从零实现 PDF 解析、正则清洗、切块、向量化入库。
- **防幻觉约束**：通过 Prompt 严格限制大模型，仅根据文档回答，不知道则回答“不知道”。
- **数据清洗调优**：利用正则表达式截断，彻底解决论文参考文献与致谢污染检索结果的问题。
- **自动化评估**：内置 Eval 脚本，基于测试集量化检索命中率与回答准确率。
- **容器化部署**：使用 Docker 和 Named Volume 管理数据库持久化。

## 🛠️ 技术栈
- **后端**：Python, FastAPI, Uvicorn
- **向量库**：Qdrant
- **大模型**：DeepSeek API
- **Embedding**：BAAI/bge-small-zh-v1.5
- **文本解析**：PyMuPDF