# 📚 智能文档问答系统 (RAG-Based PDF QA System)

一个完全从零手搓的 RAG（检索增强生成）应用。支持解析长文本、向量化存储、语义检索与本地/云端大模型问答。

## 🌟 核心亮点
- **全链路自研**：未依赖 LangChain 等重型框架，纯手写实现了文档清洗、分块、Embedding 向量化、Qdrant 检索与 Prompt 拼接。
- **数据清洗机制**：内置针对 PDF 页眉、页脚、参考文献的过滤规则，并解决了 `(cid:xxxx)` 字体加密乱码问题（通过引入 PaddleOCR 方案）。
- **防幻觉约束**：通过强系统 Prompt 约束大模型，确保其仅基于背景知识回答，不知为不知。
- **向量库防污染**：每次入库前自动销毁并重建 Collection，避免新旧数据混杂导致检索劣化。
- **全栈交互界面**：基于 Gradio 搭建了完整的 Web 端交互界面，支持文件上传、对话记录展示。

## 🛠️ 技术栈
- **前端交互**: Gradio
- **后端服务**: FastAPI, Uvicorn
- **向量数据库**: Qdrant (Docker 部署)
- **关系型数据库**: PostgreSQL (Docker 部署)
- **向量化模型**: BAAI/bge-small-zh-v1.5
- **本地推理**: Ollama (Qwen2.5-7B)
- **云端大模型**: DeepSeek API
- **文档解析**: PyMuPDF, PaddleOCR

## 🚀 快速启动
1. 确保 Qdrant 容器正在运行 (`docker start qdrant-db`)
2. 激活虚拟环境 (`source venv/bin/activate`)
3. 安装依赖 (`pip install -r requirements.txt` - *建议后续生成一个*)
4. 启动 Web UI (`python app.py`)
5. 访问 `http://127.0.0.1:7860`