# 📚 PDF RAG Assistant

<div align="center">

**Trò chuyện thông minh với tài liệu PDF bằng AI**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

</div>

---

PDF RAG Assistant là ứng dụng Q&A dựa trên kỹ thuật **RAG** (Retrieval-Augmented Generation). Upload tài liệu PDF và trò chuyện trực tiếp với AI để trích xuất, tổng hợp và tìm kiếm thông tin từ nội dung tài liệu một cách chính xác.

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🤖 **RAG Pipeline** | Trả lời chính xác dựa trên nội dung tài liệu, không bịa đặt |
| 📄 **Multi-PDF Upload** | Upload và xử lý nhiều file PDF cùng lúc |
| ⚡ **Streaming Response** | Hiển thị câu trả lời từng token (real-time) |
| 🧠 **Conversation Memory** | Nhớ ngữ cảnh hội thoại (5 lượt gần nhất) |
| 📋 **Source Attribution** | Hiển thị nguồn tham khảo (file, trang) cho mỗi câu trả lời |
| 💾 **Smart Cache** | Cache embeddings — xử lý lại file cũ < 1 giây |
| 🔍 **MMR Search** | Maximum Marginal Relevance cho kết quả đa dạng |
| 🌐 **Bilingual** | Hỗ trợ hỏi đáp tiếng Việt và tiếng Anh |
| 🎨 **Premium UI** | Giao diện hiện đại với dark mode, gradient, animations |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│   │ Upload   │  │ Chat     │  │ Source Display        │  │
│   │ Multi-PDF│  │ Streaming│  │ (File, Page, Preview) │  │
│   └────┬─────┘  └────┬─────┘  └──────────────────────┘  │
├────────┼──────────────┼──────────────────────────────────┤
│        ▼              ▼                                  │
│   ┌──────────┐  ┌──────────────────────────┐             │
│   │ PDF      │  │ RAG Chain (rag_chain.py) │             │
│   │ Processor│  │  ┌─────────┐ ┌────────┐ │             │
│   │ + Clean  │  │  │ Memory  │ │ Prompt │ │             │
│   │ + Split  │  │  │ (k=5)   │ │ Engine │ │             │
│   └────┬─────┘  │  └─────────┘ └────────┘ │             │
│        │        └──────┬───────────────────┘             │
│        ▼               │                                 │
│   ┌──────────┐         ▼                                 │
│   │ FAISS    │◄──── Retriever (MMR, k=5)                 │
│   │ Vector DB│                                           │
│   │ + Cache  │    ┌───────────────────┐                  │
│   └──────────┘    │ Gemini 2.5 Flash  │                  │
│                   │ (LLM via API)     │                  │
│                   └───────────────────┘                  │
├─────────────────────────────────────────────────────────┤
│  Embedding: paraphrase-multilingual-mpnet-base-v2       │
│  Config: config.py | Logger: utils/logger.py            │
└─────────────────────────────────────────────────────────┘
```

## 📁 Cấu trúc thư mục

```text
PDF_RAG_Assistant/
├── app.py                         # Ứng dụng chính (Streamlit UI)
├── config.py                      # Cấu hình tập trung (models, RAG params)
├── requirements.txt               # Dependencies
├── .env                           # Biến môi trường (API keys) — git ignored
├── .env.example                   # Template cho .env
│
├── models/
│   ├── __init__.py
│   ├── embedding_model.py         # Mô hình Embedding (multilingual, CUDA/CPU)
│   └── llm_model.py               # Mô hình LLM (Gemini 2.5 Flash, streaming)
│
├── processing/
│   ├── __init__.py
│   ├── pdf_processor.py           # Xử lý PDF (multi-file, text cleaning, chunking)
│   └── vector_store.py            # FAISS Vector DB (cache, MMR, merge)
│
├── rag/
│   ├── __init__.py
│   └── rag_chain.py               # RAG pipeline (memory, source attribution, prompt)
│
├── utils/
│   ├── __init__.py
│   ├── session_state.py           # Quản lý session state
│   └── logger.py                  # Logging với rotation
│
├── faiss_cache/                   # Cache embeddings (auto-generated)
└── logs/                          # Log files (auto-generated)
```

## 🛠️ Công nghệ

| Thành phần | Công nghệ |
|---|---|
| **Giao diện** | Streamlit |
| **LLM** | Google Gemini 2.5 Flash |
| **Framework** | LangChain |
| **Embedding** | sentence-transformers (multilingual) |
| **Vector DB** | FAISS |
| **PDF Parser** | PyMuPDF |

## ⚙️ Cài đặt

### 1. Clone & tạo môi trường

```bash
git clone <repository-url>
cd PDF_RAG_Assistant

# Tạo virtual environment
python -m venv rag_env

# Activate (Windows)
rag_env\Scripts\activate

# Activate (macOS/Linux)
source rag_env/bin/activate
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Thiết lập API key

```bash
# Copy file mẫu
cp .env.example .env

# Mở .env và thêm Google API key
# GOOGLE_API_KEY=your_api_key_here
```

> 💡 Lấy API key tại: [Google AI Studio](https://aistudio.google.com/apikey)

### 4. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## 🖥️ Hướng dẫn sử dụng

1. **Upload PDF** — Chọn 1 hoặc nhiều file PDF ở sidebar
2. **Xử lý** — Nhấn "🚀 Xử lý PDF" và đợi hoàn tất
3. **Hỏi đáp** — Nhập câu hỏi trong khung chat
4. **Xem nguồn** — Mở expander "📋 Nguồn tham khảo" dưới mỗi câu trả lời

## ⚙️ Tùy chỉnh cấu hình

Tất cả tham số được quản lý trong `config.py` hoặc override qua `.env`:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `LLM_MODEL_NAME` | `gemini-2.5-flash` | Model LLM |
| `LLM_TEMPERATURE` | `0.3` | Độ sáng tạo (0-1) |
| `CHUNK_SIZE` | `1000` | Kích thước chunk (ký tự) |
| `CHUNK_OVERLAP` | `200` | Overlap giữa chunks |
| `RETRIEVER_K` | `5` | Số chunks retrieve |
| `MEMORY_WINDOW_SIZE` | `5` | Số lượt chat nhớ |

## 🔧 Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| `GOOGLE_API_KEY not configured` | Kiểm tra file `.env` có chứa API key đúng |
| Embedding chậm | Cài PyTorch với CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| PDF không đọc được | Đảm bảo PDF không bị encrypt/password protect |
| Lỗi memory | Giảm `CHUNK_SIZE` hoặc upload file nhỏ hơn |

## 📝 License

MIT License
