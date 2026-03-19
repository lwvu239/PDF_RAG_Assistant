# PDF RAG Assistant

PDF RAG Assistant là một ứng dụng Hỏi & Đáp (Q&A) dựa trên tài liệu PDF, sử dụng kỹ thuật **RAG** (Retrieval-Augmented Generation). Ứng dụng cho phép người dùng tải lên các tệp PDF và trò chuyện trực tiếp với AI để trích xuất, tổng hợp và tìm kiếm thông tin từ nội dung tài liệu một cách chính xác.

## 🚀 Tính năng nổi bật
- **Giao diện thân thiện trực quan:** Xây dựng bằng [Streamlit](https://streamlit.io/).
- **Xử lý PDF tự động:** Trích xuất văn bản từ tài liệu PDF.
- **Lưu trữ Vector mạnh mẽ:** Sử dụng ChromaDB để phân chia văn bản (chunking) và tìm kiếm thông tin liên quan nhanh chóng.
- **Hỗ trợ Local LLM:** Tích hợp mô hình ngôn ngữ lớn và mô hình nhúng (Embedding) thông qua Hugging Face (Transformers, Accelerate, BitsAndBytes).

## 🗂️ Cấu trúc thư mục

```text
rag_llm/
├── app.py                     # File chạy ứng dụng chính (Streamlit UI)
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── .env                       # File chứa các biến môi trường
├── models/
│   ├── embedding_model.py     # Cấu hình và tải mô hình Embedding
│   └── llm_model.py           # Cấu hình và tải mô hình LLM (với lượng tử hóa)
├── processing/
│   ├── pdf_processor.py       # Logic xử lý và đọc file PDF
│   └── vector_store.py        # Logic tạo và quản lý Vector Database (Chroma)
├── rag/
│   └── rag_chain.py           # Thiết lập RAG pipeline (Kết nối LLM và Retriever)
└── utils/                     
    └── session_state.py       # Quản lý trạng thái phiên làm việc (Session state) của Streamlit
```

## 🛠️ Công nghệ sử dụng
- **Giao diện (Frontend):** Streamlit
- **Framework LLM:** LangChain
- **Xử lý tài liệu:** PyPDF
- **Vector Database:** Chroma (langchain-chroma)
- **Deep Learning / Models:** HuggingFace Transformers, PyTorch, Accelerate, BitsAndBytes

## ⚙️ Hướng dẫn cài đặt

1. **Clone repository này hoặc di chuyển vào thư mục dự án:**
   ```bash
   cd d:\AI\rag_llm
   ```

2. **Cài đặt môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv rag_env
   rag_env\Scripts\activate  # Đối với Windows
   ```

3. **Cài đặt các thư viện yêu cầu:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Thiết lập biến môi trường:**
   - Mở (hoặc tạo) file `.env` ở thư mục gốc của dự án.
   - Nếu bạn dùng các API từ bên ngoài như HuggingFace Token, OpenAI API, v.v., hãy thêm vào file này:
     ```env
     # Ví dụ:
     HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
     ```

## 🖥️ Hướng dẫn sử dụng

1. **Khởi chạy ứng dụng:**
   Mở terminal và chạy lệnh sau:
   ```bash
   streamlit run app.py
   ```

2. **Các bước trên giao diện:**
   - Khi giao diện mở lên trên trình duyệt (thường ở địa chỉ `http://localhost:8501`), đợi hệ thống tải mô hình (LLM & Embeddings).
   - Truy cập thanh Sidebar (⚙️ Cài đặt) ở góc trái màn hình.
   - Nhấn **Upload tài liệu PDF** để tải file của bạn lên.
   - Nhấn **🔄 Xử lý PDF** để hệ thống đọc, chia nhỏ (chunking) và đưa vào Vector Database.
   - Sau khi xử lý hoàn tất, bạn có thể bắt đầu nhắn tin vào khung chat ở màn hình chính để hỏi đáp về nội dung tài liệu.

## 📝 Lưu ý
- Việc tải mô hình ban đầu (LLM và Embeddings) có thể mất một chút thời gian tùy thuộc vào cấu hình phần cứng của bạn và dung lượng của các model HuggingFace.
- Nếu ứng dụng chạy chậm, vui lòng kiểm tra việc tối ưu hóa với GPU (đã được hỗ trợ qua thư viện `bitsandbytes` và `accelerate`).
