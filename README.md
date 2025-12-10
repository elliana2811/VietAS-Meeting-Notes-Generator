# 🎙️ AI Meeting Assistant

AI Meeting Assistant là ứng dụng web hỗ trợ ghi âm, gỡ băng (transcription), phân biệt người nói và tạo biên bản cuộc họp tự động dựa trên tài liệu tham khảo (RAG). Ứng dụng được xây dựng bằng Streamlit, OpenAI Whisper, Pyannote và ChromaDB.

---

## ✨ Tính năng chính

### 1. Ghi âm và Gỡ băng Real-time
Ghi âm trực tiếp từ trình duyệt bằng WebRTC và chuyển đổi sang văn bản bằng Whisper của OpenAI với độ chính xác cao.

### 2. Xử lý File Ghi âm
Hỗ trợ upload file `.wav` hoặc `.mp3` để xử lý offline.  
Tự động chia nhỏ file (Smart Splitting) để tránh lỗi lặp từ khi chạy Whisper.

### 3. Nhận diện người nói (Speaker Diarization)
Tích hợp `pyannote.audio` để phân biệt từng người nói (Speaker A, B...).

### 4. RAG với Tài liệu PDF
Cho phép tải tài liệu PDF, thực hiện vector hóa bằng ChromaDB và dùng làm ngữ cảnh khi tạo biên bản cuộc họp để đảm bảo thông tin chính xác, không bịa số liệu.

### 5. Sinh Biên bản Cuộc họp Tự động
Sử dụng GPT-4o hoặc GPT-3.5 để:
- Tóm tắt nội dung
- Trích xuất ý chính
- Liệt kê action items
- Tổng hợp biên bản cuộc họp hoàn chỉnh

### 6. Traceability
Theo dõi toàn bộ log xử lý qua Terminal để kiểm soát luồng dữ liệu.

---

## 🛠️ Cài đặt và Chạy ứng dụng (Local Development)

### Yêu cầu hệ thống
- Python 3.10+
- FFmpeg đã được cài đặt sẵn và thêm vào PATH.

### 1. Clone repository

git clone https://github.com/your-username/ai-meeting-assistant.git
cd ai-meeting-assistant
2. Tạo môi trường ảo
code
Bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3. Cài đặt thư viện
code
Bash
pip install -r requirements.txt
Lưu ý: Nếu dùng Windows và gặp lỗi thư viện âm thanh, hãy cài thêm FFmpeg qua Conda:
code
Bash
conda install -c conda-forge ffmpeg
🔑 Cấu hình API Keys
Ứng dụng yêu cầu tạo file cấu hình bí mật để chứa API Key. Hãy tạo file theo đường dẫn sau:
File: .streamlit/secrets.toml
Nội dung mẫu:
code
Toml
# 1. OpenAI Key (bắt buộc) cho Whisper + GPT
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 2. HuggingFace Token (không bắt buộc nếu không dùng diarization)
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
⚠️ Lưu ý quan trọng về HuggingFace Token
Để dùng tính năng phân biệt người nói (pyannote), bạn cần:
Tạo HuggingFace Token (chọn quyền READ).
Truy cập các đường link sau và nhấn Accept License (Đồng ý điều khoản):
pyannote/segmentation-3.0
pyannote/speaker-diarization-3.1
pyannote/speaker-diarization-community-1
▶️ Chạy ứng dụng
Sau khi cài đặt xong, chạy lệnh sau để khởi động:
code
Bash
streamlit run app.py
📂 Cấu trúc dự án
code
Text
ai-meeting-assistant/
├── app.py                  # Streamlit UI và điều phối logic chính
├── requirements.txt        # Danh sách thư viện Python
├── core/
│   ├── vad.py              # Voice Activity Detection (Phát hiện giọng nói)
│   ├── openai_asr.py       # Xử lý gỡ băng qua Whisper API
│   ├── diarization.py      # Nhận diện người nói (Pyannote)
│   ├── pdf_processor.py    # Vector hóa PDF bằng ChromaDB
│   ├── rag_service.py      # Logic RAG kết hợp transcript + PDF
│   ├── audio_processor.py  # Xử lý audio real-time
│   └── punctuation.py      # Xử lý dấu câu và đệm text
├── storage/                # Thư mục lưu dữ liệu Vector DB (Chroma)
└── .streamlit/
    └── secrets.toml        # API Keys (Không commit file này lên Git)
