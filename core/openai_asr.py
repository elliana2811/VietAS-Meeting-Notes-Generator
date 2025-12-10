import io
import numpy as np
import soundfile as sf
from openai import OpenAI

class OpenAIASRService:
    def __init__(self, api_key):
        """
        Khởi tạo OpenAI Client
        """
        self.client = OpenAI(api_key=api_key)
        self.sample_rate = 16000 # Sample rate chuẩn của luồng xử lý
        print("✅ OpenAI Whisper ASR Service đã sẵn sàng!")

    def predict(self, audio_data):
        """
        Input: numpy array (âm thanh thô)
        Output: Dictionary {'text': ..., 'confidence': ...}
        """
        try:
            # 1. Chuẩn hóa dữ liệu âm thanh
            # OpenAI cần file, ta sẽ tạo một file ảo trong RAM (BytesIO)
            # audio_data thường là float32 [-1, 1] hoặc int16
            
            # Tạo buffer bộ nhớ
            wav_buffer = io.BytesIO()
            wav_buffer.name = "audio.wav" # Đặt tên giả để API nhận diện định dạng
            
            # Ghi numpy array vào buffer dưới dạng WAV
            sf.write(wav_buffer, audio_data, self.sample_rate, format='WAV', subtype='PCM_16')
            
            # Đưa con trỏ file về đầu để đọc
            wav_buffer.seek(0)

            # 2. Gọi OpenAI API (Model Whisper-1)
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=wav_buffer,
                language="vi", # Bắt buộc tiếng Việt để tăng độ chính xác
                response_format="json", # Nhận về JSON đầy đủ
                temperature=0.0 # 0.0 để kết quả ổn định nhất
            )
            
            text_result = transcript.text.strip()

            # Whisper không trả về confidence score cho từng từ ở mode đơn giản
            # Ta giả lập confidence = 1.0 (hoặc dùng verbose_json nếu cần chi tiết)
            
            return {
                "text": text_result,
                "confidence": 0.99, # Whisper rất chính xác
                "tokens": [] # API không trả token ID giống model local, để rỗng
            }

        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "tokens": []
            }

