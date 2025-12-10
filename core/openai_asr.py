import io
import soundfile as sf
from openai import OpenAI
import re

class OpenAIASRService:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.sample_rate = 16000
    
    def _is_hallucination(self, text):
        """
        Kiểm tra xem text có bị lỗi lặp từ (Whisper loop) không.
        Ví dụ: "của quốc tế của quốc tế của quốc tế..."
        """
        if not text:
            return True
            
        # 1. Kiểm tra lặp ký tự vô nghĩa (vd: "cc cc cc", "g g g")
        if re.search(r'\b(\w+)( \1){4,}', text): # Lặp lại 1 từ quá 4 lần
            return True
            
        # 2. Kiểm tra lặp cụm từ (vd: "cộng đồng quốc tế cộng đồng quốc tế...")
        # Lấy 20 ký tự đầu, xem nó có lặp lại quá nhiều trong chuỗi không
        if len(text) > 50:
            prefix = text[:20]
            if text.count(prefix) > 3:
                return True
                
        return False

    def predict(self, audio_data, previous_text=""):
        """
        previous_text: Ngữ cảnh câu trước để Whisper nối từ tốt hơn
        """
        try:
            # Check độ dài audio, quá ngắn (<0.5s) thì bỏ qua để tránh hallucination
            if len(audio_data) < self.sample_rate * 0.5:
                return {}

            wav_buffer = io.BytesIO()
            wav_buffer.name = "audio.wav"
            sf.write(wav_buffer, audio_data, self.sample_rate, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)

            # Gọi API với Prompt (Context)
            # prompt=previous_text giúp model hiểu ngữ cảnh để không bị ngắt quãng
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=wav_buffer,
                language="vi", 
                response_format="json", 
                temperature=0.2, # Tăng nhẹ temp để giảm lặp
                prompt=previous_text[-200:] if previous_text else "" # Chỉ lấy 200 ký tự cuối làm prompt
            )
            
            text_result = transcript.text.strip()
            
            # Lọc ảo giác
            if self._is_hallucination(text_result):
                print(f"⚠️ [FILTERED] Phát hiện ảo giác: {text_result[:50]}...")
                return {"text": "", "confidence": 0.0}

            return {
                "text": text_result,
                "confidence": 0.99
            }

        except Exception as e:
            print(f"❌ OpenAI API Error: {e}")
            return {}