import numpy as np
import av
import queue
import logging
import traceback
from streamlit_webrtc import AudioProcessorBase

logger = logging.getLogger(__name__)

class RealTimeAudioProcessor(AudioProcessorBase):
    def __init__(self, vad_model):
        self.vad_model = vad_model
        self.buffer = np.array([], dtype=np.float32)
        self.output_queue = queue.Queue()
        
        # Cấu hình VAD
        self.is_speaking = False
        self.silence_counter = 0
        self.SILENCE_THRESHOLD = 10
        self.SPEECH_THRESHOLD = 0.5
        
        self.frame_count = 0 

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # 1. Lấy dữ liệu và xử lý Shape (như cũ)
            raw = frame.to_ndarray() 
            if raw.ndim == 2:
                if raw.shape[0] == 2: raw = np.mean(raw, axis=0)
                else: raw = raw.squeeze()
            
            if raw.dtype == np.int16:
                samples = raw.astype(np.float32) / 32768.0
            else:
                samples = raw.astype(np.float32)

            # 2. Resample 48k -> 16k
            # Kết quả: samples có độ dài 640 (vì 1920 / 3 = 640)
            if frame.sample_rate == 48000:
                samples = samples.reshape(-1, 3).mean(axis=1)
            
            if not samples.flags['C_CONTIGUOUS']:
                samples = np.ascontiguousarray(samples)

            # 3. CHUẨN BỊ INPUT CHO VAD (FIX LỖI 640 vs 512)
            # Silero bắt buộc input phải là 512 mẫu
            vad_input = samples
            
            if len(samples) > 512:
                # Nếu dài hơn (640), chỉ lấy 512 mẫu đầu để check VAD
                vad_input = samples[:512]
            elif len(samples) < 512:
                # Nếu ngắn hơn, thêm số 0 vào cho đủ 512
                vad_input = np.pad(samples, (0, 512 - len(samples)))
            
            # 4. Chạy VAD check
            prob = 0.0
            if self.vad_model:
                try:
                    # Đưa đúng 512 mẫu vào model
                    prob = self.vad_model.is_speech(vad_input, 16000)
                except Exception as e:
                    # Nếu vẫn lỗi thì bỏ qua frame này, coi như im lặng
                    prob = 0.0

            # Log
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                if prob > 0.5:
                    print(f"🗣️ ĐANG NÓI (VAD={prob:.2f})")

            # 5. Lưu vào Buffer (LƯU Ý: Lưu đủ 640 mẫu samples, KHÔNG lưu vad_input)
            self.buffer = np.concatenate((self.buffer, samples))

            # 6. Logic Cắt câu
            if prob > self.SPEECH_THRESHOLD:
                self.is_speaking = True
                self.silence_counter = 0
            else:
                if self.is_speaking:
                    self.silence_counter += 1
                    if self.silence_counter >= self.SILENCE_THRESHOLD:
                        self._cut_segment()
        
        except Exception as e:
            if self.frame_count % 50 == 0:
                print(f"❌ Error: {e}")
                # traceback.print_exc()
            
        return frame

    def _cut_segment(self):
        if len(self.buffer) > 8000:
            segment = self.buffer.copy()
            self.output_queue.put(segment)
            print(f"✂️ CẮT AUDIO ({len(segment)/16000:.2f}s)")
        
        self.buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_counter = 0