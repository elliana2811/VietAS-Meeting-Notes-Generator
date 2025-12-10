import logging
from typing import Dict, Optional, Any

# Cố gắng import fastpunct, nếu chưa cài thì cảnh báo
try:
    from fastpunct import FastPunct
except ImportError:
    FastPunct = None

# --- CẤU HÌNH LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Punctuation")

class PunctuationRestorer:
    """
    Module khôi phục dấu câu và viết hoa cho văn bản thô từ ASR.
    Sử dụng thư viện fastpunct.
    """
    
    def __init__(self, model_name: str = None, device: str = 'cpu'):
        """
        Khởi tạo PunctuationRestorer.
        
        Args:
            model_name: Tên model fastpunct (nếu None sẽ dùng mặc định)
            device: 'cuda' hoặc 'cpu'
        """
        if FastPunct is None:
            logger.error("❌ Thư viện 'fastpunct' chưa được cài đặt. Hãy chạy: pip install fastpunct")
            self.model = None
        else:
            logger.info("⏳ Đang tải model Punctuation Restoration...")
            try:
                # FastPunct tự động tải weights nếu chưa có
                self.model = FastPunct()
                logger.info("✅ Punctuation Model Loaded!")
            except Exception as e:
                logger.error(f"❌ Lỗi tải FastPunct: {e}")
                self.model = None

        # 1. Internal text buffer
        self.buffer: str = ""
        self.word_threshold: int = 20  # Ngưỡng số từ để kích hoạt xử lý

    def add_text(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Thêm text thô vào buffer và kiểm tra điều kiện xử lý.
        
        Args:
            raw_text: Chuỗi văn bản thô từ ASR (thường là chữ thường, không dấu câu)
            
        Returns:
            Dict kết quả nếu đủ điều kiện xử lý, ngược lại trả về None.
        """
        if not raw_text or not raw_text.strip():
            return None

        # Nối text mới vào buffer (thêm khoảng trắng nếu cần)
        if self.buffer:
            self.buffer += " " + raw_text.strip()
        else:
            self.buffer = raw_text.strip()

        # Đếm số từ trong buffer hiện tại
        word_count = len(self.buffer.split())

        # 3. Condition to process: Buffer > 20 words
        if word_count >= self.word_threshold:
            logger.info(f"Buffer đầy ({word_count} từ) -> Kích hoạt thêm dấu câu.")
            return self.process_buffer()
        
        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        """
        Hàm cưỡng chế xử lý buffer (dùng khi gặp Long Silence hoặc kết thúc phiên).
        """
        if not self.buffer.strip():
            return None
            
        logger.info("Forcing flush (Silence triggered)...")
        return self.process_buffer()

    def process_buffer(self) -> Dict[str, Any]:
        """
        Gọi model fastpunct để xử lý toàn bộ text trong buffer.
        Sau đó xóa buffer.
        """
        if not self.model:
            # Fallback nếu không có model: Trả về text gốc và xóa buffer
            result = {
                "punctuated_text": self.buffer,
                "word_count": len(self.buffer.split()),
                "status": "fallback_no_model"
            }
            self.buffer = ""
            return result

        try:
            # 4. Call fastpunct model
            # fastpunct nhận input là list string
            input_text = self.buffer
            
            # Model trả về list kết quả
            output = self.model.punct([input_text])
            punctuated_text = output[0] if output else input_text

            # Đếm số từ
            word_count = len(punctuated_text.split())

            # Tạo kết quả
            result = {
                "punctuated_text": punctuated_text,
                "word_count": word_count,
                "status": "success"
            }

            # 5. Remove processed portion from buffer
            # (Ở đây ta xóa sạch vì đã xử lý hết segment tích lũy)
            self.buffer = ""
            
            return result

        except Exception as e:
            logger.error(f"Lỗi khi thêm dấu câu: {e}")
            # Trả về text gốc nếu lỗi
            result = {
                "punctuated_text": self.buffer,
                "word_count": len(self.buffer.split()),
                "status": "error"
            }
            self.buffer = ""
            return result

# --- INSTANCE GLOBAL (SINGLETON) ---
_punct_instance = None

def restore_punctuation(raw_text: str, force_flush: bool = False) -> Optional[Dict[str, Any]]:
    """
    Hàm wrapper để gọi từ App chính dễ dàng hơn.
    
    Args:
        raw_text: Text mới nhận từ ASR
        force_flush: True nếu phát hiện khoảng lặng dài (Long Silence)
    """
    global _punct_instance
    if _punct_instance is None:
        _punct_instance = PunctuationRestorer()

    # Nếu có text mới, thêm vào
    result = None
    if raw_text:
        result = _punct_instance.add_text(raw_text)
    
    # Nếu chưa có kết quả (chưa đủ buffer) NHƯNG bị ép flush (do im lặng)
    if result is None and force_flush:
        result = _punct_instance.flush()
        
    return result