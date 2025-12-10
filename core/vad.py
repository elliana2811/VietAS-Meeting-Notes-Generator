# core/vad.py
import torch
import logging
import numpy as np 

logger = logging.getLogger(__name__)

class VADDetector:
    def __init__(self):
        logger.info("Initializing VAD...")
        try:
            # Thêm trust_repo=True
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True 
            )
            self.get_speech_ts, _, _, _, _ = utils
            logger.info("Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Silero VAD: {e}")
            self.model = None

    def is_speech(self, audio_float32_array, sample_rate=16000):
        if self.model is None:
            return 0.0

        # --- FIX QUAN TRỌNG ---
        # Đảm bảo array liên tục trong bộ nhớ trước khi đưa vào Torch
        if not audio_float32_array.flags['C_CONTIGUOUS']:
            audio_float32_array = np.ascontiguousarray(audio_float32_array)

        try:
            audio_tensor = torch.from_numpy(audio_float32_array)
            
            if len(audio_tensor) < 512:
                return 0.0

            # Xử lý dimension nếu cần (Silero chấp nhận mảng 1 chiều)
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            with torch.no_grad():
                speech_prob = self.model(audio_tensor, sample_rate).item()
            
            return speech_prob
        except Exception as e:
            logger.error(f"VAD Error: {e}")
            return 0.0