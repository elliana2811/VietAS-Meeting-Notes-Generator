import torch
import logging
import torchaudio
from pyannote.audio import Pipeline

# --- CẤU HÌNH ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force backend soundfile 
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

class OfflineDiarizer:
    def __init__(self, hf_token: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initiating Diarization Pipeline on device: {self.device}")
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            if self.pipeline is None:
                raise ValueError("Không thể load pipeline (None).")
            
            self.pipeline.to(self.device)
            logger.info("✅ Diarization Pipeline loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Pyannote Pipeline: {str(e)}")
            raise e

    def process_file(self, audio_path: str) -> dict:
        if not self.pipeline:
            raise RuntimeError("Pipeline chưa được khởi tạo.")

        logger.info(f"Starting diarization for: {audio_path}")
        
        try:
            # 1. Đọc file thủ công bằng Soundfile
            waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
            
            if waveform.numel() == 0:
                return {"speaker_segments": [], "error": "Empty waveform"}

            # Chuyển về Mono nếu cần
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 2. Chạy Inference
            input_data = {"waveform": waveform, "sample_rate": sample_rate}
            result_obj = self.pipeline(input_data)
            
            # --- FIX CHÍNH XÁC CHO LỖI CỦA BẠN ---
            diarization = None
            
            # Kiểm tra xem object có thuộc tính 'speaker_diarization' không (như trong log bạn gửi)
            if hasattr(result_obj, 'speaker_diarization'):
                diarization = result_obj.speaker_diarization
                
            # Fallback cho các trường hợp khác (bản cũ/mới hơn)
            elif hasattr(result_obj, 'itertracks'):
                diarization = result_obj
            elif isinstance(result_obj, tuple):
                diarization = result_obj[0]
            elif hasattr(result_obj, 'annotation'):
                diarization = result_obj.annotation
            else:
                logger.error(f"⚠️ Unknown output type: {type(result_obj)}")
                return {"speaker_segments": [], "error": f"Unknown output format: {type(result_obj)}"}
            # --------------------------------------

            # 3. Trích xuất Segment
            speaker_segments = []
            
            # Lúc này 'diarization' chắc chắn là Annotation object
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3)
                })
            
            result = {
                "speaker_segments": speaker_segments,
                "total_speakers": len(set(s['speaker'] for s in speaker_segments))
            }
            
            logger.info(f"Diarization finished. Found {result['total_speakers']} speakers.")
            return result

        except Exception as e:
            logger.error(f"Error during diarization processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"speaker_segments": [], "error": str(e)}

def diarize_segment(audio_path: str, hf_token: str) -> dict:
    diarizer = OfflineDiarizer(hf_token)
    return diarizer.process_file(audio_path)