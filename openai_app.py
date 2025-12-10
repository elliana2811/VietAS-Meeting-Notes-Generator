import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import time
import logging
import os
import uuid
import soundfile as sf
import numpy as np
import librosa

# --- IMPORT MODULES ---
from core.vad import VADDetector
from core.audio_processor import RealTimeAudioProcessor
from core.punctuation import restore_punctuation, _punct_instance
from core.openai_asr import OpenAIASRService 
from core.diarization import OfflineDiarizer 
from core.pdf_processor import PDFKnowledgeBase
from core.rag_service import MeetingMinuteGenerator

# Cấu hình Log
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")
st.title("🎙️ AI Meeting Assistant (All-in-One)")

# --- CẤU HÌNH API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 Chưa tìm thấy OPENAI_API_KEY trong .streamlit/secrets.toml")
    st.stop()

if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    st.warning("⚠️ Chưa tìm thấy HF_TOKEN. Chức năng phân biệt người nói sẽ bị tắt.")
    HF_TOKEN = None

# Tạo Session ID cho ChromaDB collection (để không bị lẫn giữa các lần chạy)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .draft-box { padding: 10px; background-color: #f0f2f6; border-radius: 5px; color: #555; font-style: italic; margin-bottom: 10px; border: 1px dashed #ccc; }
    .final-box { padding: 15px; border-left-width: 5px; border-left-style: solid; background-color: #ffffff; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD MODELS & SERVICES ---

# HÀM NỘI BỘ: Chỉ thực hiện logic load model, KHÔNG chứa code UI (st.spinner, st.toast...)
@st.cache_resource
def _get_core_services_cached(session_id):
    """
    Hàm này thực sự load model và được cache.
    Tuyệt đối không dùng st.write, st.spinner, st.toast trong này.
    """
    print("--- Bắt đầu load models ---") # Dùng print thay vì st.write
    
    # VAD & ASR
    vad = VADDetector()
    asr = OpenAIASRService(api_key=API_KEY)
    
    # Diarization
    diarizer = None
    if HF_TOKEN:
        try:
            diarizer = OfflineDiarizer(hf_token=HF_TOKEN)
        except Exception as e:
            # Dùng logging thay vì st.error để tránh lỗi cache
            logging.error(f"Lỗi load Diarization: {e}")
            print(f"Lỗi load Diarization: {e}")
    
    # RAG Services
    pdf_kb = PDFKnowledgeBase(api_key=API_KEY, collection_name=f"meeting_{session_id}")
    rag_gen = MeetingMinuteGenerator(api_key=API_KEY)
    
    # Punctuation Buffer
    restore_punctuation("", force_flush=False)
    
    return vad, asr, diarizer, pdf_kb, rag_gen

# HÀM WRAPPER: Chứa code UI và gọi hàm cache bên trên
def load_core_services():
    """
    Hàm này quản lý giao diện loading và gọi hàm cache.
    Không dùng @st.cache_resource cho hàm này.
    """
    # UI Element 1: Spinner
    with st.spinner("Đang khởi động AI Models (VAD, ASR, Diarization, RAG)..."):
        # Gọi hàm cache thực sự
        models = _get_core_services_cached(st.session_state.session_id)
        
    # UI Element 2: Toast
    st.toast("✅ Hệ thống đã sẵn sàng!", icon="🚀")
    
    return models

# Load tất cả models
vad_model, asr_model, diarizer_model, pdf_service, rag_service = load_core_services()

# --- 2. QUẢN LÝ TRẠNG THÁI (STATE MANAGEMENT) ---
if "transcript_history" not in st.session_state:
    st.session_state.transcript_history = ""
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = [] 
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

def clear_session():
    """Hàm xóa dữ liệu cũ để bắt đầu cuộc họp mới"""
    st.session_state.transcript_history = ""
    st.session_state.full_transcript = []
    st.session_state.final_minutes = ""
    # Reset buffer của punctuation
    restore_punctuation("", force_flush=True)
    st.toast("Đã xóa dữ liệu cũ!", icon="🗑️")

# --- 3. GIAO DIỆN & LOGIC ---

# >>> SIDEBAR: QUẢN LÝ TÀI LIỆU PDF <<<
with st.sidebar:
    st.header("📂 Tài liệu tham khảo")
    uploaded_pdf = st.file_uploader("Upload PDF (Bối cảnh/Tài liệu)", type="pdf")
    
    if uploaded_pdf and not st.session_state.pdf_processed:
        with st.spinner("Đang đọc và vector hóa PDF..."):
            # Lưu file tạm
            pdf_path = f"temp_{st.session_state.session_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            # Xử lý RAG
            pdf_service.process_and_store_pdf(pdf_path)
            st.session_state.pdf_processed = True
            st.success("Đã học xong tài liệu!")
            
            # Dọn dẹp
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    st.divider()
    st.header("⚙️ Chức năng")
    if st.button("🗑️ Xóa dữ liệu cũ & Làm mới", type="primary"):
        clear_session()
        st.rerun()

# >>> MAIN UI: HAI TAB CHÍNH <<<
tab1, tab2 = st.tabs(["🎙️ Ghi âm Real-time", "🎧 Upload File Ghi âm"])

# HÀM HELPER CHUNG: XỬ LÝ 1 CHUNK AUDIO
def process_audio_chunk(audio_chunk, status_container):
    """Hàm xử lý logic cốt lõi: Diarization -> ASR -> Punctuation -> Update UI"""
    
    status_container.warning("🔄 Đang xử lý...")
    
    # A. Diarization
    current_speaker = "Người nói"
    if diarizer_model:
        try:
            temp_wav = "temp_proc.wav"
            sf.write(temp_wav, audio_chunk, 16000)
            diar_res = diarizer_model.process_file(temp_wav)
            
            # Logic Dominant Speaker
            segments = diar_res.get("speaker_segments", [])
            if segments:
                spk_dur = {}
                for s in segments:
                    dur = s['end'] - s['start']
                    spk_dur[s['speaker']] = spk_dur.get(s['speaker'], 0) + dur
                if spk_dur:
                    current_speaker = max(spk_dur, key=spk_dur.get)
            
            if os.path.exists(temp_wav): os.remove(temp_wav)
        except:
            pass

    # B. ASR (OpenAI)
    raw_text = ""
    if asr_model:
        res = asr_model.predict(audio_chunk)
        raw_text = res.get('text', '').strip()

    # C. Punctuation & Update UI
    if raw_text:
        punct_res = restore_punctuation(raw_text, force_flush=False)
        if punct_res:
            final_text = punct_res['punctuated_text']
            add_to_transcript(final_text, current_speaker)
        else:
            # Update Draft UI
            update_draft_ui(raw_text)

def add_to_transcript(text, speaker):
    """Thêm vào lịch sử và render UI"""
    color = {
        "SPEAKER_00": "#00cc66", "SPEAKER_01": "#0099ff", 
        "SPEAKER_02": "#ff9900", "Người nói": "#999999"
    }.get(speaker, "#333333")
    
    # 1. Update HTML History
    st.session_state.transcript_history += (
        f"<div class='final-box' style='border-left-color: {color};'>"
        f"<b style='color:{color}'>{speaker}:</b> {text}</div>"
    )
    
    # 2. Update Structured Data (Cho RAG)
    st.session_state.full_transcript.append({
        "speaker": speaker,
        "text": text
    })

def update_draft_ui(text):
    """Cập nhật placeholder draft (nếu cần)"""
    pass # Logic này được xử lý trực tiếp trong loop

# ================= TAB 1: REAL-TIME RECORDING =================
with tab1:
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.info("Nhấn START để bắt đầu họp.")
        
        def processor_factory():
            return RealTimeAudioProcessor(vad_model=vad_model)

        ctx = webrtc_streamer(
            key="meeting-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=processor_factory,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
        )

    with col_r:
        chat_container = st.container()
        draft_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Render Lịch sử
        with chat_container:
            st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)

        # WEB RTC LOOP
        if ctx.state.playing:
            last_time = time.time()
            while True:
                if ctx.audio_processor:
                    try:
                        audio_chunk = ctx.audio_processor.output_queue.get_nowait()
                        last_time = time.time()
                        
                        # Gọi hàm xử lý chung
                        process_audio_chunk(audio_chunk, status_placeholder)
                        
                        # Force refresh UI sau khi xử lý xong chunk này
                        with chat_container:
                            st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
                            
                    except queue.Empty:
                        # Auto flush
                        if time.time() - last_time > 2.0:
                            flush = restore_punctuation("", force_flush=True)
                            if flush:
                                add_to_transcript(flush['punctuated_text'], "Bot")
                                with chat_container:
                                    st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
                            last_time = time.time()
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)

# ================= TAB 2: UPLOAD AUDIO FILE =================
with tab2:
    st.info("Tải lên file ghi âm cuộc họp (.wav, .mp3) để xử lý.")
    audio_file = st.file_uploader("Chọn file audio", type=["wav", "mp3", "m4a"])
    
    if audio_file:
        st.audio(audio_file)
        if st.button("🚀 Bắt đầu xử lý File"):
            # Clear data cũ trước khi chạy file mới
            clear_session()
            
            status_bar = st.progress(0)
            status_text = st.empty()
            
            # Load file bằng librosa
            with st.spinner("Đang tải file vào bộ nhớ..."):
                y, sr = librosa.load(audio_file, sr=16000)
                duration = librosa.get_duration(y=y, sr=sr)
            
            # Cắt file thành các chunk 10 giây để giả lập luồng xử lý
            chunk_duration = 10 # giây
            total_chunks = int(duration // chunk_duration) + 1
            
            chat_box = st.container()
            
            for i in range(total_chunks):
                start_sample = i * chunk_duration * sr
                end_sample = min((i + 1) * chunk_duration * sr, len(y))
                chunk = y[int(start_sample):int(end_sample)]
                
                if len(chunk) > 0:
                    status_text.text(f"Đang xử lý đoạn {i+1}/{total_chunks}...")
                    
                    # Gọi hàm xử lý chung
                    # (Lưu ý: status_placeholder ở đây là dummy, ta dùng status_text)
                    process_audio_chunk(chunk, st.empty())
                    
                    # Update Progress
                    status_bar.progress((i + 1) / total_chunks)
                    
                    # Update UI Realtime
                    with chat_box:
                        st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
            
            # Flush cuối cùng
            flush = restore_punctuation("", force_flush=True)
            if flush:
                add_to_transcript(flush['punctuated_text'], "End")
                
            st.success("✅ Đã xử lý xong file!")
            with chat_box:
                st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)

# --- 4. SECTION TẠO BIÊN BẢN (CHUNG CHO CẢ 2 TAB) ---
st.divider()
st.subheader("📝 Tạo biên bản cuộc họp")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("📥 Tải Transcript (.txt)"):
        text_content = "\n".join([f"{x['speaker']}: {x['text']}" for x in st.session_state.full_transcript])
        st.download_button("Click tải xuống", text_content, "meeting.txt", "text/plain")

with col_btn2:
    if st.button("🤖 Tạo Biên bản thông minh (RAG PDF)"):
        if not st.session_state.full_transcript:
            st.warning("Chưa có nội dung hội thoại!")
        else:
            full_summary = ""
            progress = st.progress(0)
            
            # 1. Convert transcript to text lines
            raw_lines = [f"{x['speaker']}: {x['text']}" for x in st.session_state.full_transcript]
            
            # 2. Chunking transcript (Gộp mỗi 10 dòng hội thoại để tóm tắt 1 lần)
            chunk_size = 10
            trans_chunks = ["\n".join(raw_lines[i:i+chunk_size]) for i in range(0, len(raw_lines), chunk_size)]
            
            for idx, t_chunk in enumerate(trans_chunks):
                # 3. RAG: Tìm trang PDF liên quan
                relevant_pages = []
                if st.session_state.pdf_processed:
                    relevant_pages = pdf_service.find_relevant_pages(t_chunk)
                
                # 4. LLM Summary
                res = rag_service.generate_minute_with_rag(t_chunk, relevant_pages)
                
                full_summary += f"\n#### Phần {idx+1}\n{res['summary']}\n"
                if res['ref_pages']:
                    full_summary += f"*(Nguồn tham khảo: Trang {res['ref_pages']})*\n"
                
                progress.progress((idx+1)/len(trans_chunks))
            
            st.session_state.final_minutes = full_summary
            st.balloons()

if "final_minutes" in st.session_state and st.session_state.final_minutes:
    st.markdown("---")
    st.markdown("### 📋 KẾT QUẢ BIÊN BẢN")
    st.markdown(st.session_state.final_minutes)