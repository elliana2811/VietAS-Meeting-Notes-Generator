import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import time
import logging
import os
import uuid
import soundfile as sf
import librosa

# --- IMPORT MODULES ---
from core.vad import VADDetector
from core.audio_processor import RealTimeAudioProcessor
from core.punctuation import restore_punctuation
from core.openai_asr import OpenAIASRService 
from core.diarization import OfflineDiarizer 
from core.pdf_processor import PDFKnowledgeBase
from core.rag_service import MeetingMinuteGenerator

# Cấu hình Log để in ra Terminal đẹp hơn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Meeting Assistant", layout="wide")
st.title("🎙️ AI Meeting Assistant (Traceable RAG)")

# --- CẤU HÌNH API KEYS ---
if "OPENAI_API_KEY" in st.secrets:
    API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    st.error("🚨 Chưa tìm thấy OPENAI_API_KEY")
    st.stop()

if "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]
else:
    HF_TOKEN = None

# Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# CSS
st.markdown("""
<style>
    .draft-box { padding: 10px; background-color: #f0f2f6; border: 1px dashed #ccc; margin-bottom: 5px;}
    .final-box { padding: 15px; border-left: 5px solid #00cc66; background-color: #fff; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .log-box { font-family: monospace; font-size: 12px; background: #333; color: #0f0; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD SERVICES ---
@st.cache_resource
def _get_core_services_cached(session_id):
    print(f"\n🚀 [SYSTEM] KHỞI TẠO SERVICES CHO SESSION: {session_id}")
    vad = VADDetector()
    asr = OpenAIASRService(api_key=API_KEY)
    diarizer = OfflineDiarizer(hf_token=HF_TOKEN) if HF_TOKEN else None
    
    # PDF Service
    pdf_kb = PDFKnowledgeBase(api_key=API_KEY, collection_name=f"meeting_{session_id}")
    
    # RAG Service
    rag_gen = MeetingMinuteGenerator(api_key=API_KEY)
    
    restore_punctuation("", force_flush=False)
    return vad, asr, diarizer, pdf_kb, rag_gen

def load_core_services():
    with st.spinner("Đang khởi động AI Models..."):
        models = _get_core_services_cached(st.session_state.session_id)
    return models

vad_model, asr_model, diarizer_model, pdf_service, rag_service = load_core_services()

# --- 2. STATE MANAGEMENT ---
if "transcript_history" not in st.session_state: st.session_state.transcript_history = ""
if "full_transcript" not in st.session_state: st.session_state.full_transcript = [] 
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state: st.session_state.pdf_name = ""

def clear_session():
    st.session_state.transcript_history = ""
    st.session_state.full_transcript = []
    st.session_state.final_minutes = ""
    restore_punctuation("", force_flush=True)
    st.toast("Đã xóa dữ liệu cũ!", icon="🗑️")

# --- 3. UI SIDEBAR (PDF FLOW) ---
with st.sidebar:
    st.header("1. Tài liệu (PDF)")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    
    # Xử lý PDF
    if uploaded_pdf:
        # Kiểm tra nếu file mới khác file cũ hoặc chưa process
        if uploaded_pdf.name != st.session_state.pdf_name:
            st.info(f"🔄 Đang xử lý PDF: {uploaded_pdf.name}...")
            print(f"\n📄 [PDF FLOW] Bắt đầu xử lý file: {uploaded_pdf.name}")
            
            pdf_path = f"temp_{st.session_state.session_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            
            # Gọi service xử lý
            pdf_service.process_and_store_pdf(pdf_path)
            
            # Cập nhật State
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = uploaded_pdf.name
            
            if os.path.exists(pdf_path): os.remove(pdf_path)
            print(f"✅ [PDF FLOW] Hoàn tất vector hóa PDF.\n")
    
    # Hiển thị trạng thái PDF
    if st.session_state.pdf_processed:
        st.success(f"✅ Đã học: {st.session_state.pdf_name}")
    else:
        st.warning("⚠️ Chưa có tài liệu tham khảo.")

    st.divider()
    if st.button("🗑️ Reset Cuộc họp"):
        clear_session()
        st.rerun()

# --- 4. MAIN UI (AUDIO FLOW) ---
tab1, tab2 = st.tabs(["🎙️ Real-time", "🎧 Upload File"])

# Helper functions
def add_to_transcript(text, speaker):
    color = {"SPEAKER_00": "#00cc66", "SPEAKER_01": "#0099ff", "Người nói": "#999999"}.get(speaker, "#333333")
    st.session_state.transcript_history += f"<div class='final-box' style='border-left-color: {color};'><b style='color:{color}'>{speaker}:</b> {text}</div>"
    st.session_state.full_transcript.append({"speaker": speaker, "text": text})

def process_chunk_logic(audio_chunk):
    # 1. Diarization
    speaker = "Người nói"
    if diarizer_model:
        try:
            temp_wav = "temp_proc.wav"
            sf.write(temp_wav, audio_chunk, 16000)
            diar = diarizer_model.process_file(temp_wav)
            # Dominant speaker logic
            segs = diar.get("speaker_segments", [])
            if segs:
                durations = {}
                for s in segs: durations[s['speaker']] = durations.get(s['speaker'], 0) + (s['end'] - s['start'])
                speaker = max(durations, key=durations.get)
            if os.path.exists(temp_wav): os.remove(temp_wav)
        except: pass
    
    # 2. ASR
    raw_text = ""
    if asr_model:
        res = asr_model.predict(audio_chunk)
        raw_text = res.get('text', '').strip()
    
    # 3. Punctuation & Add
    if raw_text:
        punct = restore_punctuation(raw_text, force_flush=False)
        if punct:
            add_to_transcript(punct['punctuated_text'], speaker)
        return raw_text # Trả về để biết có text hay không
    return None

# --- TAB 1: REAL-TIME ---
with tab1:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        def factory(): return RealTimeAudioProcessor(vad_model=vad_model)
        ctx = webrtc_streamer(key="rec", mode=WebRtcMode.SENDONLY, audio_processor_factory=factory,
                              rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    with col_r:
        chat_box = st.container()
        status_txt = st.empty()
        with chat_box: st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
        
        if ctx.state.playing:
            while True:
                if ctx.audio_processor:
                    try:
                        chunk = ctx.audio_processor.output_queue.get_nowait()
                        status_txt.info("⚡ Đang xử lý...")
                        res = process_chunk_logic(chunk)
                        if res: 
                            with chat_box: st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
                            status_txt.empty()
                    except queue.Empty:
                        time.sleep(0.1)

# ================= TAB 2: UPLOAD AUDIO FILE (SỬA LẠI) =================
with tab2:
    st.info("Tải lên file ghi âm cuộc họp (.wav, .mp3) để xử lý.")
    audio_file = st.file_uploader("Chọn file audio", type=["wav", "mp3", "m4a"])
    
    if audio_file:
        st.audio(audio_file)
        if st.button("🚀 Bắt đầu xử lý File"):
            # Clear data cũ
            clear_session()
            print(f"\n🎧 [AUDIO FLOW] Bắt đầu xử lý file audio: {audio_file.name}")
            
            with st.spinner("Đang tải và phân tích file..."):
                # 1. Load file
                y, sr = librosa.load(audio_file, sr=16000)
                
                # 2. Smart Splitting (Cắt bỏ khoảng lặng)
                # top_db=25: Các âm thanh nhỏ hơn 25dB so với peak sẽ bị coi là im lặng
                # frame_length, hop_length: Cấu hình cửa sổ quét
                non_silent_intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
                
                total_segments = len(non_silent_intervals)
                print(f"✂️ Đã cắt thành {total_segments} đoạn hội thoại (bỏ qua khoảng lặng).")
                
                status_bar = st.progress(0)
                status_text = st.empty()
                chat_box_file = st.container()
                
                # Biến lưu context để gửi cho Whisper (giúp nối từ tốt hơn)
                previous_context = ""
                
                # 3. Duyệt qua từng đoạn hội thoại thực sự
                for i, (start, end) in enumerate(non_silent_intervals):
                    # Lấy đoạn audio
                    chunk = y[start:end]
                    
                    # Nếu đoạn quá ngắn (< 0.5s) thì bỏ qua
                    duration = (end - start) / sr
                    if duration < 0.5:
                        continue
                        
                    # Hiển thị log
                    status_text.text(f"Đang xử lý đoạn {i+1}/{total_segments} ({duration:.1f}s)...")
                    if i % 5 == 0: print(f"   ⏳ [AUDIO] Processing segment {i+1}/{total_segments}")
                    
                    # --- GỌI XỬ LÝ (SỬA LẠI LOGIC GỌI) ---
                    # Logic tách ra để truyền previous_context vào
                    
                    # A. Diarization (Vẫn chạy như cũ)
                    speaker = "Người nói"
                    if diarizer_model:
                        try:
                            temp_wav = "temp_proc.wav"
                            sf.write(temp_wav, chunk, 16000)
                            diar = diarizer_model.process_file(temp_wav)
                            segs = diar.get("speaker_segments", [])
                            if segs:
                                durations = {}
                                for s in segs: durations[s['speaker']] = durations.get(s['speaker'], 0) + (s['end'] - s['start'])
                                speaker = max(durations, key=durations.get)
                            if os.path.exists(temp_wav): os.remove(temp_wav)
                        except: pass
                    
                    # B. ASR (OpenAI) - TRUYỀN THÊM PREVIOUS CONTEXT
                    raw_text = ""
                    if asr_model:
                        # Lưu ý: method predict cần update ở core/openai_asr.py để nhận tham số thứ 2
                        res = asr_model.predict(chunk, previous_text=previous_context)
                        raw_text = res.get('text', '').strip()
                    
                    # C. Update UI & Context
                    if raw_text:
                        # Cập nhật context cho vòng lặp sau
                        previous_context = raw_text 
                        
                        punct = restore_punctuation(raw_text, force_flush=False)
                        if punct:
                            add_to_transcript(punct['punctuated_text'], speaker)
                            with chat_box_file: 
                                st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)
                    
                    # Update Progress
                    status_bar.progress((i + 1) / total_segments)
            
            # Flush cuối cùng
            flush = restore_punctuation("", force_flush=True)
            if flush:
                add_to_transcript(flush['punctuated_text'], "End")
                
            st.success("✅ Đã xử lý xong File!")
            with chat_box_file:
                st.markdown(st.session_state.transcript_history, unsafe_allow_html=True)

# --- 5. RAG GENERATION (LOGIC GHÉP NỐI) ---
st.divider()
st.subheader("📝 Tạo biên bản & RAG Log")

if st.button("🤖 Tạo Biên bản thông minh"):
    if not st.session_state.full_transcript:
        st.warning("Chưa có nội dung hội thoại!")
    else:
        print("\n==================================================")
        print("🤖 [RAG START] BẮT ĐẦU QUY TRÌNH TẠO BIÊN BẢN")
        print(f"📊 Tổng số câu hội thoại: {len(st.session_state.full_transcript)}")
        print(f"📚 Trạng thái PDF: {'Đã có' if st.session_state.pdf_processed else 'Không có'}")
        print("==================================================\n")

        full_summary = ""
        
        # 1. Convert transcript to text lines
        raw_lines = [f"{x['speaker']}: {x['text']}" for x in st.session_state.full_transcript]
        
        # 2. Chunking Transcript (Gom 10 câu làm 1 chunk để query)
        chunk_size = 10
        trans_chunks = ["\n".join(raw_lines[i:i+chunk_size]) for i in range(0, len(raw_lines), chunk_size)]
        
        rag_progress = st.progress(0)
        
        for idx, t_chunk in enumerate(trans_chunks):
            print(f"\n--- 🔄 [CHUNK {idx+1}/{len(trans_chunks)}] XỬ LÝ ĐOẠN HỘI THOẠI ---")
            print(f"📝 Nội dung chunk (rút gọn): {t_chunk[:100].replace(chr(10), ' ')}...")
            
            # 3. Retrieval (Tìm kiếm PDF)
            relevant_pages = []
            if st.session_state.pdf_processed:
                print(f"🔎 [RETRIEVAL] Đang tìm kiếm trong ChromaDB...")
                relevant_pages = pdf_service.find_relevant_pages(t_chunk)
                
                if relevant_pages:
                    print(f"✅ [FOUND] Tìm thấy {len(relevant_pages)} ngữ cảnh liên quan:")
                    for p in relevant_pages:
                        print(f"    - [Trang {p['page']}]: {p['text'][:80]}...")
                else:
                    print("⚠️ [NOT FOUND] Không tìm thấy thông tin khớp trong PDF.")
            else:
                print("⏭️ [SKIP] Không có PDF, bỏ qua bước Retrieval.")

            # 4. Generation (Gọi LLM)
            print(f"🧠 [LLM] Đang gửi prompt tới OpenAI...")
            res = rag_service.generate_minute_with_rag(t_chunk, relevant_pages)
            print(f"✅ [DONE] LLM đã trả về tóm tắt cho chunk này.")
            
            # 5. Ghép kết quả
            full_summary += f"\n#### Phần {idx+1}\n{res['summary']}\n"
            if res['ref_pages']:
                full_summary += f"*(Nguồn tham khảo: Trang {res['ref_pages']})*\n"
            
            rag_progress.progress((idx+1)/len(trans_chunks))
        
        print("\n==================================================")
        print("✅ [RAG FINISH] ĐÃ TẠO XONG BIÊN BẢN")
        print("==================================================\n")
        
        st.session_state.final_minutes = full_summary
        st.success("Đã tạo biên bản xong! Kiểm tra Terminal để xem chi tiết log.")

if "final_minutes" in st.session_state and st.session_state.final_minutes:
    st.markdown("---")
    st.markdown("### 📋 KẾT QUẢ BIÊN BẢN")
    st.markdown(st.session_state.final_minutes)