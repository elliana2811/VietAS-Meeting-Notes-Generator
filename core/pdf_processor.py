import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
import os
from openai import OpenAI

class PDFKnowledgeBase:
    def __init__(self, api_key, collection_name, persist_directory="./storage/vector_store"):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
        # 1. Khởi tạo ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # 2. Sử dụng OpenAI Embedding Function cho Chroma
        # Nó sẽ tự động gọi API 'text-embedding-3-small' khi thêm/tìm dữ liệu
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        # 3. Tạo hoặc lấy Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def process_and_store_pdf(self, pdf_path):
        """
        Đọc từng trang PDF và lưu vào Vector DB.
        """
        print(f"📄 Đang xử lý PDF: {pdf_path}")
        
        # Dùng pdfplumber để trích xuất text tốt hơn
        with pdfplumber.open(pdf_path) as pdf:
            documents = []
            metadatas = []
            ids = []
            
            for i, page in enumerate(pdf.pages):
                # 1. Trích xuất text từ trang
                text = page.extract_text()
                if not text: 
                    text = "" # Xử lý trang chỉ có ảnh (nếu cần OCR thì phải dùng thư viện khác)
                
                # Làm sạch text cơ bản
                text = text.strip()
                
                # 2. (Optional) Nếu muốn phân tích cả ẢNH:
                # Ở bước này bạn có thể convert page -> image, gửi lên GPT-4o Vision 
                # để lấy mô tả ảnh, rồi cộng vào biến `text`.
                # Tuy nhiên, để tiết kiệm, ta dùng text trích xuất là đủ cho MVP.
                
                if len(text) > 10: # Chỉ lưu trang có nội dung đáng kể
                    page_num = i + 1
                    
                    # Chuẩn bị dữ liệu cho Chroma
                    documents.append(text)
                    
                    # Metadata cực kỳ quan trọng để map ngược lại
                    metadatas.append({
                        "source": os.path.basename(pdf_path),
                        "page_number": page_num
                    })
                    
                    # ID duy nhất
                    ids.append(f"{os.path.basename(pdf_path)}_page_{page_num}")
            
            # 3. Lưu Batch vào ChromaDB
            if documents:
                print(f"💾 Đang lưu {len(documents)} trang vào Vector DB...")
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print("✅ Đã lưu xong PDF.")
            else:
                print("⚠️ PDF không có text trích xuất được.")

    def find_relevant_pages(self, transcript_chunk, n_results=2):
        """
        Input: Một đoạn transcript (lời nói)
        Output: Nội dung các trang PDF liên quan nhất
        """
        results = self.collection.query(
            query_texts=[transcript_chunk],
            n_results=n_results
        )
        
        # Format kết quả trả về cho dễ dùng
        relevant_context = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                relevant_context.append({
                    "text": doc,
                    "page": meta['page_number'],
                    "source": meta['source']
                })
        
        return relevant_context