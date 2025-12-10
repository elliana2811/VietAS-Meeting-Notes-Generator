def generate_final_minutes(full_transcript: str, collection_name: str, pdf_was_provided: bool) -> dict:
    """Tạo biên bản cuối cùng."""
    context = ""
    if pdf_was_provided:
        # 1. Phân tích transcript, tạo query
        # 2. Query ChromaDB từ collection_name để lấy context
        context = "Đây là ngữ cảnh lấy từ PDF..."
    
    # 3. Xây dựng prompt tăng cường (ghép context và transcript)
    # 4. Gọi API LLM
    # 5. Parse kết quả JSON và trả về
    
    # Giả lập kết quả
    return {
        "title": "Biên bản họp Demo",
        "summary": "Đã thảo luận nhiều vấn đề quan trọng.",
        "action_items": [{"task": "Làm slide", "assignee": "User A", "deadline": "EOD"}]
    }