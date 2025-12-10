from openai import OpenAI

class MeetingMinuteGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_minute_with_rag(self, transcript_segment, pdf_context_list):
        """
        transcript_segment: "Anh Nam nói về doanh thu quý 3..."
        pdf_context_list: List các trang PDF tìm được từ hàm find_relevant_pages
        """
        
        # 1. Xây dựng Context từ PDF
        context_str = ""
        used_pages = []
        if pdf_context_list:
            context_str += "TÀI LIỆU THAM KHẢO (PDF):\n"
            for item in pdf_context_list:
                context_str += f"- [Trang {item['page']}]: {item['text'][:800]}...\n" # Cắt ngắn nếu dài
                used_pages.append(item['page'])
        else:
            context_str = "Không tìm thấy tài liệu tham khảo liên quan."

        # 2. Tạo Prompt
        system_prompt = """
        Bạn là thư ký cuộc họp chuyên nghiệp. 
        Nhiệm vụ: Dựa vào 'Hội thoại' và 'Tài liệu tham khảo', hãy viết tóm tắt ngắn gọn.
        Yêu cầu:
        - Chỉ trích xuất thông tin quan trọng, quyết định, hoặc con số.
        - Nếu hội thoại nhắc đến nội dung trong tài liệu, hãy bổ sung chi tiết từ tài liệu để làm rõ.
        - Ghi chú rõ thông tin lấy từ trang nào của tài liệu (nếu có).
        """
        
        user_message = f"""
        {context_str}
        
        ----------------
        HỘI THOẠI (TRANSCRIPT):
        {transcript_segment}
        ----------------
        
        Hãy viết biên bản cho đoạn hội thoại trên:
        """

        # 3. Gọi GPT-4o-mini (Rẻ và nhanh) hoặc GPT-4o (Thông minh hơn)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        
        return {
            "summary": response.choices[0].message.content,
            "ref_pages": list(set(used_pages)) # Trả về danh sách trang đã tham khảo
        }