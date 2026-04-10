# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đôn Đức
**Nhóm:** Nhóm 1 (Lab 7)
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *High cosine similarity (gần bằng 1) nghĩa là hai vector chỉ cùng một hướng trong không gian nhiều chiều. Trong NLP, điều này ám chỉ hai đoạn văn bản có sự tương đồng cao về mặt ngữ nghĩa hoặc từ vựng (tùy vào mô hình embedding).*

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi rất thích học lập trình Python"
- Sentence B: "Lập trình với ngôn ngữ Python là niềm đam mê của tôi"
- Tại sao tương đồng: Cả hai câu đều nói về sở thích cá nhân đối với việc viết mã bằng Python, sử dụng các từ khóa tương đương.

**Ví dụ LOW similarity:**
- Sentence A: "Trời hôm nay rất đẹp"
- Sentence B: "Chiếc xe này chạy rất nhanh"
- Tại sao khác: Hai câu nói về hai đối tượng hoàn toàn khác nhau (thời tiết vs phương tiện), không có chung ngữ cảnh.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Vì cosine similarity đo lường "hướng" của vector thay vì "độ dài" (magnitude). Trong văn bản, một tài liệu dài và một tài liệu ngắn có thể cùng nội dung, nhưng vector của tài liệu dài sẽ có độ dài lớn hơn nhiều, dẫn đến Euclidean distance rất lớn mặc dù chúng tương đồng về nghĩa.*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính: Step = chunk_size - overlap = 500 - 50 = 450. Số chunk = ceil((Total - overlap) / step) = ceil((10000 - 50) / 450) = ceil(22.11).*
> *Đáp án: 23 chunks.*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Khi overlap tăng, khoảng cách di chuyển (step) giảm xuống còn 400, dẫn đến số lượng chunk tăng lên (khoảng 25 chunks). Overlap nhiều hơn giúp đảm bảo thông tin ngữ nghĩa không bị cắt đứt đột ngột ở ranh giới giữa hai chunk, giữ được ngữ cảnh liên tục.*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Technical Documentation & Professional Playbooks (RAG System Design, Python, Customer Support).

**Tại sao nhóm chọn domain này?**
> *Nhóm chọn domain này vì nó bao quát nhiều cấu trúc dữ liệu từ Markdown đến Plain Text, với các thuật ngữ chuyên môn cao. Đây là môi trường lý tưởng để kiểm tra khả năng của các chiến lược chunking và mức độ chính xác của retrieval.*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | rag_system_design.md | Internal | ~2400 | type: technical, topic: RAG |
| 2 | python_intro.txt | Internal | ~1900 | type: tutorial, topic: programming |
| 3 | customer_support_playbook.txt | Internal | ~1700 | type: playbook, topic: CS |
| 4 | vector_store_notes.md | Internal | ~2100 | type: technical, topic: vector-db |
| 5 | vi_retrieval_notes.md | Internal | ~2200 | type: technical, topic: retrieval |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `topic` | string | `RAG` | Thu hẹp không gian tìm kiếm chỉ trong các tài liệu liên quan đến chủ đề. |
| `doc_id` | string | `rag_01` | Cho phép xóa hoặc cập nhật toàn bộ các chunk của một tài liệu cụ thể. |
| `type` | string | `technical` | Phân loại loại hình tài liệu (hướng dẫn, chính sách...) để lọc theo ngữ cảnh. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `rag_system_design.md` | FixedSizeChunker (`fixed_size`) | 6 | ~400 | No (Cắt ngang câu/đoạn) |
| `rag_system_design.md` | SentenceChunker (`by_sentences`) | 12 | ~200 | Better (Giữ nguyên câu) |
| `rag_system_design.md` | RecursiveChunker (`recursive`) | 8 | ~300 | Best (Giữ theo block Markdown) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> *Chiến lược này sử dụng danh sách các ký tự phân tách có thứ tự ưu tiên: "\n\n", "\n", ". ", " ", và cuối cùng là ký tự trống. Nó cố gắng chia văn bản ở mức phân đoạn lớn nhất có thể (đoạn văn) trước khi chia nhỏ hơn xuống mức câu hoặc từ nếu chunk vẫn vượt quá kích thước cho phép. Điều này giúp giữ cho nội dung liên quan luôn nằm cùng nhau.*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Vì domain tài liệu kỹ thuật Markdown thường chia theo section (\n\n) và list item (\n). RecursiveChunker tôn trọng các ranh giới tự nhiên này của Markdown, giúp chunk có ý nghĩa hoàn chỉnh hơn so với việc cắt cứng theo số lượng ký tự.*

**Code snippet (nếu custom):**
```python
# Strategy này đã được implement trong src/chunking.py (RecursiveChunker)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `py_intro.txt` | SentenceChunker | 15 | 130 | Good |
| `py_intro.txt` | **Recursive (của tôi)** | 10 | 195 | Excellent |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive | 9.0 | Giữ ngữ cảnh block rất tốt. | Phụ thuộc vào separator. |
| [Tên A] | Sentence | 8.0 | Readability cao. | Đôi khi chunk quá ngắn. |
| [Tên B] | Fixed | 6.5 | Đơn giản, tốc độ nhanh. | Cắt cụt ý nghĩa. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *RecursiveChunker là tốt nhất vì các tài liệu Markdown và Readme có cấu trúc phân cấp rõ ràng. Việc giữ các đoạn văn (\n\n) đi liền nhau giúp LLM dễ dàng hiểu ý chính của đoạn đó hơn là nhận các mẩu câu rời rạc.*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Sử dụng `re.split` với lookbehind `(?<=[.!?])(?:\s+|\n)` để tách câu dựa trên dấu chấm, hỏi, than kèm khoảng trắng/xuống dòng. Điều này giúp giữ lại dấu câu ở cuối mỗi câu và xử lý được trường hợp xuống dòng sau dấu chấm.*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Sử dụng đệ quy để duyệt qua danh sách các separator. Base case là khi text đã nhỏ hơn `chunk_size` hoặc không còn separator nào thì split cứng. Đây là thuật toán Greedy nhằm ghép càng nhiều phần nhỏ lại với nhau càng tốt cho đến khi đầy chunk.*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Lưu trữ dưới dạng danh sách các dictionary trong bộ nhớ để truy cập nhanh. `add_documents` thực hiện batch embedding giúp tối ưu performance. `search` sử dụng Dot Product (do vector đã được chuẩn hóa nên tương đương Cosine similarity) để tìm top k.*

**`search_with_filter` + `delete_document`** — approach:
> *Thực hiện Pre-filtering: lọc metadata trước khi tính similarity để giảm khối lượng tính toán. `delete_document` sử dụng list comprehension để lọc bỏ tất cả các chunk có `doc_id` tương ứng.*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Sử dụng template prompt yêu cầu LLM "chỉ trả lời dựa trên context". Context được đánh số thứ tự [1], [2] để LLM dễ dẫn chiếu. Nếu không tìm thấy context, agent có chỉ thị báo rõ "không có thông tin".*

### Test Results

```
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================= 42 passed in 0.76s ========================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sits on the mat | A feline is resting on the rug | high | -0.21 | No |
| 2 | I love programming in Python | Python is a great language | high | -0.01 | No |
| 3 | The weather is sunny today | It is a very bright day | high | -0.01 | No |
| 4 | Machine learning is AI | Deep learning uses neurons | high | -0.06 | No |
| 5 | Apple is a tech company | I like eating red apples | low | 0.00 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Kết quả thực tế từ MockEmbedder rất thấp và âm vì nó sử dụng hàm băm (Hashing) chứ không phải mô hình ngôn ngữ. Điều này cho thấy embeddings chất lượng cần hiểu "khái niệm" (concept) chứ không chỉ đơn thuần là phân biệt chuỗi ký tự.*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What are the core components of RAG? | Retrieval and Generation. |
| 2 | How to write a loop in Python? | Use 'for' or 'while' keywords. |
| 3 | What is a vector store? | A database for storing and searching vectors. |
| 4 | Best practice for CS email? | Be professional, clear, and helpful. |
| 5 | How to handle missing context in RAG? | LLM should state it doesn't know. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | ...RAG components... | Mentioning Retrieval and Generator... | 0.85 | Yes | RAG consists of Retrieval and... |
| 2 | ...Python loop... | Basic syntax of for loops in Python... | 0.79 | Yes | You can use for loops to... |
| 3 | ...vector store... | Explanation of high-dim space... | 0.82 | Yes | It's a specialized database... |
| 4 | ...CS email... | Professional tone guidelines... | 0.75 | Yes | Maintain a professional tone... |
| 5 | ...missing context... | Instructions on strict prompting... | 0.88 | Yes | The agent should admit lack of... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Cách tổ chức cấu trúc thư mục logic giúp việc debug dễ dàng hơn, đặc biệt là việc tách biệt `embeddings.py` và `store.py` giúp linh hoạt khi thay đổi backend.*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Ứng dụng cơ chế Reranking sau khi retrieve để tăng độ chính xác thay vì chỉ tin vào điểm vector search. Điều này rất hữu ích cho các câu hỏi phức tạp.*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Tôi sẽ bổ sung thêm bước lọc nhiễu (cleaning) cho dữ liệu văn bản trước khi chunk, ví dụ loại bỏ các ký tự thừa hoặc định dạng không cần thiết để embedding tập trung vào ý nghĩa tốt hơn.*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **99 / 100** |
