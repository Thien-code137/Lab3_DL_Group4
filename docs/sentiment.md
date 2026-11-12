# Sentiment Module – `src/sentiment.py`

## 1. Mục đích

Module `sentiment.py` thực hiện phân tích cảm xúc (Sentiment Analysis) bằng mô hình đã được huấn luyện sẵn từ Hugging Face.

Chương trình nhận một đoạn văn bản, sử dụng tokenizer để chuyển văn bản thành dữ liệu đầu vào, sau đó đưa dữ liệu qua mô hình để dự đoán cảm xúc.

Module này thuộc Exercise 1 và chỉ thực hiện inference, không huấn luyện hoặc fine-tune mô hình.

## 2. Mô hình sử dụng

Module sử dụng checkpoint mặc định:

```python
DEFAULT_CHECKPOINT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
```

Đây là mô hình DistilBERT đã được fine-tune trên bộ dữ liệu SST-2 để phân loại cảm xúc thành hai lớp:

- `NEGATIVE`: Cảm xúc tiêu cực.
- `POSITIVE`: Cảm xúc tích cực.

Người dùng cũng có thể truyền một checkpoint khác vào hàm `infer()`.

## 3. Thư viện sử dụng

- `argparse`: Xử lý tham số đầu vào từ dòng lệnh.
- `torch`: Thực hiện suy luận và tính xác suất dự đoán.
- `AutoTokenizer`: Chuyển văn bản thành các token và tensor đầu vào.
- `AutoModelForSequenceClassification`: Tải mô hình phân loại cảm xúc đã được huấn luyện sẵn.

## 4. Chức năng chính

### 4.1. Hàm `infer(text, checkpoint=DEFAULT_CHECKPOINT)`

Đây là hàm chính thực hiện quá trình phân tích cảm xúc.

**Tham số đầu vào:**

| Tham số | Ý nghĩa |
|---|---|
| `text` | Đoạn văn bản cần phân tích cảm xúc |
| `checkpoint` | Tên hoặc đường dẫn checkpoint mô hình, mặc định là mô hình DistilBERT đã fine-tune trên SST-2 |

**Quy trình xử lý:**

**Bước 1: Tải tokenizer và mô hình**

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```

Tokenizer và mô hình được tải từ checkpoint được chỉ định.

**Bước 2: Chuyển mô hình sang chế độ đánh giá**

```python
model.eval()
```

Đưa mô hình sang chế độ đánh giá để thực hiện inference.

**Bước 3: Tokenize văn bản**

```python
inputs = tokenizer(
    text,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

Văn bản được chuyển thành các tensor đầu vào gồm:

- `input_ids`: ID đại diện cho các token.
- `attention_mask`: Đánh dấu những vị trí token cần được mô hình chú ý.

Tham số `max_length=512` giới hạn độ dài đầu vào tối đa là 512 token. `truncation=True` cho phép cắt bớt phần vượt quá giới hạn này.

**Bước 4: Thực hiện inference**

```python
with torch.no_grad():
    logits = model(**inputs).logits
```

Mô hình nhận các tensor đầu vào và trả về logits, tức điểm số dự đoán cho từng lớp cảm xúc.

`torch.no_grad()` tắt việc tính gradient vì quá trình này chỉ dự đoán, không cập nhật trọng số mô hình.

**Bước 5: Chuyển logits thành xác suất**

```python
probabilities = torch.softmax(logits, dim=-1)[0]
```

Hàm Softmax chuyển logits thành xác suất của từng lớp. Các xác suất này được sử dụng để xác định cảm xúc có khả năng cao nhất.

**Bước 6: Xác định nhãn dự đoán**

```python
index = int(probabilities.argmax())
```

`argmax()` tìm vị trí của lớp có xác suất cao nhất.

Nhãn dự đoán được lấy từ cấu hình mô hình:

```python
model.config.id2label[index]
```

Cách này sử dụng ánh xạ nhãn được định nghĩa trong checkpoint thay vì tự gán nhãn cố định theo index.

**Bước 7: Trả về kết quả**

Hàm `infer()` trả về một dictionary gồm:

| Trường | Ý nghĩa |
|---|---|
| `text` | Văn bản đầu vào |
| `tokens` | Danh sách token sau khi tokenize |
| `input_ids` | Danh sách ID tương ứng với các token |
| `attention_mask` | Danh sách attention mask |
| `label` | Nhãn cảm xúc được dự đoán |
| `score` | Xác suất của nhãn được dự đoán |

## 5. Chạy chương trình

Chạy bằng lệnh:

```bash
python src/sentiment.py
```

Khi không truyền tham số `--text`, chương trình sử dụng văn bản mặc định:

```text
I love this movie. It is amazing!
```

Có thể truyền văn bản khác:

```bash
python src/sentiment.py --text "This movie is terrible."
```

Chương trình sẽ in dictionary chứa văn bản, thông tin tokenization, nhãn cảm xúc và score.

Lưu ý: Máy cần có kết nối Internet khi tải checkpoint từ Hugging Face lần đầu, trừ trường hợp checkpoint đã có sẵn trong bộ nhớ cache hoặc được cung cấp từ đường dẫn local.

## 6. Vai trò trong Exercise 1

Module này giúp minh họa cách sử dụng một mô hình Transformer đã được fine-tune để giải quyết bài toán phân loại cảm xúc.

Luồng xử lý:

**Văn bản đầu vào → Tokenization → Input IDs và Attention Mask → DistilBERT → Logits → Softmax → Nhãn cảm xúc và Score**

Ngoài nhãn dự đoán, module còn trả về tokens, input IDs và attention mask để hỗ trợ quan sát, tìm hiểu quá trình tiền xử lý văn bản trước khi đưa vào mô hình.

## 7. Đặc điểm và giới hạn

- Sử dụng mô hình pretrained đã fine-tune trên SST-2.
- Không thực hiện huấn luyện hoặc fine-tuning.
- Giới hạn đầu vào tối đa 512 token.
- Kết quả score là xác suất Softmax của lớp được dự đoán, không phải thước đo bảo đảm độ chính xác tuyệt đối.
- Tokenizer và mô hình được tải trong mỗi lần gọi `infer()`, không có cơ chế lưu đối tượng để tái sử dụng trong module này.
