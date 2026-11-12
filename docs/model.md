# MODEL.PY – KHỞI TẠO TOKENIZER VÀ MÔ HÌNH PHÂN LOẠI

## 1. Mục đích

File `src/model.py` chịu trách nhiệm khởi tạo Tokenizer và mô hình phân loại cảm xúc dựa trên kiến trúc Transformer được cung cấp bởi thư viện Hugging Face Transformers.

Mô hình được thiết lập để thực hiện bài toán phân loại văn bản thành hai nhóm cảm xúc: NEGATIVE và POSITIVE.

Thay vì tự xây dựng kiến trúc mạng từ đầu, chương trình sử dụng các mô hình đã được huấn luyện trước thông qua `AutoModelForSequenceClassification` và `AutoTokenizer`.

## 2. Các thư viện sử dụng

File sử dụng hai thành phần từ thư viện Transformers:

- `AutoTokenizer`: Tự động tải Tokenizer tương ứng với mô hình được chỉ định.
- `AutoModelForSequenceClassification`: Tải mô hình Transformer dành cho bài toán phân loại chuỗi.

Hai thành phần này sử dụng cùng tên mô hình để bảo đảm Tokenizer và mô hình tương thích với nhau.

## 3. Hàm chính: load_model_and_tokenizer(model_name)

### Mục đích

Hàm này thực hiện việc tải Tokenizer và khởi tạo mô hình phân loại cảm xúc hai lớp dựa trên tên mô hình đầu vào.

### Input

`model_name`: Tên hoặc đường dẫn của mô hình trên Hugging Face Hub hoặc thư mục mô hình được lưu cục bộ.

Tên mô hình được truyền từ cấu hình hoặc thành phần gọi hàm.

### Quy trình thực hiện

**Bước 1: Khởi tạo Tokenizer**

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Chương trình tải Tokenizer tương ứng với mô hình. Tokenizer có nhiệm vụ chuyển văn bản thành các token và mã hóa thành dạng số để mô hình có thể xử lý.

**Bước 2: Khởi tạo mô hình phân loại**

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)
```

Mô hình được tải từ checkpoint pretrained theo `model_name`.

Tham số `num_labels=2` xác định bài toán có hai lớp phân loại.

Hai tham số `id2label` và `label2id` quy định cách ánh xạ giữa mã số và tên nhãn:

| Mã nhãn | Tên nhãn |
|---|---|
| 0 | NEGATIVE |
| 1 | POSITIVE |

Việc khai báo ánh xạ giúp kết quả phân loại được thể hiện thống nhất trong quá trình huấn luyện và dự đoán.

### Bước 3: Trả về kết quả

```python
return tokenizer, model
```

Hàm trả về hai đối tượng:

- `tokenizer`: Dùng để mã hóa văn bản đầu vào.
- `model`: Mô hình Transformer thực hiện phân loại cảm xúc.

## 4. Cách sử dụng trong hệ thống

Các thành phần khác có thể gọi hàm này bằng cách truyền tên mô hình tương ứng.

Ví dụ:

```python
tokenizer, model = load_model_and_tokenizer(model_name)
```

Sau khi khởi tạo:

- Tokenizer thực hiện mã hóa văn bản.
- Model tiếp nhận các tensor đầu vào và tạo logits cho hai lớp cảm xúc.
- Các module huấn luyện hoặc dự đoán sử dụng model và tokenizer để thực hiện nhiệm vụ tương ứng.

Tên mô hình cụ thể được quyết định bởi cấu hình của từng bài thực hành.

## 5. Vai trò trong Lab3

File `model.py` đóng vai trò kết nối giữa mô hình pretrained và các bước xử lý tiếp theo của hệ thống.

Trong Exercise 1, mô hình phân loại cảm xúc pretrained được sử dụng để dự đoán trực tiếp.

Trong Exercise 2, DistilBERT được khởi tạo cho bài toán hai lớp và tiếp tục được fine-tune trên dữ liệu IMDb.

Hai bài thực hành có mục tiêu sử dụng mô hình khác nhau, nhưng đều dùng cơ chế ánh xạ hai nhãn NEGATIVE và POSITIVE.

## 6. Kiểm chứng

Có thể kiểm tra hàm khởi tạo bằng cách:

- Kiểm tra Tokenizer được tải thành công.
- Kiểm tra mô hình có cấu hình `num_labels = 2`.
- Kiểm tra `id2label` và `label2id` ánh xạ đúng hai lớp.
- Đưa một văn bản đã mã hóa vào mô hình và kiểm tra đầu ra có hai logits.

Lưu ý: File `model.py` chỉ thực hiện tải và khởi tạo Tokenizer, mô hình. Quá trình huấn luyện, tối ưu trọng số và đánh giá được xử lý ở các module khác.

## 7. Kết luận

File `model.py` giúp khởi tạo thống nhất Tokenizer và mô hình Transformer cho bài toán phân loại cảm xúc hai lớp.

Việc sử dụng các thành phần tự động từ Hugging Face Transformers giúp giảm phần code phải xây dựng thủ công, đồng thời tạo điều kiện để sử dụng các mô hình pretrained trong cả quá trình dự đoán và fine-tuning.
