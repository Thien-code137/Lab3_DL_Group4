# Predict Module – `src/predict.py`

## 1. Mục đích

Module `predict.py` thực hiện suy luận (inference) bằng mô hình phân loại cảm xúc đã được fine-tune và lưu thành checkpoint.

Người dùng truyền vào đường dẫn checkpoint và một đoạn văn bản. Chương trình sử dụng tokenizer để chuyển văn bản thành đầu vào của mô hình, sau đó dự đoán nhãn cảm xúc và in ra kết quả cùng độ tin cậy.

Module này chỉ thực hiện dự đoán, không huấn luyện hoặc fine-tune mô hình.

## 2. Thư viện sử dụng

- `argparse`: Đọc các tham số được truyền từ dòng lệnh.
- `torch`: Thực hiện suy luận và tính xác suất bằng PyTorch.
- `AutoTokenizer`: Tải tokenizer tương ứng với checkpoint.
- `AutoModelForSequenceClassification`: Tải mô hình phân loại văn bản đã được huấn luyện.

## 3. Chức năng chính

### 3.1. Hàm `main()`

Đây là hàm chính điều khiển toàn bộ quá trình dự đoán.

**Bước 1: Nhận tham số từ dòng lệnh**

Chương trình yêu cầu hai tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--checkpoint` | Đường dẫn đến checkpoint mô hình đã fine-tune và lưu trên máy |
| `--text` | Văn bản đầu vào cần dự đoán cảm xúc |

Cả hai tham số đều bắt buộc phải được cung cấp.

**Bước 2: Tải tokenizer và mô hình**

```python
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
```

Tokenizer và mô hình được tải từ cùng một checkpoint để bảo đảm cách tiền xử lý văn bản tương thích với mô hình đã huấn luyện.

**Bước 3: Chuyển mô hình sang chế độ đánh giá**

```python
model.eval()
```

Lệnh này đưa mô hình sang chế độ đánh giá, phù hợp với quá trình suy luận thay vì huấn luyện.

**Bước 4: Thực hiện dự đoán**

```python
with torch.no_grad():
    probabilities = torch.softmax(
        model(**tokenizer(
            args.text,
            return_tensors="pt",
            truncation=True
        ))["logits"],
        dim=-1
    )[0]
```

Quá trình gồm các thao tác:

1. Tokenizer chuyển văn bản thành các tensor đầu vào.
2. `return_tensors="pt"` yêu cầu kết quả ở định dạng tensor PyTorch.
3. `truncation=True` cho phép cắt ngắn văn bản nếu vượt quá giới hạn đầu vào của tokenizer.
4. Mô hình nhận dữ liệu và trả về `logits`.
5. `torch.softmax()` chuyển logits thành xác suất của hai lớp.
6. `torch.no_grad()` tắt việc tính gradient vì chương trình chỉ dự đoán, không cập nhật trọng số mô hình.

**Bước 5: Xác định nhãn cảm xúc**

```python
index = int(probabilities.argmax())
```

`argmax()` lấy vị trí có xác suất cao nhất.

Quy ước nhãn được sử dụng:

| Index | Nhãn |
|---|---|
| `0` | NEGATIVE |
| `1` | POSITIVE |

**Bước 6: In kết quả**

```python
print({
    "label": "POSITIVE" if index else "NEGATIVE",
    "confidence": float(probabilities[index])
})
```

Kết quả gồm hai thông tin:

- `label`: Nhãn cảm xúc được dự đoán.
- `confidence`: Xác suất của nhãn được dự đoán, được chuyển sang kiểu số thực Python.

## 4. Cách chạy chương trình

Có thể chạy trực tiếp từ thư mục gốc của repository:

```bash
python src/predict.py --checkpoint "outputs/exercise2/checkpoint-125" --text "This movie is amazing."
```

Trong đó:

- `--checkpoint`: Đường dẫn đến thư mục checkpoint đã được xuất và lưu trên máy.
- `--text`: Nội dung văn bản muốn phân loại.

Đường dẫn checkpoint trong ví dụ cần được thay bằng đường dẫn thực tế trong repository hoặc trên máy đang chạy.

## 5. Ví dụ kết quả

Ví dụ minh họa định dạng kết quả:

```python
{
    'label': 'POSITIVE',
    'confidence': 0.98
}
```

Kết quả trên cho biết mô hình dự đoán văn bản thuộc lớp `POSITIVE`, với xác suất dự đoán của lớp này là `0.98`.

Đây là ví dụ minh họa, không phải kết quả được xác nhận từ một lần chạy thực tế.

## 6. Vai trò trong hệ thống

Module `predict.py` sử dụng checkpoint đã được huấn luyện và lưu lại để phân loại các văn bản mới.

Luồng xử lý:

**Văn bản đầu vào → Tokenizer → Mô hình đã fine-tune → Softmax → Nhãn cảm xúc và confidence**

Module này bổ sung chức năng sử dụng mô hình sau huấn luyện, giúp người dùng kiểm tra dự đoán trên những văn bản không nằm trong tập dữ liệu huấn luyện.
