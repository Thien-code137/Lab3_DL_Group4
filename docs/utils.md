# Utils Module – `src/utils.py`

## 1. Mục đích

Module `utils.py` cung cấp các hàm tiện ích phục vụ quá trình huấn luyện và quản lý kết quả trong dự án Lab3.

Module đảm nhiệm hai nhiệm vụ chính:

- Kiểm tra tính hợp lệ của file cấu hình trước khi huấn luyện.
- Đọc cấu hình JSON và lưu kết quả dưới định dạng JSON.

Việc tách các chức năng này thành module riêng giúp các thành phần khác trong dự án có thể sử dụng chung, hạn chế lặp lại code.

## 2. Thư viện sử dụng

| Thư viện | Chức năng |
|---|---|
| `json` | Đọc và ghi dữ liệu JSON |
| `math` | Kiểm tra tính hữu hạn của giá trị số |
| `pathlib.Path` | Xử lý đường dẫn và thao tác với file |

## 3. Các chức năng chính

### 3.1. Hàm `validate_config(cfg)`

Hàm kiểm tra cấu hình trước khi bắt đầu quá trình huấn luyện.

**Tham số đầu vào:**

`cfg`: Dictionary chứa các tham số cấu hình được đọc từ file JSON.

**Bước 1: Kiểm tra các trường bắt buộc**

```python
required = (
    "seed", "dataset", "model", "train_size",
    "validation_size", "test_size", "max_length",
    "epochs", "batch_size", "learning_rate",
    "weight_decay", "selection_metric"
)
```

Hàm kiểm tra xem tất cả các trường bắt buộc có tồn tại trong cấu hình hay không.

Nếu thiếu trường, chương trình phát sinh `ValueError` và hiển thị danh sách các trường còn thiếu.

**Bước 2: Kiểm tra các giá trị số nguyên**

Các trường sau phải là số nguyên dương:

- `train_size`: Số lượng mẫu huấn luyện.
- `validation_size`: Số lượng mẫu validation.
- `test_size`: Số lượng mẫu test.
- `max_length`: Độ dài token tối đa.
- `epochs`: Số vòng huấn luyện.
- `batch_size`: Kích thước batch.

Nếu giá trị không phải số nguyên hoặc nhỏ hơn hoặc bằng 0, chương trình phát sinh lỗi.

**Bước 3: Kiểm tra kích thước tập dữ liệu**

```python
if cfg["train_size"] % 2 or cfg["validation_size"] % 2:
    raise ValueError("Balanced train/validation sizes must be even")
```

Do dự án sử dụng tập train và validation cân bằng giữa hai lớp cảm xúc, kích thước của hai tập này phải là số chẵn.

Điều kiện này bảo đảm có thể chia đều số lượng mẫu cho hai nhãn NEGATIVE và POSITIVE.

**Bước 4: Kiểm tra Learning Rate**

```python
if not (0 < cfg["learning_rate"] < 1):
    raise ValueError("Invalid learning rate")
```

Learning rate phải lớn hơn 0 và nhỏ hơn 1.

Giá trị này kiểm soát mức độ cập nhật trọng số mô hình trong quá trình tối ưu hóa.

**Bước 5: Kiểm tra Weight Decay**

```python
if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
    raise ValueError("Invalid weight decay")
```

Weight decay phải là một giá trị hữu hạn và không âm.

Hàm `math.isfinite()` giúp phát hiện các giá trị không hợp lệ như NaN hoặc Infinity.

**Bước 6: Kiểm tra chỉ số lựa chọn checkpoint**

```python
if cfg["selection_metric"] != "eval_loss":
    raise ValueError("Use eval_loss for checkpoint selection")
```

Dự án yêu cầu sử dụng `eval_loss` làm chỉ số lựa chọn checkpoint.

Điều này có nghĩa mô hình sẽ ưu tiên checkpoint có validation loss thấp hơn.

Nếu cấu hình sử dụng chỉ số khác, chương trình sẽ phát sinh lỗi.

**Giá trị trả về:**

Nếu toàn bộ điều kiện hợp lệ, hàm trả về chính dictionary `cfg` ban đầu.

### 3.2. Hàm `read_config(path)`

Hàm đọc file cấu hình JSON và kiểm tra tính hợp lệ trước khi trả về kết quả.

```python
def read_config(path):
    return validate_config(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
```

Quá trình thực hiện:

1. Chuyển đường dẫn đầu vào thành đối tượng `Path`.
2. Đọc nội dung file bằng mã hóa UTF-8.
3. Sử dụng `json.loads()` chuyển nội dung JSON thành dictionary.
4. Gọi `validate_config()` để kiểm tra cấu hình.
5. Trả về dictionary nếu hợp lệ.

**Tham số đầu vào:**

`path`: Đường dẫn đến file cấu hình JSON, ví dụ `configs/default.json`.

**Giá trị trả về:**

Dictionary chứa các tham số cấu hình đã được kiểm tra.

### 3.3. Hàm `save_json(path, data)`

Hàm lưu dữ liệu vào file JSON.

```python
def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
```

Quá trình thực hiện:

1. Chuyển đường dẫn thành đối tượng `Path`.
2. Tạo thư mục cha nếu chưa tồn tại.
3. Chuyển dữ liệu Python thành chuỗi JSON.
4. Ghi dữ liệu vào file với mã hóa UTF-8.

Trong đó:

- `ensure_ascii=False`: Giữ nguyên các ký tự Unicode, bao gồm tiếng Việt.
- `indent=2`: Định dạng JSON với mức thụt lề 2 khoảng trắng.
- `exist_ok=True`: Không phát sinh lỗi nếu thư mục đã tồn tại.

Hàm không trả về giá trị cụ thể.

## 4. Ví dụ sử dụng

### 4.1. Đọc cấu hình

```python
from src.utils import read_config

config = read_config("configs/default.json")
print(config)
```

Chương trình đọc và kiểm tra cấu hình trước khi sử dụng.

### 4.2. Lưu dữ liệu JSON

```python
from src.utils import save_json

results = {
    "accuracy": 0.89,
    "f1_positive": 0.9018
}

save_json("outputs/results.json", results)
```

Sau khi thực hiện, dữ liệu được lưu vào file `outputs/results.json`.

## 5. Vai trò trong hệ thống

Module `utils.py` được sử dụng bởi các module khác trong dự án:

| Module | Cách sử dụng |
|---|---|
| `train.py` | Sử dụng `read_config()` để đọc và kiểm tra cấu hình; sử dụng `save_json()` để lưu thống kê dữ liệu và thông tin lần chạy |
| `evaluate.py` | Sử dụng `save_json()` để lưu các chỉ số đánh giá vào file JSON |

Nhờ đó, việc đọc cấu hình và lưu kết quả được thực hiện thống nhất trên toàn bộ dự án.

## 6. Kết luận

`utils.py` là module hỗ trợ quản lý cấu hình và dữ liệu đầu ra của Lab3.

Mặc dù có dung lượng code nhỏ, module đóng vai trò quan trọng trong việc kiểm tra cấu hình trước khi huấn luyện, hạn chế lỗi do tham số không hợp lệ và chuẩn hóa cách lưu kết quả của hệ thống.
