# Visualize Module – `src/visualize.py`

## 1. Mục đích

Module `visualize.py` được sử dụng để lưu lại lịch sử huấn luyện và trực quan hóa sự thay đổi của loss trong quá trình fine-tuning mô hình.

Module thực hiện hai nhiệm vụ chính:

- Chuyển lịch sử log của Hugging Face Trainer thành danh sách dữ liệu có cấu trúc.
- Xuất lịch sử huấn luyện ra file CSV và tạo biểu đồ so sánh Training Loss với Validation Loss.

## 2. Thư viện sử dụng

| Thư viện | Chức năng |
|---|---|
| `csv` | Ghi lịch sử huấn luyện ra file CSV |
| `pathlib.Path` | Xử lý đường dẫn và thao tác với file |
| `matplotlib.pyplot` | Vẽ và lưu biểu đồ loss |

`matplotlib.pyplot` được import bên trong hàm `save_training_history()` để chỉ sử dụng khi cần tạo biểu đồ.

## 3. Các chức năng chính

### 3.1. Hàm `rows_from_log_history(log_history)`

Hàm chuyển lịch sử log do Hugging Face Trainer cung cấp thành danh sách dictionary có cấu trúc thống nhất.

**Tham số đầu vào:**

`log_history`: Danh sách các dictionary chứa thông tin được ghi nhận trong quá trình huấn luyện và đánh giá.

**Cách xử lý:**

```python
def rows_from_log_history(log_history):
    return [
        {
            "epoch": item.get("epoch", ""),
            "step": item.get("step", ""),
            "train_loss": item.get("loss", ""),
            "val_loss": item.get("eval_loss", ""),
            "val_accuracy": item.get("eval_accuracy", ""),
            "val_f1": item.get("eval_f1_positive", "")
        }
        for item in log_history
    ]
```

Hàm lấy các trường thông tin từ từng phần tử của lịch sử log:

| Trường đầu ra | Trường trong log gốc | Ý nghĩa |
|---|---|---|
| `epoch` | `epoch` | Epoch tại thời điểm ghi nhận |
| `step` | `step` | Bước huấn luyện |
| `train_loss` | `loss` | Training loss |
| `val_loss` | `eval_loss` | Validation loss |
| `val_accuracy` | `eval_accuracy` | Độ chính xác trên tập validation |
| `val_f1` | `eval_f1_positive` | F1-score của lớp Positive trên tập validation |

Hàm sử dụng `item.get()` để lấy dữ liệu. Nếu một trường không tồn tại trong log, giá trị mặc định là chuỗi rỗng.

Điều này giúp xử lý các bản ghi không chứa đầy đủ tất cả các chỉ số.

**Giá trị trả về:**

Một danh sách dictionary chứa thông tin lịch sử huấn luyện đã được chuẩn hóa.

### 3.2. Hàm `save_training_history(log_history, output_dir)`

Hàm lưu lịch sử huấn luyện thành file CSV và tạo biểu đồ loss.

**Tham số đầu vào:**

| Tham số | Ý nghĩa |
|---|---|
| `log_history` | Lịch sử log từ Hugging Face Trainer |
| `output_dir` | Thư mục lưu file CSV và biểu đồ |

**Bước 1: Chuẩn bị dữ liệu**

```python
output = Path(output_dir)
rows = rows_from_log_history(log_history)
```

Chuyển đường dẫn đầu ra thành đối tượng `Path` và gọi `rows_from_log_history()` để trích xuất dữ liệu.

**Bước 2: Xuất lịch sử ra CSV**

```python
with (output / "history.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
```

Chương trình tạo file `history.csv` trong thư mục đầu ra.

Các cột được ghi gồm:

```text
epoch, step, train_loss, val_loss, val_accuracy, val_f1
```

Sử dụng `csv.DictWriter` để ghi dữ liệu theo từng dictionary.

File CSV giúp lưu trữ và kiểm tra lại các chỉ số qua từng lần ghi log.

**Bước 3: Khởi tạo biểu đồ**

```python
fig, ax = plt.subplots(figsize=(7, 4))
```

Tạo biểu đồ với kích thước 7 × 4 inch.

**Bước 4: Trích xuất dữ liệu loss**

```python
for name, legend in (
    ("train_loss", "Training loss"),
    ("val_loss", "Validation loss")
):
    points = [
        (float(row["epoch"]), float(row[name]))
        for row in rows
        if row["epoch"] != "" and row[name] != ""
    ]
```

Chương trình lần lượt xử lý hai loại loss:

- `train_loss`: Loss trên tập huấn luyện.
- `val_loss`: Loss trên tập validation.

Chỉ những bản ghi có đầy đủ `epoch` và giá trị loss mới được sử dụng để vẽ.

Các giá trị được chuyển sang kiểu số thực để phục vụ việc biểu diễn trên biểu đồ.

**Bước 5: Vẽ đường biểu diễn**

```python
if points:
    x, y = zip(*points)
    ax.plot(x, y, marker="o", label=legend)
```

Trục X biểu diễn epoch, trục Y biểu diễn giá trị loss.

Mỗi loại loss được thể hiện bằng một đường riêng, có các điểm đánh dấu để quan sát sự thay đổi qua từng epoch.

**Bước 6: Hoàn thiện biểu đồ**

```python
ax.set(
    xlabel="Epoch",
    ylabel="Loss",
    title="Loss during fine-tuning"
)
ax.grid(alpha=0.3)
```

Biểu đồ được bổ sung:

- Tên trục X: Epoch.
- Tên trục Y: Loss.
- Tiêu đề: Loss during fine-tuning.
- Đường lưới để dễ quan sát giá trị.

Nếu có ít nhất một đường biểu diễn, chương trình hiển thị chú thích bằng `ax.legend()`.

**Bước 7: Lưu biểu đồ**

```python
fig.tight_layout()
fig.savefig(output / "loss_curves.png", dpi=180)
plt.close(fig)
```

Biểu đồ được lưu dưới tên `loss_curves.png` với độ phân giải 180 DPI.

Sau khi lưu, `plt.close(fig)` đóng figure để giải phóng tài nguyên.

## 4. Các file đầu ra

Sau khi gọi `save_training_history()`, module tạo hai file:

| File | Nội dung |
|---|---|
| `history.csv` | Lịch sử huấn luyện gồm epoch, step, train loss, validation loss, accuracy và F1-score |
| `loss_curves.png` | Biểu đồ thể hiện Training Loss và Validation Loss theo epoch |

Các file được lưu trong thư mục `output_dir` được truyền vào hàm.

Lưu ý: Thư mục đầu ra phải tồn tại trước khi gọi hàm. Trong quy trình hiện tại, `train.py` tạo thư mục này trước khi bắt đầu huấn luyện.

## 5. Cách sử dụng trong dự án

Module được gọi từ `src/train.py` sau khi quá trình huấn luyện hoàn tất:

```python
save_training_history(
    trainer.state.log_history,
    output
)
```

Trong đó:

- `trainer.state.log_history` cung cấp lịch sử log từ Trainer.
- `output` là thư mục lưu kết quả của lần chạy.

Nhờ đó, thông tin huấn luyện được lưu lại để phục vụ việc theo dõi và phân tích.

## 6. Ý nghĩa của biểu đồ Loss

Biểu đồ Training Loss và Validation Loss giúp quan sát quá trình học của mô hình.

**Training Loss:** Thể hiện mức độ sai lệch của mô hình trên dữ liệu huấn luyện.

**Validation Loss:** Thể hiện mức độ sai lệch trên dữ liệu validation, được sử dụng để theo dõi khả năng tổng quát hóa và lựa chọn checkpoint.

Một số cách diễn giải thường gặp:

- Cả hai loss giảm: Mô hình đang cải thiện trên cả tập huấn luyện và validation.
- Training loss tiếp tục giảm nhưng validation loss tăng: Có thể là dấu hiệu overfitting.
- Cả hai loss đều cao hoặc ít thay đổi: Mô hình có thể chưa học hiệu quả.

Cần kết hợp biểu đồ với các chỉ số đánh giá khác để đưa ra kết luận về chất lượng mô hình.

## 7. Vai trò trong hệ thống

`visualize.py` hỗ trợ phân tích quá trình fine-tuning và được sử dụng bởi `train.py`.

Luồng xử lý:

**Hugging Face Trainer → Log History → rows_from_log_history() → history.csv và loss_curves.png**

Module giúp lưu lại quá trình huấn luyện dưới dạng dữ liệu có thể phân tích và biểu đồ trực quan, từ đó hỗ trợ đánh giá quá trình học của DistilBERT.
