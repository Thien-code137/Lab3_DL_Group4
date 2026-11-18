# Train Module – `src/train.py`

## 1. Mục đích

Module `train.py` điều phối quá trình huấn luyện (fine-tuning) mô hình DistilBERT cho bài toán phân loại cảm xúc trên bộ dữ liệu IMDb.

Module thực hiện các công việc chính:

- Đọc cấu hình huấn luyện.
- Chuẩn bị dữ liệu train, validation và test.
- Tải tokenizer và mô hình.
- Tokenize dữ liệu văn bản.
- Fine-tune mô hình bằng Hugging Face Trainer.
- Theo dõi quá trình huấn luyện và lựa chọn checkpoint.
- Đánh giá mô hình trên tập test.
- Lưu mô hình tốt nhất, các chỉ số đánh giá và thông tin của lần chạy.

## 2. Thư viện và module sử dụng

### 2.1. Thư viện Python và bên ngoài

| Thư viện | Chức năng |
|---|---|
| `argparse` | Nhận tham số từ dòng lệnh |
| `json` | Xuất kết quả dưới định dạng JSON |
| `platform` | Lấy phiên bản Python |
| `Counter` | Thống kê số lượng mẫu theo nhãn |
| `datetime` | Ghi nhận thời gian bắt đầu và kết thúc huấn luyện |
| `Path` | Quản lý đường dẫn và thư mục |
| `torch` | Nền tảng tính toán PyTorch |
| `transformers` | Cung cấp các công cụ huấn luyện mô hình Transformer |
| `Trainer` | Quản lý vòng lặp huấn luyện và đánh giá |
| `TrainingArguments` | Cấu hình các tham số huấn luyện |
| `DataCollatorWithPadding` | Padding dữ liệu theo batch |
| `set_seed` | Thiết lập seed để tăng khả năng tái lập kết quả |

### 2.2. Các module nội bộ

| Module | Chức năng |
|---|---|
| `src.data` | Chuẩn bị tập train, validation và test |
| `src.evaluate` | Tính toán và lưu các chỉ số đánh giá |
| `src.model` | Khởi tạo tokenizer và mô hình |
| `src.utils` | Đọc cấu hình và lưu dữ liệu JSON |
| `src.visualize` | Lưu lịch sử huấn luyện và biểu đồ |

## 3. Quy trình huấn luyện

### 3.1. Hàm `train(config_path, output_dir)`

Đây là hàm chính điều khiển toàn bộ quá trình fine-tuning.

Hai tham số mặc định:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `config_path` | `configs/default.json` | Đường dẫn đến file cấu hình |
| `output_dir` | `outputs/new_run` | Thư mục lưu kết quả của lần chạy |

**Bước 1: Đọc cấu hình và tạo thư mục đầu ra**

```python
config = read_config(config_path)
output = Path(output_dir)
output.mkdir(parents=True, exist_ok=True)
set_seed(config["seed"])
```

Chương trình đọc các tham số từ file cấu hình, tạo thư mục đầu ra nếu chưa tồn tại và thiết lập seed.

Seed giúp tăng khả năng tái lập kết quả khi thực hiện lại quá trình huấn luyện với cùng cấu hình và môi trường.

**Bước 2: Chuẩn bị dữ liệu**

```python
train_set, val_set, test_set = prepare_splits(config)
```

Hàm `prepare_splits()` từ `src/data.py` chia dữ liệu thành ba tập:

| Tập dữ liệu | Vai trò |
|---|---|
| Train | Dùng để cập nhật trọng số mô hình |
| Validation | Theo dõi và lựa chọn checkpoint trong quá trình huấn luyện |
| Test | Đánh giá cuối cùng sau khi huấn luyện |

Sau đó, chương trình thống kê số lượng mẫu của từng tập và số lượng mẫu thuộc mỗi nhãn bằng `Counter`.

Kết quả được lưu vào:

```text
outputs/new_run/data_summary.json
```

Việc lưu thống kê giúp kiểm tra phân bố dữ liệu được sử dụng trong mỗi lần chạy.

**Bước 3: Tải tokenizer và mô hình**

```python
tokenizer, model = load_model_and_tokenizer(config["model"])
```

Chương trình sử dụng hàm `load_model_and_tokenizer()` từ `src/model.py` để tải mô hình và tokenizer theo tên checkpoint được khai báo trong cấu hình.

**Bước 4: Tokenize dữ liệu**

```python
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=config["max_length"]
    )
```

Văn bản được chuyển thành các token và ID để mô hình có thể xử lý.

Trong đó:

- `batch["text"]`: Văn bản đầu vào.
- `truncation=True`: Cắt ngắn văn bản nếu vượt quá giới hạn.
- `max_length`: Độ dài tối đa được khai báo trong file cấu hình.

Sau đó, thao tác tokenize được áp dụng riêng cho ba tập dữ liệu:

```python
train_tokens = train_set.map(tokenize, batched=True)
val_tokens = val_set.map(tokenize, batched=True)
test_tokens = test_set.map(tokenize, batched=True)
```

**Bước 5: Cấu hình huấn luyện**

Chương trình sử dụng `TrainingArguments` để thiết lập các tham số:

| Tham số | Chức năng |
|---|---|
| `output_dir` | Thư mục lưu checkpoint |
| `seed` | Seed cho quá trình huấn luyện |
| `data_seed` | Seed cho thao tác liên quan đến dữ liệu |
| `num_train_epochs` | Số epoch huấn luyện |
| `per_device_train_batch_size` | Batch size cho tập train trên mỗi thiết bị |
| `per_device_eval_batch_size` | Batch size cho tập đánh giá trên mỗi thiết bị |
| `learning_rate` | Tốc độ học |
| `weight_decay` | Hệ số điều chuẩn trọng số |
| `eval_strategy="epoch"` | Đánh giá sau mỗi epoch |
| `save_strategy="epoch"` | Lưu checkpoint sau mỗi epoch |
| `logging_strategy="epoch"` | Ghi nhận log sau mỗi epoch |
| `load_best_model_at_end=True` | Khôi phục mô hình từ checkpoint tốt nhất ở cuối quá trình huấn luyện |
| `metric_for_best_model` | Chỉ số được sử dụng để lựa chọn checkpoint |
| `greater_is_better=False` | Ưu tiên giá trị nhỏ hơn của chỉ số lựa chọn |
| `save_total_limit=2` | Giới hạn số checkpoint được giữ lại ở mức 2 |
| `report_to="none"` | Không gửi log đến các nền tảng theo dõi bên ngoài |
| `optim="adamw_torch"` | Sử dụng optimizer AdamW của PyTorch |

Các giá trị như số epoch, batch size và learning rate được đọc từ file cấu hình, không được cố định trực tiếp trong module.

**Bước 6: Khởi tạo Trainer**

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
```

`Trainer` quản lý quá trình huấn luyện và đánh giá.

Các thành phần chính:

- `model`: Mô hình DistilBERT cần fine-tune.
- `args`: Các tham số huấn luyện.
- `train_dataset`: Dữ liệu dùng để huấn luyện.
- `eval_dataset`: Dữ liệu dùng để đánh giá trong quá trình huấn luyện.
- `processing_class`: Tokenizer được sử dụng.
- `data_collator`: Padding động các mẫu trong từng batch.
- `compute_metrics`: Hàm tính các chỉ số đánh giá từ `src/evaluate.py`.

**Bước 7: Fine-tune mô hình**

```python
start_time = datetime.now(timezone.utc).isoformat()
train_result = trainer.train()
trainer.save_state()
save_training_history(trainer.state.log_history, output)
```

`trainer.train()` bắt đầu quá trình fine-tuning.

Sau khi huấn luyện:

- `trainer.save_state()` lưu trạng thái của Trainer.
- `trainer.state.log_history` chứa lịch sử log của quá trình huấn luyện.
- `save_training_history()` chuyển lịch sử này cho module `src/visualize.py` để lưu kết quả huấn luyện.

**Bước 8: Đánh giá trên tập test**

```python
test_output = trainer.predict(test_tokens)
```

Sau khi huấn luyện, chương trình sử dụng Trainer để dự đoán trên tập test.

Do `load_best_model_at_end=True`, Trainer đã khôi phục trọng số từ checkpoint được lựa chọn theo validation trước khi thực hiện bước đánh giá cuối cùng.

Kết quả dự đoán được chuyển đến:

```python
summary = evaluate_and_save(
    test_set,
    test_output.predictions,
    output,
    trainer.state.best_model_checkpoint,
    trainer.state.best_metric
)
```

Hàm `evaluate_and_save()` từ `src/evaluate.py` thực hiện:

- Tính Accuracy.
- Tính Precision, Recall và F1-score cho lớp Positive.
- Tạo confusion matrix.
- Xuất classification report.
- Lưu dự đoán của từng mẫu test.
- Lưu các kết quả đánh giá vào thư mục đầu ra.

**Bước 9: Xuất mô hình tốt nhất**

```python
trainer.save_model(str(output / "best_model"))
tokenizer.save_pretrained(str(output / "best_model"))
```

Mô hình và tokenizer được lưu vào:

```text
outputs/new_run/best_model/
```

Thư mục này có thể được sử dụng cho quá trình inference sau huấn luyện.

**Bước 10: Lưu thông tin lần chạy**

Chương trình tạo file `run_info.json`, bao gồm:

| Trường | Nội dung |
|---|---|
| `started_at_utc` | Thời gian bắt đầu huấn luyện theo UTC |
| `finished_at_utc` | Thời gian kết thúc huấn luyện theo UTC |
| `config` | Cấu hình được sử dụng |
| `python` | Phiên bản Python |
| `pytorch` | Phiên bản PyTorch |
| `transformers` | Phiên bản Transformers |
| `device` | Thiết bị tính toán được TrainingArguments sử dụng |
| `training_loss` | Training loss được Trainer báo cáo |
| `best_checkpoint` | Đường dẫn checkpoint được lựa chọn |
| `best_val_loss` | Giá trị chỉ số lựa chọn checkpoint được Trainer ghi nhận |

Cuối cùng, chương trình in kết quả đánh giá dưới dạng JSON, in đường dẫn thư mục lưu kết quả và trả về `summary`.

## 4. Cách chạy chương trình

Sử dụng cấu hình và thư mục đầu ra mặc định:

```bash
python -m src.train
```

Chỉ định file cấu hình:

```bash
python -m src.train --config configs/default.json
```

Chỉ định thư mục lưu kết quả:

```bash
python -m src.train --output outputs/experiment_01
```

Kết hợp cả hai tham số:

```bash
python -m src.train --config configs/default.json --output outputs/experiment_01
```

Có thể sử dụng `python src/train.py` khi chạy từ thư mục gốc của repository và môi trường Python nhận diện được package `src`.

## 5. Các kết quả đầu ra

Với thư mục đầu ra mặc định `outputs/new_run`, module tạo hoặc sử dụng các thành phần sau:

| Đường dẫn | Nội dung |
|---|---|
| `data_summary.json` | Thống kê số lượng mẫu và nhãn của ba tập dữ liệu |
| `checkpoints/` | Các checkpoint được lưu trong quá trình huấn luyện |
| `best_model/` | Mô hình và tokenizer sau khi khôi phục checkpoint tốt nhất |
| `run_info.json` | Cấu hình, phiên bản thư viện, thời gian và thông tin huấn luyện |
| Các file lịch sử huấn luyện | Được tạo thông qua `save_training_history()` |
| `test_metrics.json` | Các chỉ số đánh giá trên tập test |
| `confusion_matrix.csv` | Ma trận nhầm lẫn dạng CSV |
| `confusion_matrix.png` | Hình ảnh ma trận nhầm lẫn |
| `classification_report.txt` | Báo cáo phân loại |
| `test_predictions.csv` | Dự đoán của mô hình cho từng mẫu test |

Các file đánh giá được tạo thông qua `evaluate_and_save()` trong `src/evaluate.py`.

## 6. Luồng hoạt động tổng thể

```text
configs/default.json
        |
        v
   Đọc cấu hình
        |
        v
  Chuẩn bị dữ liệu IMDb
        |
        v
 Train / Validation / Test
        |
        v
 Tải Tokenizer và DistilBERT
        |
        v
     Tokenization
        |
        v
  Khởi tạo Hugging Face Trainer
        |
        v
     Fine-tuning
        |
        v
 Đánh giá trên Validation
        |
        v
 Lựa chọn checkpoint tốt nhất
        |
        v
 Khôi phục checkpoint tốt nhất
        |
        v
 Dự đoán trên tập Test
        |
        v
 Tính metrics và lưu kết quả
        |
        v
 Xuất best_model và run_info.json
```

## 7. Vai trò trong hệ thống

`train.py` là module trung tâm kết nối các thành phần của Lab3.

Module không tự triển khai riêng thuật toán Transformer hay các hàm tính metrics mà sử dụng những module chuyên biệt:

- `data.py`: Chuẩn bị dữ liệu.
- `model.py`: Tải tokenizer và mô hình.
- `evaluate.py`: Đánh giá kết quả.
- `utils.py`: Đọc cấu hình và lưu JSON.
- `visualize.py`: Lưu lịch sử huấn luyện.

Cách tổ chức này giúp tách biệt các nhiệm vụ, dễ theo dõi, kiểm thử và bảo trì code.

## 8. Lưu ý về lựa chọn checkpoint

Trong code hiện tại, cấu hình được sử dụng là:

```python
metric_for_best_model=config["selection_metric"],
greater_is_better=False
```

Điều này có nghĩa Trainer sẽ ưu tiên checkpoint có giá trị **nhỏ hơn** đối với chỉ số được lựa chọn.

Vì vậy, cần kiểm tra `selection_metric` trong `configs/default.json`:

- Nếu sử dụng `eval_loss`, thiết lập `greater_is_better=False` là phù hợp.
- Nếu sử dụng `eval_accuracy`, cần đặt `greater_is_better=True` để ưu tiên accuracy cao hơn.

Ngoài ra, giá trị `trainer.state.best_metric` được truyền vào tham số `best_val_loss` của `evaluate_and_save()`. Vì vậy, giá trị được lưu dưới tên `best_val_loss` chỉ thực sự là validation loss khi `selection_metric` là `eval_loss`.

Cần bảo đảm cấu hình lựa chọn checkpoint đúng với mục tiêu đánh giá của nhóm để tránh chọn nhầm mô hình tốt nhất.
