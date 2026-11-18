# 03 Huấn luyện

## Cấu hình

| Thành phần | Giá trị |
|---|---|
| Seed | 42 |
| Train/validation/test | 1.000/200/200 |
| Epoch | 2 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max length | 256 |
| Optimizer | AdamW |
| Loss | Cross-entropy |

Tham số nằm trong `configs/default.json`. File `src/utils.py` kiểm tra các trường cấu hình bắt buộc trước khi huấn luyện.

## Vòng lặp Trainer

`Trainer` thực hiện forward, tính loss, backward và optimizer step. Cuối mỗi epoch, Trainer chạy validation, ghi `eval_loss`, accuracy và F1. `save_strategy='epoch'` lưu checkpoint, `load_best_model_at_end=True` tải checkpoint có validation loss thấp nhất.

## Kết quả sinh ra

`history.csv`, `loss_curves.png`, `run_info.json`, `best_model/` và các checkpoint thuộc lần chạy. Chỉ nhận xét xu hướng overfit khi đã có đường train/validation loss thực tế.
