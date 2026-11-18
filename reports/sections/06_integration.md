# 06 Tổ chức mã nguồn và kiểm tra

| Module | Chức năng |
|---|---|
| `data.py` | Tải, chia dữ liệu, lấy mẫu cân bằng |
| `model.py` | Tải DistilBERT và tokenizer |
| `train.py` | Điều phối huấn luyện, lựa chọn checkpoint |
| `evaluate.py` | Metrics, dự đoán test, confusion matrix |
| `visualize.py` | Lưu `history.csv`, biểu đồ loss |
| `predict.py` | Suy luận bằng model đã xuất |
| `utils.py` | Xác thực cấu hình, xuất JSON |

Test cấu hình và metrics nằm trong thư mục `tests/` và chạy bằng `python -m unittest discover -s tests -v`. Unit test chỉ kiểm tra logic và xuất file bằng dữ liệu nhỏ; không thay thế cho huấn luyện trên IMDb.

Checkpoint được lưu trong `outputs/new_run/best_model/` cùng tokenizer. Có thể kiểm tra suy luận bằng lệnh `python -m src.predict --checkpoint outputs/new_run/best_model --text "I enjoyed the film"` sau khi mô hình được huấn luyện và xuất thành công.
