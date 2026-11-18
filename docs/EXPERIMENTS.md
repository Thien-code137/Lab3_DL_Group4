# Phân biệt các lần thực nghiệm

| Lần chạy | Nguồn trong project | Test accuracy | F1 Positive | Best validation loss | Checkpoint |
|---|---|---:|---:|---:|---|
| A – lịch sử | `outputs/exercise2/test_metrics.json`, `best_val_loss.json` | 89% | 90,18% | 0,283972 | 125 (validation accuracy) |
| B – notebook tổng hợp gốc | `notebooks/archive/practice_3_original.ipynb` | 86,5% | 86,57% | 0,264297 | 250 (validation loss) |
| C – notebook mới | `notebooks/practice_3.ipynb` | Chưa chạy | Chưa chạy | Chưa chạy | Chưa có |

Các số liệu A và B là output đã lưu trong ZIP gốc, không phải kết quả của lần chạy mới. Không trộn các metric, confusion matrix hay checkpoint giữa A/B/C. Báo cáo DOCX gốc được giữ nguyên để tham khảo thí nghiệm A; có thể còn dẫn tên notebook lịch sử không có trong ZIP. Notebook mới chuẩn hóa HTML, loại trùng và lấy test cân bằng, trong khi `src/data.py` gốc lấy test ngẫu nhiên; vì vậy không xem hai đường chạy là tương đương tuyệt đối. Sau khi chạy C, cập nhật báo cáo bằng toàn bộ kết quả của cùng lần chạy nếu muốn dùng C làm kết quả chính thức.

ZIP gốc không có trọng số mô hình đã fine-tune hoặc toàn bộ IMDb. Cần Internet để tải dữ liệu/mô hình và huấn luyện lại để tạo checkpoint sử dụng được.
