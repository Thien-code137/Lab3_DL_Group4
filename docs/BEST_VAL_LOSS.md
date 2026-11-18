# Best validation loss Lab 3

Trong lần huấn luyện được lưu tại notebook Exercise 2 lịch sử (file riêng không có trong ZIP gốc; xem `outputs/exercise2/` và `best_val_loss.json`), checkpoint được chọn tại bước 125 theo validation accuracy, đạt 90,5% ở epoch 1.

Pipeline trong `src/train.py` chọn checkpoint theo validation loss thấp nhất. Sau khi chạy `python main.py`, các giá trị `best_val_loss` và `best_checkpoint` nằm trong `outputs/new_run/test_metrics.json` và `outputs/new_run/run_info.json`; bảng qua từng epoch nằm trong `history.csv`. Những file này được tạo khi hoàn thành lần huấn luyện tương ứng.
