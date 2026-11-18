# 05 Cấu hình thử nghiệm và lựa chọn checkpoint

Notebook Exercise 2 ban đầu sử dụng 2 epoch, batch 8, learning rate 2e-5 trên CPU và chọn checkpoint theo validation accuracy. Các chỉ số test đã lưu tại `outputs/exercise2/`.

Pipeline module trong `src/train.py` chọn checkpoint theo **validation loss** thấp nhất. Hai phương pháp lựa chọn có thể dẫn đến checkpoint khác nhau. Khi so sánh, cần giữ nguyên tập dữ liệu, seed và cấu hình; thay đổi duy nhất tiêu chí chọn rồi ghi rõ kết quả.

| Cấu hình | Metric chọn | Kết quả |
|---|---|---|
| Notebook Exercise 2 đã lưu | Validation accuracy | 89% accuracy trên 200 mẫu test |
| Pipeline `src/train.py` | Validation loss | Ghi sau khi chạy |

Không điền số đo của pipeline chưa chạy. Thí nghiệm bổ sung có thể khảo sát `max_length` 128/256 hoặc learning rate 1e-5/2e-5, nhưng cần đủ thời gian và đánh giá validation trước khi sử dụng test.
