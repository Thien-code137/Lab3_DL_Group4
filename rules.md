# RULES — Quy tắc thực nghiệm và nghiệm thu

1. Không dùng test để chọn mô hình hoặc tinh chỉnh siêu tham số. Chỉ dùng validation để chọn checkpoint.
2. Ghi rõ seed, dataset, split, cấu hình, phiên bản thư viện và thiết bị cho mỗi lượt chạy.
3. `best_val_loss` là **chỉ số nhỏ nhất của validation loss**, không phải tên file trọng số. File `best_model.pth` hoặc thư mục checkpoint là hiện vật riêng.
4. Chỉ báo cáo kết quả của từng lần chạy theo đúng file metrics và checkpoint tương ứng.
5. Không sửa số liệu trong `outputs/` bằng tay. Mọi hình và bảng cần ghi rõ đường dẫn nguồn.
6. Khi có thay đổi mã nguồn hoặc cấu hình, cập nhật tài liệu liên quan để thông tin nhất quán.
7. Chỉ thêm dữ liệu và trọng số được phép chia sẻ; không đẩy cache, venv hoặc dữ liệu bản quyền lên Git.
8. EDA/PCA chỉ dùng tập train để quyết định đặc trưng và pipeline; validation/test được giữ riêng.
