# Tài liệu chuẩn bị demo ANN

Thư mục này chứa phần giải thích, lộ trình học và thiết kế notebook cho demo Artificial Neural Network. Tài liệu viết cho người đã biết Python và một số khái niệm machine learning cơ bản, nhưng chưa học ANN một cách có hệ thống.

## Thứ tự đọc đề xuất

1. [`very-simple.md`](./very-simple.md): nắm một neuron bằng một phép tính nhỏ.
2. [`overview.md`](./overview.md): biết ANN gồm những thành phần nào.
3. [`how-it-works.md`](./how-it-works.md): hiểu mạng biến input thành output ra sao.
4. [`how-to-train.md`](./how-to-train.md): nối forward pass với loss, gradient descent và backpropagation.
5. [`simple-notebook-demo.md`](./simple-notebook-demo.md): chuẩn bị notebook demo.
6. [`learn-first.md`](./learn-first.md): kiểm tra phần kiến thức còn thiếu trước khi thuyết trình.
7. [`demo-directions.md`](./demo-directions.md): chọn kịch bản và hướng mở rộng thành web app.

## Mục lục

### Nền tảng

- [`very-simple.md`](./very-simple.md)
- [`overview.md`](./overview.md)
- [`how-it-works.md`](./how-it-works.md)

### Huấn luyện và thực hành

- [`how-to-train.md`](./how-to-train.md)
- [`simple-notebook-demo.md`](./simple-notebook-demo.md)
- [`learn-first.md`](./learn-first.md)

### Thiết kế demo

- [`demo-directions.md`](./demo-directions.md)

## Nguồn PDF

Các file PDF nằm ngay trong thư mục `gk/`:

| Nguồn | Nội dung dùng trong tài liệu |
| --- | --- |
| [`7-nn1-intro.ppt.pdf`](../7-nn1-intro.ppt.pdf) | Artificial neuron, bias, activation, topology và MLP |
| [`chap4_ann.pdf`](../chap4_ann.pdf) | Perceptron, luật cập nhật trọng số, XOR, gradient descent và backpropagation |
| [`lecture-21.pdf`](../lecture-21.pdf) | Logistic regression như một mạng đơn giản, MLP, activation và gradient descent |
| [`mit15_773_s24_lec01.pdf`](../mit15_773_s24_lec01.pdf) | Hidden layer, số lượng tham số và ví dụ forward pass |

Số slide hoặc số trang được ghi trong từng tài liệu khi cần đối chiếu. Các nhận xét về cấu hình demo được ghi rõ là đề xuất, không phải kết quả đã chạy.

## Cách dùng

- Khi cần hiểu nhanh, đọc `very-simple.md` và chạy theo các phép tính bằng tay.
- Khi cần chuẩn bị phần nói, đọc `overview.md`, `how-it-works.md` và `demo-directions.md`.
- Khi cần viết notebook, dùng `simple-notebook-demo.md` như danh sách cell và nội dung cần kiểm tra.
- Khi gặp thuật ngữ chưa rõ, tra lại `learn-first.md` rồi quay về PDF nguồn tương ứng.

