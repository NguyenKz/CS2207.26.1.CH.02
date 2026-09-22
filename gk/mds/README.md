# Tài liệu chuẩn bị demo ANN

Thư mục này chứa phần giải thích, lộ trình học và thiết kế notebook cho demo Artificial Neural Network (ANN). Tài liệu viết cho người đã biết Python và một số khái niệm machine learning cơ bản, nhưng chưa học ANN một cách có hệ thống.

## Thứ tự đọc đề xuất

1. [`very-simple-liner-model.md`](./very-simple-liner-model.md): model tuyến tính `z = w·x + b`, ý nghĩa `w`/`b`, và bước if để ra Đậu/Rớt.
2. [`very-simple-ann.md`](./very-simple-ann.md): nối activation vào `z` — bước đầu của 1 neuron / perceptron.
3. [`overview.md`](./overview.md): bức tranh tổng quan ANN (neuron, MLP, activation, giới hạn).
4. [`how-it-works.md`](./how-it-works.md): mạng biến input thành output ra sao.
5. [`how-to-train.md`](./how-to-train.md): loss, gradient descent và backpropagation.
6. [`simple-notebook-demo.md`](./simple-notebook-demo.md): chuẩn bị notebook demo.
7. [`learn-first.md`](./learn-first.md): checklist kiến thức trước khi thuyết trình.
8. [`demo-directions.md`](./demo-directions.md): kịch bản demo và hướng mở rộng web app.

## Mục lục

### Nền tảng (đọc tay trước)

- [`very-simple-liner-model.md`](./very-simple-liner-model.md)
- [`very-simple-ann.md`](./very-simple-ann.md)
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

- Khi mới bắt đầu: làm phép tính tay trong `very-simple-liner-model.md`, rồi đọc `very-simple-ann.md` để thấy activation thay bước if thế nào.
- Khi chuẩn bị phần nói: đọc `overview.md`, `how-it-works.md` và `demo-directions.md`.
- Khi viết notebook: dùng `simple-notebook-demo.md` như danh sách cell.
- Khi gặp thuật ngữ chưa rõ: tra `learn-first.md` rồi quay về PDF nguồn.
