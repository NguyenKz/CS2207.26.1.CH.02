# Tài liệu chuẩn bị demo ANN

Thư mục này chứa phần giải thích, lộ trình học và thiết kế notebook cho demo Artificial Neural Network (ANN). Tài liệu viết cho người đã biết Python và một số khái niệm machine learning cơ bản, nhưng chưa học ANN một cách có hệ thống.

## Thứ tự đọc đề xuất

1. [`very-simple-liner-model.md`](./very-simple-liner-model.md): model tuyến tính `z = w·x + b`, ý nghĩa `w`/`b`, và bước if để ra Đậu/Rớt.
2. [`very-simple-ann.md`](./very-simple-ann.md): nối activation vào `z`, bước đầu của 1 neuron / perceptron.
3. [`very-simple-n-neurons.md`](./very-simple-n-neurons.md): chiều rộng, nhiều neuron trong một layer.
4. [`very-simple-depth.md`](./very-simple-depth.md): chiều sâu, nhiều layer nối tiếp (MLP).
5. [`overview.md`](./overview.md): bức tranh tổng quan ANN (neuron, MLP, activation, giới hạn).
6. [`how-it-works.md`](./how-it-works.md): mạng biến input thành output ra sao.
7. [`how-to-train.md`](./how-to-train.md): loss, gradient descent và backpropagation.
8. [`simple-notebook-demo.md`](./simple-notebook-demo.md): chuẩn bị notebook demo.
9. [`learn-first.md`](./learn-first.md): checklist kiến thức trước khi thuyết trình.
10. [`demo-directions.md`](./demo-directions.md): kịch bản demo và hướng mở rộng web app.

## Mục lục

### Nền tảng (đọc tay trước)

- [`very-simple-liner-model.md`](./very-simple-liner-model.md)
- [`very-simple-ann.md`](./very-simple-ann.md)
- [`very-simple-n-neurons.md`](./very-simple-n-neurons.md)
- [`very-simple-depth.md`](./very-simple-depth.md)
- [`overview.md`](./overview.md)
- [`how-it-works.md`](./how-it-works.md)

### Huấn luyện và thực hành

- [`how-to-train.md`](./how-to-train.md)
- [`simple-notebook-demo.md`](./simple-notebook-demo.md)
- [`ann_from_scratch_demo.ipynb`](./ann_from_scratch_demo.ipynb)
- [`learn-first.md`](./learn-first.md)

### Thiết kế demo

- [`demo-directions.md`](./demo-directions.md)
- [`mnist_ann_training.ipynb`](./mnist_ann_training.ipynb): notebook tổng hợp đọc thư mục dataset local, minh họa và train bốn model Predict.
- [`mnist_dataset.ipynb`](./mnist_dataset.ipynb): đọc `ANN_DIGIT_SIZE` từ `.env`, chạy một lần cho mỗi kích thước và xuất thư mục raw 28×28/normalized `N×N` cùng các file JSON.
- [`mnist_train.ipynb`](./mnist_train.ipynb): đọc thư mục dataset, train bốn model và xuất file weight.
- [`mnist_test.ipynb`](./mnist_test.ipynb): đọc weight, chọn mẫu test và kiểm tra prediction/probability.
- [`mnist_models.ipynb`](./mnist_models.ipynb): đọc, kiểm tra cấu hình cố định và xuất `mnist_model_config.json` vào thư mục dataset.

Cấu hình mặc định của bốn model và augmentation nằm ở [`../web/backend/model_config.py`](../web/backend/model_config.py). Sửa file này, chạy `mnist_models.ipynb` để ghi cấu hình, sau đó chạy `mnist_train.ipynb` để train và cập nhật artifact cho web.

Kích thước ảnh hợp lệ là `8`, `16` hoặc `24`. Nếu không có `.env`, pipeline mặc định dùng `8×8`. Dataset nằm trong `gk/web/backend/artifacts/mnist_dataset/NxN/`, còn artifact model nằm ở `gk/web/backend/artifacts/digits_models_NxN.json`.

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

- Khi mới bắt đầu: `very-simple-liner-model` → `very-simple-ann` (1 neuron) → `very-simple-n-neurons` (rộng) → `very-simple-depth` (sâu).
- Khi chuẩn bị phần nói: đọc `overview.md`, `how-it-works.md` và `demo-directions.md`.
- Khi viết notebook: dùng `simple-notebook-demo.md` như danh sách cell.
- Khi gặp thuật ngữ chưa rõ: tra `learn-first.md` rồi quay về PDF nguồn.
