# Demo Artificial Neural Network

Tài liệu này chuẩn bị cho phần thuyết trình về Artificial Neural Network (ANN). Mục tiêu là giúp người trình bày đi từ một neuron đơn giản đến một mạng nhiều lớp, sau đó minh họa quá trình huấn luyện bằng một notebook Python.

## Mục tiêu của demo

Demo chính dùng dữ liệu hai chiều tạo bằng `make_moons` của scikit-learn. Người xem có thể nhìn thấy:

- Logistic regression tạo một ranh giới tuyến tính.
- MLP (Multi-Layer Perceptron) dùng hidden layer để học ranh giới phi tuyến.
- Loss thay đổi trong quá trình huấn luyện.
- Số neuron, activation function và mức nhiễu ảnh hưởng đến decision boundary.

XOR được dùng như ví dụ lý thuyết ngắn để giải thích vì sao một perceptron đơn không giải quyết được mọi bài toán phân loại.

## Cấu trúc tài liệu

Các tài liệu ANN nằm trong [`gk/mds/`](./gk/mds/).

| Tài liệu | Mục đích |
| --- | --- |
| [`very-simple.md`](./gk/mds/very-simple.md) | Giải thích neuron bằng phép tính nhỏ có thể làm bằng tay |
| [`overview.md`](./gk/mds/overview.md) | Bức tranh tổng quan về ANN và các thành phần chính |
| [`how-it-works.md`](./gk/mds/how-it-works.md) | Forward propagation, hidden layer và activation |
| [`how-to-train.md`](./gk/mds/how-to-train.md) | Loss, gradient descent, backpropagation và đánh giá model |
| [`simple-notebook-demo.md`](./gk/mds/simple-notebook-demo.md) | Thiết kế notebook demo theo từng cell |
| [`learn-first.md`](./gk/mds/learn-first.md) | Lộ trình học và checklist chuẩn bị thuyết trình |
| [`demo-directions.md`](./gk/mds/demo-directions.md) | Các hướng demo, kịch bản chính và hướng mở rộng thành web app |

Đọc [`gk/mds/README.md`](./gk/mds/README.md) để xem mục lục đầy đủ và nguồn PDF.

## Lộ trình thực hiện

1. Đọc các phần cơ bản về neuron, bias, activation và perceptron.
2. Hiểu forward propagation, hidden layer và giới hạn của ranh giới tuyến tính.
3. Chạy notebook với `make_moons`.
4. So sánh logistic regression với `MLPClassifier`.
5. Dùng decision boundary và loss curve làm hình ảnh chính khi thuyết trình.
6. Nếu notebook đã ổn định, chuyển cùng luồng xử lý sang Streamlit.

## Phạm vi hiện tại

Commit này chỉ chứa tài liệu Markdown và các liên kết tới tài liệu nguồn. Chưa có file `.py`, `.ipynb`, HTML hay web app. Các con số như `n_samples=400` và `noise=0.20` trong tài liệu là tham số cấu hình cho demo, không phải kết quả thực nghiệm hay số liệu đánh giá.

## Nguồn học tập

- [`gk/7-nn1-intro.ppt.pdf`](./gk/7-nn1-intro.ppt.pdf): trực giác sinh học, neuron, bias, activation và topology.
- [`gk/chap4_ann.pdf`](./gk/chap4_ann.pdf): perceptron, XOR, MLP, gradient descent và backpropagation.
- [`gk/lecture-21.pdf`](./gk/lecture-21.pdf): logistic regression, MLP, activation, gradient descent và các lưu ý về deep learning.
- [`gk/mit15_773_s24_lec01.pdf`](./gk/mit15_773_s24_lec01.pdf): hidden layer, activation và ví dụ tính forward pass.

