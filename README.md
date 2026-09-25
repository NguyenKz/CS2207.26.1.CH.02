# Demo Artificial Neural Network

Tài liệu này chuẩn bị cho phần thuyết trình về Artificial Neural Network (ANN). Mục tiêu là giúp người trình bày đi từ model tuyến tính đơn giản → một neuron + activation → mạng nhiều lớp, rồi tự xây và huấn luyện một ANN bằng NumPy.

## Mục tiêu của demo

Demo chính dùng dữ liệu Iris của scikit-learn, nhưng chỉ dùng scikit-learn để lấy dữ liệu. Model ANN được tự viết bằng NumPy. Người xem có thể nhìn thấy:

- Bốn số đo đi qua input layer, hidden layer và output layer.
- Hidden layer có 8 neuron, cho phép so sánh `tanh`, `sigmoid`, `ReLU`, `Leaky ReLU`, `softplus` và `identity`.
- Output layer có 3 neuron dùng `softmax` cho ba loài hoa.
- Code tự tính forward pass, cross-entropy loss, backpropagation và gradient descent.
- Giao diện hiển thị input/output của từng layer cùng loss curve.

XOR được dùng như ví dụ lý thuyết ngắn để giải thích vì sao một perceptron đơn không giải quyết được mọi bài toán phân loại.

## Cấu trúc tài liệu

Các tài liệu ANN nằm trong [`gk/mds/`](./gk/mds/).

| Tài liệu | Mục đích |
| --- | --- |
| [`very-simple-liner-model.md`](./gk/mds/very-simple-liner-model.md) | Model tuyến tính `z = w·x + b` qua ví dụ Đậu/Rớt, làm bằng tay |
| [`very-simple-ann.md`](./gk/mds/very-simple-ann.md) | Từ `z` sang nhãn qua activation; 1 neuron / perceptron |
| [`very-simple-n-neurons.md`](./gk/mds/very-simple-n-neurons.md) | Chiều rộng: n neuron trong một layer |
| [`very-simple-depth.md`](./gk/mds/very-simple-depth.md) | Chiều sâu: nhiều layer nối tiếp (MLP) |
| [`overview.md`](./gk/mds/overview.md) | Bức tranh tổng quan về ANN và các thành phần chính |
| [`how-it-works.md`](./gk/mds/how-it-works.md) | Forward propagation, hidden layer và activation |
| [`how-to-train.md`](./gk/mds/how-to-train.md) | Loss, gradient descent, backpropagation và đánh giá model |
| [`simple-notebook-demo.md`](./gk/mds/simple-notebook-demo.md) | Thiết kế notebook demo theo từng cell |
| [`ann_from_scratch_demo.ipynb`](./gk/mds/ann_from_scratch_demo.ipynb) | Notebook chạy ANN tự viết bằng NumPy |
| [`learn-first.md`](./gk/mds/learn-first.md) | Lộ trình học và checklist chuẩn bị thuyết trình |
| [`demo-directions.md`](./gk/mds/demo-directions.md) | Các hướng demo, kịch bản chính và hướng mở rộng thành web app |

Đọc [`gk/mds/README.md`](./gk/mds/README.md) để xem mục lục đầy đủ và nguồn PDF.

## Chạy notebook

Tạo virtual environment và cài dependency:

```bash
./setup_venv.sh
source .venv/bin/activate
```

Mở notebook:

```bash
jupyter notebook gk/mds/ann_from_scratch_demo.ipynb
```

Kernel cần chọn trong Jupyter là `Python (CS2207 ANN)`.

## Chạy web demo realtime

Web demo có tab `Train`, dùng React + TypeScript ở frontend và FastAPI + NumPy ở backend. Sáu activation (`tanh`, `sigmoid`, `ReLU`, `Leaky ReLU`, `softplus`, `identity`) được train đồng thời; delay chỉ làm chậm event để dễ quan sát, không làm model học tốt hơn.

Terminal 1 — backend:

```bash
source .venv/bin/activate
uvicorn gk.web.backend.server:app --reload --port 6788
```

Terminal 2 — frontend:

```bash
cd gk/web/frontend
npm install
npm run dev
```

Mở `http://localhost:5113`. Chọn delay `0.01s`, `0.05s` hoặc `0.1s` để nhìn rõ từng epoch, loss và validation accuracy của mỗi activation.

Hoặc chạy cả backend và frontend bằng một lệnh:

```bash
./run.sh
```

## Lộ trình thực hiện

1. Đọc `very-simple-liner-model` → `very-simple-ann` → `very-simple-n-neurons` → `very-simple-depth` (1 neuron → rộng → sâu).
2. Đọc `overview.md` để nối sang MLP, phi tuyến và giới hạn của perceptron.
3. Hiểu forward propagation, hidden layer và ranh giới tuyến tính vs phi tuyến.
4. Chạy notebook Iris với ANN tự viết bằng NumPy.
5. Kiểm tra output của từng layer trên một mẫu hoa.
6. Dùng sơ đồ `4 -> 8 -> 3`, loss curve và confusion matrix khi thuyết trình.
7. Chạy web demo React + FastAPI để quan sát training realtime.

## Phạm vi hiện tại

Project hiện có notebook NumPy tự xây và web demo React + FastAPI. Tab Train stream từng epoch qua WebSocket; tab Predict sẽ được bổ sung ở phase sau.

## Nguồn học tập

- [`gk/7-nn1-intro.ppt.pdf`](./gk/7-nn1-intro.ppt.pdf): trực giác sinh học, neuron, bias, activation và topology.
- [`gk/chap4_ann.pdf`](./gk/chap4_ann.pdf): perceptron, XOR, MLP, gradient descent và backpropagation.
- [`gk/lecture-21.pdf`](./gk/lecture-21.pdf): logistic regression, MLP, activation, gradient descent và các lưu ý về deep learning.
- [`gk/mit15_773_s24_lec01.pdf`](./gk/mit15_773_s24_lec01.pdf): hidden layer, activation và ví dụ tính forward pass.
