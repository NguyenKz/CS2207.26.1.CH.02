# Chiều sâu (nhiều layer) trong một ví dụ rất đơn giản

## Tiếp nối phần trước:

- **Chiều rộng**: số neuron trong **một** layer (`very-simple-n-neurons.md`).
- **Chiều sâu**: số layer xếp **nối tiếp** nhau. Output của layer này là input của layer kế.

```text
input  →  layer 1  →  layer 2  →  ...  →  layer L (output)
  x         a¹          a²                  aᴸ = dự đoán
```

- MLP (Multi-Layer Perceptron) = mạng feed-forward có **ít nhất một hidden layer** giữa input và output. Đó chính là tăng chiều sâu so với 1 neuron / logistic regression.

## Ý tưởng:

- Mỗi layer vẫn làm cùng một việc: với mỗi neuron trong layer, `z = w·(input layer) + b`, rồi `a = f(z)`.
- Khác chỗ: **input của layer sau không còn là `x` gốc**, mà là vector activation `a` của layer trước.
- Layer ở giữa gọi là **hidden layer** — không phải nhãn cuối, chỉ là biểu diễn trung gian.
- Activation **phi tuyến** ở hidden rất quan trọng: nếu mọi layer chỉ tuyến tính (`f(z) = z`), xếp bao nhiêu layer cũng rút gọn thành **một** phép tuyến tính → độ sâu mất tác dụng.

## Ví dụ tay: 2 hidden layer rồi mới Đậu/Rớt

Sinh viên `x_1 = 0.8`, `x_2 = 0.6`. Mạng siêu nhỏ:

```text
input (2) → hidden1 (2 neuron, ReLU) → hidden2 (2 neuron, ReLU) → output (1, sigmoid)
```

**Layer 1** (nhận `x`):

```text
a¹_1 = ReLU(w·x + b)   # neuron 1 của hidden1
a¹_2 = ReLU(w·x + b)   # neuron 2 của hidden1
```

**Layer 2** (nhận `a¹ = (a¹_1, a¹_2)`, không nhận `x`):

```text
a²_1 = ReLU(w·a¹ + b)
a²_2 = ReLU(w·a¹ + b)
```

**Output** (nhận `a²`):

```text
a_out = sigmoid(w·a² + b)   # ≈ xác suất Đậu
```

Đếm độ sâu theo cách nói chuyện thông thường trong demo:

- Có **2 hidden layer** → mạng “sâu hơn” ví dụ chỉ 1 hidden trong `very-simple-n-neurons.md`.
- Input layer thường **không** tính là layer học tham số (chỉ đưa `x` vào).

## Rộng vs sâu — nhớ một câu:

| | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| Rộng | Nhiều neuron **trong một** layer | Hidden có 32 neuron |
| Sâu | Nhiều layer **nối tiếp** | Input → h1 → h2 → out |

- Rộng hơn: mỗi layer “nhìn” nhiều kiểu hơn cùng lúc.
- Sâu hơn: biến đổi **theo tầng** — tầng sau làm việc trên đặc trưng đã được tầng trước xử lý.
- Demo `make_moons` thường đủ với **1 hidden** không quá rộng; sâu hơn không tự động tốt hơn (dễ train khó, dễ overfit nếu data nhỏ).

## Cần nhớ:

- ANN = khung chung; **rộng** và **sâu** là hai núm chỉnh kiến trúc, không phải định nghĩa ANN.
- Chiều sâu có ích khi có activation phi tuyến giữa các layer.
- Tiếp: [`overview.md`](./overview.md) (MLP, phi tuyến, giới hạn) và [`how-it-works.md`](./how-it-works.md) (forward qua cả mạng).
