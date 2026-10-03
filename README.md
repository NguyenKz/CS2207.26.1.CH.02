# Demo Artificial Neural Network

Tài liệu này chuẩn bị cho phần thuyết trình về Artificial Neural Network (ANN). Mục tiêu là giúp người trình bày đi từ model tuyến tính đơn giản → một neuron + activation → mạng nhiều lớp, rồi tự xây và huấn luyện một ANN bằng NumPy.

## Mục tiêu của demo

Demo notebook dùng dữ liệu Iris để giữ ví dụ gần gũi. Web demo dùng `make_classification` của scikit-learn để tạo nhiều mẫu với độ khó điều chỉnh được; scikit-learn chỉ tạo dữ liệu, còn model ANN được tự viết bằng NumPy. Người xem có thể nhìn thấy:

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

Web demo có tab `Train`, dùng React + TypeScript ở frontend và FastAPI + NumPy ở backend. Sáu activation (`tanh`, `sigmoid`, `ReLU`, `Leaky ReLU`, `softplus`, `identity`) được train đồng thời trên dữ liệu `make_classification`; slider `Difficulty` điều chỉnh class separation, label noise và độ phức tạp cluster. Tổng mẫu mặc định là `300` (`3 class × 100`) nhưng có thể chỉnh trên UI; dữ liệu được chia mặc định thành train `60%`, validation `15%`, test `15%` và holdout cuối `10%`. Holdout chỉ được dùng một lần sau khi train xong để báo final accuracy, không tham gia cập nhật weight hay early stopping. UI cũng cho chỉnh tỷ lệ các tập và `batch size`. Delay chỉ làm chậm event để dễ quan sát, không làm model học tốt hơn. Có thể bật `Early stopping` để dừng từng activation khi validation loss không cải thiện trong 40 epoch.

Terminal 1: backend:

```bash
source .venv/bin/activate
uvicorn gk.web.backend.server:app --reload --port 6788
```

Terminal 2: frontend:

```bash
cd gk/web/frontend
npm install
npm run dev
```

Mở `http://localhost:5113`. Kéo `Difficulty` để tạo bài toán dễ hoặc khó hơn; chọn delay `0.01s`, `0.05s` hoặc `0.1s` để nhìn rõ từng epoch, loss và validation accuracy của mỗi activation.

Hoặc chạy cả backend và frontend bằng một lệnh:

```bash
./run.sh
```

## Tab Predict với chữ số viết tay theo kích thước cấu hình

Tab `Predict` dùng ảnh chữ số viết tay thật từ MNIST ở kích thước gốc 28×28. `ANN_DIGIT_SIZE=28` là kích thước duy nhất được hỗ trợ. Pixel được chuẩn hóa về khoảng `0..1`. Ảnh đi qua đúng pipeline của canvas: khử nhiễu theo 50% cường độ lớn nhất, giữ vùng nét lớn nhất, crop, vuông hóa, area-average resize về 28×28, rồi mới dùng `StandardScaler`. Bốn model được train offline trên cùng một split dữ liệu và cùng `StandardScaler`:

- `Logistic Regression`: baseline tuyến tính chính thức từ scikit-learn, `N*N → 10`.
- MLP một hidden layer: `N*N → H → 10`, với `H = N//2` theo cấu hình mặc định.
- MLP hai hidden layer: `N*N → H → H → 10`.
- Compact tuned MLP: cấu hình nhiều hidden layer, được cố định trong `model_config.py`.

Logistic Regression và ANN dùng cùng tập validation/test. Riêng tập train được augmentation thành ba phiên bản mỗi ảnh: ảnh gốc, ảnh dịch/co giãn nhẹ và ảnh thay đổi độ dày nét. Augmentation chỉ áp dụng cho `fit_indices`, không áp dụng cho validation hoặc test. Cấu hình augmentation nằm trong `model_config.py`; artifact lưu lại số mẫu thực tế đã dùng. Hai MLP minh họa được giữ nhỏ có chủ đích để minh họa underfitting. Mạng tuned dùng hai hidden layer lớn hơn để thể hiện năng lực phi tuyến.

Artifact weight được lưu riêng theo kích thước tại `gk/web/backend/artifacts/digits_models_NxN.json`. Khi chạy web, backend chỉ đọc artifact tương ứng với `.env` và thực hiện forward pass bằng NumPy. Không có quá trình train lại khi mở tab `Predict`.

Tạo file cấu hình local:

```bash
cp .env.example .env
```

Sau đó sửa, nếu cần:

```env
ANN_DIGIT_SIZE=28
```

Tạo lại artifact sau khi thay đổi script hoặc dependency:

```bash
./train_mnist.sh all
```

Lệnh trên chạy bốn model bằng bốn process độc lập, rồi ghép thành artifact tương ứng. Mỗi model cũng có thể chạy riêng:

```bash
./train_mnist.sh logistic
./train_mnist.sh compact_sigmoid
./train_mnist.sh compact_tanh
./train_mnist.sh compact_relu
./train_mnist.sh merge
```

`merge` chỉ ghép lại các shard đã train trong `gk/web/backend/artifacts/mnist_dataset/28x28/model_shards/`. Nếu dataset chưa tồn tại, hãy chạy `mnist_dataset.ipynb` trước. Bốn process cùng lúc cần nhiều RAM hơn; log của chế độ `all` nằm trong thư mục `model_shards/logs/`. Dữ liệu gốc không được commit vào repository.

Phần train được tách thành ba file để dễ chỉnh trong notebook hoặc chạy trực tiếp:

- [`digits_dataset.py`](./gk/web/backend/digits_dataset.py): tải MNIST, đọc IDX và normalize ảnh theo pipeline của UI.
- [`model_config.py`](./gk/web/backend/model_config.py): nguồn cấu hình duy nhất của Logistic Regression, hai ANN minh họa và ANN tuned.
- [`digits_models.py`](./gk/web/backend/digits_models.py): factory scikit-learn, export layer và các helper dùng chung.
- [`train_digits_models.py`](./gk/web/backend/train_digits_models.py): train tuần tự và export artifact đầy đủ.
- [`train_digit_model.py`](./gk/web/backend/train_digit_model.py): train từng model, ghi shard và ghép artifact khi chạy song song.

Notebook tổng hợp nhanh: [`mnist_ann_training.ipynb`](./gk/mds/mnist_ann_training.ipynb).

Ba notebook chuyên dụng nên chạy theo thứ tự:

1. [`mnist_dataset.ipynb`](./gk/mds/mnist_dataset.ipynb): tạo `gk/web/backend/artifacts/mnist_dataset/NxN/` gồm ảnh raw 28×28, ảnh normalized theo kích thước cấu hình, `dataset_meta.json` và `split.json`.
2. [`mnist_models.ipynb`](./gk/mds/mnist_models.ipynb): đọc, kiểm tra cấu hình cố định và ghi `mnist_model_config.json` vào thư mục kích thước hiện tại.
3. [`mnist_train.ipynb`](./gk/mds/mnist_train.ipynb): đọc thư mục dataset, train và ghi `mnist_weights.json` cùng `digits_models_NxN.json`.
4. [`mnist_test.ipynb`](./gk/mds/mnist_test.ipynb): đọc file weight, chọn index và kiểm tra mẫu qua bốn model.

Notebook train đọc ảnh 28×28 local rồi flatten thành 784 feature ngay trước khi đưa vào model. Dataset và artifact của pipeline nằm riêng trong thư mục `28x28`.

Muốn đổi kiến trúc, activation, learning rate, số vòng lặp hoặc augmentation, sửa [`model_config.py`](./gk/web/backend/model_config.py), chạy lại `mnist_models.ipynb`, rồi chạy `mnist_train.ipynb`. Các notebook sẽ hiển thị log loss, epoch, accuracy, confusion matrix và ghi artifact cho web; không cần chạy lệnh train bằng terminal.

Trang Predict cho phép chọn mẫu trong test set hoặc vẽ tự do trên canvas. `test accuracy` là metric của toàn bộ test set; `predicted digit` và `confidence` là kết quả của mẫu đang hiển thị.

Baseline dùng implementation chính thức [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), cùng `StandardScaler`, train/test split và random seed với các model ANN. Dataset chữ số lấy từ [MNIST handwritten digit database](https://yann.lecun.com/exdb/mnist/). Accuracy hiển thị trong app được đo lại trên test set của project, không lấy từ trang tài liệu.

Người dùng cũng có thể vẽ tự do trên canvas. Backend sẽ khử nhiễu, giữ vùng nét lớn nhất, crop, vuông hóa, resize về `N×N` rồi mới chuẩn hóa và predict. Preview `NORMALIZED INPUT · N×N` là đúng ảnh cuối cùng được đưa vào model.

## Lộ trình thực hiện

1. Đọc `very-simple-liner-model` → `very-simple-ann` → `very-simple-n-neurons` → `very-simple-depth` (1 neuron → rộng → sâu).
2. Đọc `overview.md` để nối sang MLP, phi tuyến và giới hạn của perceptron.
3. Hiểu forward propagation, hidden layer và ranh giới tuyến tính vs phi tuyến.
4. Chạy notebook Iris với ANN tự viết bằng NumPy.
5. Kiểm tra output của từng layer trên một mẫu hoa.
6. Dùng sơ đồ `4 -> 8 -> 3`, loss curve và confusion matrix khi thuyết trình.
7. Chạy tab Train để quan sát training realtime.
8. Chạy tab Predict để so sánh bốn model trên chữ số viết tay thật.
9. Chạy tab Inspect để xem chi tiết forward pass của một mạng.

## Phạm vi hiện tại

Project hiện có notebook NumPy tự xây và web demo React + FastAPI. Tab Train stream từng epoch qua WebSocket; tab Predict dùng weight train offline và tab Inspect cho phép thiết kế, build và xem chi tiết một ANN.

## Nguồn học tập

- [`gk/7-nn1-intro.ppt.pdf`](./gk/7-nn1-intro.ppt.pdf): trực giác sinh học, neuron, bias, activation và topology.
- [`gk/chap4_ann.pdf`](./gk/chap4_ann.pdf): perceptron, XOR, MLP, gradient descent và backpropagation.
- [`gk/lecture-21.pdf`](./gk/lecture-21.pdf): logistic regression, MLP, activation, gradient descent và các lưu ý về deep learning.
- [`gk/mit15_773_s24_lec01.pdf`](./gk/mit15_773_s24_lec01.pdf): hidden layer, activation và ví dụ tính forward pass.
