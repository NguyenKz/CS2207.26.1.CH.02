# Các hướng làm demo ANN

## Hướng được chọn: phân loại hoa Iris

Demo chính dùng một ANN tự viết bằng NumPy để phân loại một bông hoa vào một trong ba loài Iris:

- `setosa`
- `versicolor`
- `virginica`

Người xem đưa vào bốn số đo của hoa. ANN tự tính forward pass, loss, backpropagation và cập nhật weights để trả về loài hoa dự đoán cùng xác suất tương ứng. `load_iris()` chỉ cung cấp dữ liệu, không cung cấp model đã train.

### Hình minh họa các loài hoa

`load_iris()` chỉ chứa số đo và nhãn, không chứa ảnh chụp. Ba ảnh dưới đây là hình minh họa để người xem hình dung các lớp dữ liệu; model vẫn học từ bốn số đo ở phần tiếp theo.

| `setosa` | `versicolor` | `virginica` |
| --- | --- | --- |
| ![Iris setosa](https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG?width=420) | ![Iris versicolor](https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor_01.jpg?width=420) | ![Iris virginica](https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg?width=420) |
| [Nguồn ảnh](https://commons.wikimedia.org/wiki/File:Iris_setosa.JPG), public domain | [Nguồn ảnh](https://commons.wikimedia.org/wiki/File:Iris_versicolor_01.jpg), CC0 | [Nguồn ảnh](https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg), CC BY-SA 2.0 |

### Input và output

Input của mỗi mẫu gồm bốn feature:

```text
sepal length
sepal width
petal length
petal width
```

Output là một trong ba nhãn:

```text
0 -> setosa
1 -> versicolor
2 -> virginica
```

Đây là bài toán **phân loại đa lớp**, không phải dự đoán một giá trị liên tục.

### ANN gồm những phần nào?

Với cấu hình trong notebook, model có **hai lớp có trọng số**:

| Thành phần | Kích thước | Vai trò |
| --- | ---: | --- |
| Input layer | 4 giá trị | Nhận bốn số đo của một bông hoa. |
| Hidden layer | 8 neuron | Tính biến đổi trung gian bằng `tanh`. |
| Output layer | 3 neuron | Tạo xác suất cho `setosa`, `versicolor`, `virginica` bằng output activation đa lớp (`softmax`). |

Notebook tự tính mean và standard deviation trên training set để chuẩn hóa feature trước khi đưa vào ANN. Bước này là preprocessing, không phải một lớp neuron.

Nếu chỉ đếm các lớp có trọng số, model có **2 lớp**: hidden layer và output layer. Nếu tính cả input layer trong sơ đồ, ta thấy **3 tầng**. Cách nói này giúp tránh nhầm input layer với một lớp cần học tham số.

### Sơ đồ chạy của model

```text
Một bông hoa
    |
    v
[sepal length, sepal width, petal length, petal width]
    | 4 giá trị
    v
Chuẩn hóa bằng mean/std của train set
    |
    v
Hidden layer: 8 neuron, activation = tanh
    | vector kích thước 8
    v
Output layer: 3 neuron, activation = softmax
    |
    v
Chọn xác suất lớn nhất
    |
    v
Tên loài hoa được dự đoán
```

Với một mẫu, kích thước dữ liệu đi qua model như sau:

```text
(1, 4) -> (1, 8) -> (1, 3)
```

Với cả batch, `n` là số mẫu:

```text
(n, 4) -> (n, 8) -> (n, 3)
```

### Forward pass và training diễn ra thế nào?

Khi dự đoán, ANN chỉ chạy theo chiều từ trái sang phải:

```text
input
  -> weighted sum + bias ở hidden layer
  -> tanh
  -> weighted sum + bias ở output layer
  -> xác suất của ba loài
  -> nhãn dự đoán
```

Khi training, model thêm loss và backpropagation:

```text
training_features, training_labels
      |
      v
  forward pass
      |
      v
dự đoán y_hat
      |
      v
loss(training_labels, predicted_probabilities)
      |
      v
backpropagation tính gradient
      |
      v
cập nhật weights và bias
      |
      v
lặp lại qua nhiều iteration
```

Trong phần trình chiếu, nên hiện sơ đồ này trước khi chạy code MLP. Sau đó dùng một mẫu hoa để nối từng mũi tên với output thật của `predict_proba`.

### Vì sao phù hợp?

- Dữ liệu có ý nghĩa thực tế và có sẵn trong `scikit-learn`.
- Input, nhãn và mục tiêu dự đoán dễ giải thích.
- Có thể dùng toàn bộ bốn feature để train ANN.
- Có thể kiểm tra một mẫu hoa mới bằng các số đo cụ thể.
- Có thể hiển thị loss curve và confusion matrix để nối lý thuyết với kết quả.
- Không cần đưa ra accuracy cố định trước khi chạy notebook.

### Kịch bản thuyết trình

1. Giới thiệu câu hỏi: từ số đo của một bông hoa, model dự đoán loài hoa nào?
2. Hiển thị vài mẫu dữ liệu và giải thích bốn feature cùng nhãn đúng.
3. Chia dữ liệu thành training, validation và test set.
4. Khởi tạo ANN 4 -> 8 -> 3 bằng NumPy.
5. Tự chạy forward pass, cross-entropy loss, backpropagation và gradient descent.
6. Nhập hoặc chọn một mẫu hoa trong test set, rồi hiển thị output của từng layer.
7. Hiển thị loss curve để giải thích việc cập nhật trọng số qua các vòng lặp.
8. Hiển thị confusion matrix để xem model thường nhầm loài nào với loài nào.
9. Kết luận giới hạn: dữ liệu nhỏ, kết quả phụ thuộc vào cách chia dữ liệu và cấu hình model.

### Blueprint notebook Iris

#### Cell 1: import thư viện

```python
# NumPy xử lý mảng, phép nhân ma trận và các bước cập nhật gradient thủ công.
import numpy as np
# Matplotlib dùng để vẽ loss curve và confusion matrix.
import matplotlib.pyplot as plt

# scikit-learn chỉ cung cấp dataset. ANN được tự viết ở các cell bên dưới.
from sklearn.datasets import load_iris
```

`scikit-learn` chỉ được dùng để lấy dữ liệu Iris. Không dùng `MLPClassifier`, `Pipeline`, `StandardScaler` hoặc một model đã train sẵn.

#### Cell 2: đọc dữ liệu

```python
# Dataset có bốn số đo cho mỗi bông hoa và một nhãn lớp dạng số nguyên.
iris_dataset = load_iris()
feature_matrix = iris_dataset.data
target_labels = iris_dataset.target

# Dùng random generator riêng để lần chạy lại có cùng cách chia dữ liệu.
random_generator = np.random.default_rng(42)
training_indices_by_class = []
validation_indices_by_class = []
testing_indices_by_class = []

# Giữ 60% train, 20% validation và 20% test của mỗi lớp.
for class_label in np.unique(target_labels):
    class_indices = np.flatnonzero(target_labels == class_label)
    random_generator.shuffle(class_indices)
    training_sample_count = int(0.60 * len(class_indices))
    validation_sample_count = int(0.20 * len(class_indices))
    training_indices_by_class.append(class_indices[:training_sample_count])
    validation_indices_by_class.append(
        class_indices[training_sample_count:training_sample_count + validation_sample_count]
    )
    testing_indices_by_class.append(
        class_indices[training_sample_count + validation_sample_count:]
    )

training_indices = np.concatenate(training_indices_by_class)
validation_indices = np.concatenate(validation_indices_by_class)
testing_indices = np.concatenate(testing_indices_by_class)
random_generator.shuffle(training_indices)
random_generator.shuffle(validation_indices)
random_generator.shuffle(testing_indices)

# Giữ giá trị gốc cho giao diện. Model sẽ nhận bản đã chuẩn hóa bên dưới.
training_features_raw = feature_matrix[training_indices]
training_labels = target_labels[training_indices]
validation_features_raw = feature_matrix[validation_indices]
validation_labels = target_labels[validation_indices]
testing_features_raw = feature_matrix[testing_indices]
testing_labels = target_labels[testing_indices]

# Chỉ tính thống kê từ training set để không làm rò rỉ thông tin từ test set.
training_feature_means = training_features_raw.mean(axis=0)
training_feature_stds = training_features_raw.std(axis=0)
training_features = (training_features_raw - training_feature_means) / training_feature_stds
validation_features = (validation_features_raw - training_feature_means) / training_feature_stds
testing_features = (testing_features_raw - training_feature_means) / training_feature_stds
```

Điểm cần nói: mỗi hàng của `feature_matrix` là bốn số đo của một bông hoa, còn `target_labels` là loài hoa đúng. Validation dùng để theo dõi model trong lúc phát triển; test chỉ dùng ở cuối. Mean và standard deviation chỉ được tính từ training set.

#### Cell 3: chuẩn bị nhãn one-hot

```python
# Đổi class ID thành dạng như [1, 0, 0] để tính cross-entropy loss.
def one_hot_encode(class_labels, number_of_classes):
    encoded_labels = np.zeros((class_labels.size, number_of_classes))
    encoded_labels[np.arange(class_labels.size), class_labels] = 1
    return encoded_labels

training_one_hot_labels = one_hot_encode(training_labels, number_of_classes=3)
validation_one_hot_labels = one_hot_encode(validation_labels, number_of_classes=3)
```

ANN dùng one-hot target để so sánh xác suất dự đoán của ba output neuron với nhãn đúng.

#### Cell 4: tự xây ANN bằng NumPy

```python
# Dịch logits trước khi exp để softmax ổn định hơn với giá trị lớn.
def softmax(logits):
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentiated_logits = np.exp(shifted_logits)
    return exponentiated_logits / exponentiated_logits.sum(axis=1, keepdims=True)


# Đây là mạng 4 -> 8 -> 3: một hidden layer và một output layer.
class SimpleANN:
    def __init__(
        self,
        input_feature_count=4,
        hidden_neuron_count=8,
        output_class_count=3,
        learning_rate=0.05,
        random_seed=42,
    ):
        # Các ma trận này là connection có thể học giữa hai layer liền kề.
        random_generator = np.random.default_rng(random_seed)
        self.input_to_hidden_weights = random_generator.normal(
            0, 0.5, size=(input_feature_count, hidden_neuron_count)
        )
        self.hidden_layer_biases = np.zeros((1, hidden_neuron_count))
        self.hidden_to_output_weights = random_generator.normal(
            0, 0.5, size=(hidden_neuron_count, output_class_count)
        )
        self.output_layer_biases = np.zeros((1, output_class_count))
        self.learning_rate = learning_rate
        # Giao diện vẽ danh sách này để cho thấy training có làm loss giảm không.
        self.loss_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []

    def forward(self, input_features):
        # Hidden layer: bốn input được biến đổi thành tám giá trị trung gian.
        hidden_layer_pre_activation = (
            input_features @ self.input_to_hidden_weights
            + self.hidden_layer_biases
        )
        hidden_layer_output = np.tanh(hidden_layer_pre_activation)
        # Output layer: tám giá trị hidden trở thành điểm số của ba lớp.
        output_layer_pre_activation = (
            hidden_layer_output @ self.hidden_to_output_weights
            + self.output_layer_biases
        )
        output_probabilities = softmax(output_layer_pre_activation)
        # Giữ giá trị trung gian để demo hiện từng layer, không chỉ hiện nhãn cuối.
        forward_cache = {
            "hidden_layer_pre_activation": hidden_layer_pre_activation,
            "hidden_layer_output": hidden_layer_output,
            "output_layer_pre_activation": output_layer_pre_activation,
        }
        return output_probabilities, forward_cache

    def fit(
        self,
        input_features,
        one_hot_targets,
        epochs=2000,
        validation_features=None,
        validation_one_hot_targets=None,
    ):
        training_sample_count = input_features.shape[0]

        # Demo dùng full-batch gradient descent: mỗi lần update nhìn toàn bộ training set.
        for _ in range(epochs):
            # Forward pass tạo xác suất và các giá trị cần cho backpropagation.
            output_probabilities, forward_cache = self.forward(input_features)
            clipped_probabilities = np.clip(output_probabilities, 1e-12, 1.0)
            # Cross-entropy phạt model khi gán xác suất thấp cho lớp đúng.
            loss = -np.mean(
                np.sum(one_hot_targets * np.log(clipped_probabilities), axis=1)
            )
            self.loss_history.append(loss)

            # Với softmax và cross-entropy, đây là tín hiệu lỗi của output layer.
            output_layer_pre_activation_gradients = (
                output_probabilities - one_hot_targets
            ) / training_sample_count
            hidden_to_output_weight_gradients = (
                forward_cache["hidden_layer_output"].T
                @ output_layer_pre_activation_gradients
            )
            output_layer_bias_gradients = output_layer_pre_activation_gradients.sum(
                axis=0, keepdims=True
            )

            # Truyền lỗi qua output weights và đạo hàm của tanh.
            hidden_layer_output_gradients = (
                output_layer_pre_activation_gradients
                @ self.hidden_to_output_weights.T
            )
            hidden_layer_pre_activation_gradients = (
                hidden_layer_output_gradients
                * (1 - forward_cache["hidden_layer_output"] ** 2)
            )
            input_to_hidden_weight_gradients = (
                input_features.T @ hidden_layer_pre_activation_gradients
            )
            hidden_layer_bias_gradients = hidden_layer_pre_activation_gradients.sum(
                axis=0, keepdims=True
            )

            # Di chuyển tham số ngược hướng gradient để loss ở lần sau có xu hướng giảm.
            self.hidden_to_output_weights -= (
                self.learning_rate * hidden_to_output_weight_gradients
            )
            self.output_layer_biases -= (
                self.learning_rate * output_layer_bias_gradients
            )
            self.input_to_hidden_weights -= (
                self.learning_rate * input_to_hidden_weight_gradients
            )
            self.hidden_layer_biases -= (
                self.learning_rate * hidden_layer_bias_gradients
            )

            # Validation chỉ đo khả năng tổng quát hóa, không tạo gradient update.
            if validation_features is not None and validation_one_hot_targets is not None:
                validation_probabilities = self.predict_proba(validation_features)
                clipped_validation_probabilities = np.clip(
                    validation_probabilities, 1e-12, 1.0
                )
                validation_loss = -np.mean(
                    np.sum(
                        validation_one_hot_targets
                        * np.log(clipped_validation_probabilities),
                        axis=1,
                    )
                )
                validation_predictions = validation_probabilities.argmax(axis=1)
                validation_labels = validation_one_hot_targets.argmax(axis=1)
                validation_accuracy = np.mean(
                    validation_predictions == validation_labels
                )
                self.validation_loss_history.append(validation_loss)
                self.validation_accuracy_history.append(validation_accuracy)

        return self

    def predict_proba(self, input_features):
        # Prediction chỉ chạy forward pass, không thay đổi weights.
        output_probabilities, _ = self.forward(input_features)
        return output_probabilities

    def predict(self, input_features):
        return self.predict_proba(input_features).argmax(axis=1)

    def trace(self, input_features):
        # Trace cung cấp dữ liệu cho panel activation của hidden và output layer.
        output_probabilities, forward_cache = self.forward(input_features)
        return {
            "input": input_features,
            "hidden_pre_activation": forward_cache["hidden_layer_pre_activation"],
            "hidden_output": forward_cache["hidden_layer_output"],
            "output_pre_activation": forward_cache["output_layer_pre_activation"],
            "output_probabilities": output_probabilities,
        }


# Train ANN tự viết; validation được đo sau mỗi epoch, test vẫn để dành cho cuối.
neural_network = SimpleANN()
neural_network.fit(
    training_features,
    training_one_hot_labels,
    validation_features=validation_features,
    validation_one_hot_targets=validation_one_hot_labels,
)

validation_predictions = neural_network.predict(validation_features)
validation_accuracy = np.mean(validation_predictions == validation_labels)
testing_predictions = neural_network.predict(testing_features)
testing_accuracy = np.mean(testing_predictions == testing_labels)
print("ANN validation accuracy:", validation_accuracy)
print("ANN test accuracy:", testing_accuracy)
```

Điểm cần nói: toàn bộ weights, bias, activation, loss và cập nhật gradient đều nằm trong code của nhóm. Test set chỉ dùng để kiểm tra sau khi model đã học.

#### Cell 4a: kiểm tra kích thước các lớp

```python
# Các shape này cho thấy số connection và bias trong từng layer.
print(
    "Input -> hidden weights:",
    neural_network.input_to_hidden_weights.shape,
)
print(
    "Hidden -> output weights:",
    neural_network.hidden_to_output_weights.shape,
)
print("Hidden bias:", neural_network.hidden_layer_biases.shape)
print("Output bias:", neural_network.output_layer_biases.shape)
```

Với cấu hình này, kết quả mong đợi về **shape** là:

```text
Input -> hidden weights: (4, 8)
Hidden -> output weights: (8, 3)
Hidden bias: (8,)
Output bias: (3,)
```

Các shape này là cách kiểm tra trực tiếp rằng model có 4 input, 8 neuron ở hidden layer và 3 output.

#### Cell 4b: lấy output của từng lớp

Cell này dùng cùng các weights đã học để in ra giá trị đi qua từng bước của một mẫu. Nhờ vậy người xem không chỉ thấy nhãn cuối cùng.

```python
# Giữ bốn số đo gốc để giao diện hiển thị input mà người dùng hiểu được.
raw_sample_features = testing_features_raw[0:1]
# Áp dụng đúng cách chuẩn hóa từ training set khi training.
sample_features = (
    raw_sample_features - training_feature_means
) / training_feature_stds
# trace() trả về input và output của hidden layer cùng output layer.
forward_pass_trace = neural_network.trace(sample_features)

print("Raw input:", raw_sample_features)
print("Hidden layer input:", forward_pass_trace["input"])
print("Hidden layer output:", forward_pass_trace["hidden_output"])
print("Output layer input:", forward_pass_trace["hidden_output"])
print(
    "Output layer output:",
    forward_pass_trace["output_probabilities"],
)
print(
    "Predicted species:",
    iris_dataset.target_names[
        forward_pass_trace["output_probabilities"].argmax()
    ],
)
```

Kích thước cần hiện trong demo:

```text
Raw input                    (1, 4)
Scaled input                 (1, 4)
Hidden layer output          (1, 8)
Output layer probabilities   (1, 3)
```

#### Cell 5: dự đoán một mẫu hoa

```python
# Dùng lại mẫu này để hiện lớp cuối và xác suất của cả ba lớp.
predicted_class_index = neural_network.predict(sample_features)[0]
class_probabilities = neural_network.predict_proba(sample_features)[0]

print("Predicted species:", iris_dataset.target_names[predicted_class_index])
for name, probability in zip(iris_dataset.target_names, class_probabilities):
    print(f"{name}: {probability:.3f}")
```

Đây là phần giúp người xem thấy rõ input thực tế đi qua ANN và tạo ra output như thế nào.

#### Cell 6: loss curve

```python
# Đường cong giảm nghĩa là objective dùng để update đang nhỏ dần.
plt.figure(figsize=(7, 4))
plt.plot(neural_network.loss_history, label="Training loss")
plt.plot(neural_network.validation_loss_history, label="Validation loss")
plt.xlabel("Training iteration")
plt.ylabel("Loss")
plt.title("ANN training and validation loss")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

Loss curve cho thấy cross-entropy loss trên training và validation thay đổi trong quá trình code tự cập nhật weights. Đây không phải đồ thị accuracy; test set vẫn chỉ dùng sau khi hoàn thành training.

#### Cell 7: confusion matrix

```python
# Hàng là lớp thật, cột là lớp model dự đoán.
confusion_matrix = np.zeros((3, 3), dtype=int)
for actual_class, predicted_class in zip(testing_labels, testing_predictions):
    confusion_matrix[actual_class, predicted_class] += 1

plt.imshow(confusion_matrix, cmap="Blues")
plt.xticks(range(3), iris_dataset.target_names)
plt.yticks(range(3), iris_dataset.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ANN test confusion matrix")
for actual_class in range(3):
    for predicted_class in range(3):
        plt.text(
            predicted_class,
            actual_class,
            confusion_matrix[actual_class, predicted_class],
            ha="center",
            va="center",
        )
plt.show()
```

Đường chéo là số mẫu được phân loại đúng. Các ô ngoài đường chéo cho biết model nhầm giữa hai loài nào.

## Vai trò của `make_moons`

`make_moons` không còn là demo chính. Giữ nó như một phần phụ ngắn để giải thích năng lực biểu diễn của hidden layer:

1. Tạo dữ liệu hai lớp có ranh giới cong.
2. Vẽ Logistic Regression để thấy boundary tuyến tính.
3. Vẽ MLP để minh họa boundary phi tuyến.

Phần này giải thích hình học tốt, nhưng hai lớp `0` và `1` không có ý nghĩa thực tế. Vì vậy không nên dùng nó làm câu chuyện chính của bài thuyết trình.

## Hướng phụ 1: XOR

XOR phù hợp để mở đầu phần lý thuyết:

- Chỉ có bốn điểm nên có thể vẽ và giải thích nhanh.
- Perceptron đơn không tạo được đường thẳng phân tách hai lớp.
- MLP cho thấy hidden layer mở rộng khả năng biểu diễn.

XOR nên là ví dụ lý thuyết ngắn, không phải notebook chính.

## Hướng phụ 2: `make_circles`

`make_circles` tiếp tục minh họa ranh giới phi tuyến dạng vòng. Chỉ thêm phần này nếu còn thời gian sau khi notebook Iris đã chạy ổn định.

## Vì sao không chọn dataset chẩn đoán khối u?

Dataset này có thể dùng cho ANN, nhưng không nên chọn cho demo chính vì:

- Có nhiều feature nên khó minh họa trực tiếp bằng hình.
- Câu chuyện dễ chuyển thành so sánh accuracy thay vì giải thích ANN.
- Nhóm khác đã dùng dataset này cho decision tree.

Iris tạo sự khác biệt rõ hơn mà vẫn giữ được phần dự đoán dễ hiểu.

## Giao diện khi chạy demo

Khi chạy notebook hoặc web app, màn hình chính cần cho người xem theo dõi được một mẫu đi qua model. Không chỉ hiển thị nhãn cuối cùng.

```text
┌──────────────────────────────────────────────────────────────┐
│ Mẫu hoa đầu vào                                              │
│ sepal length | sepal width | petal length | petal width      │
│ [    5.1    ] [    3.5   ] [    1.4    ] [    0.2   ]       │
└──────────────────────────────┬───────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────┐
│ Input layer: 4 giá trị sau chuẩn hóa thủ công                │
│ [x1, x2, x3, x4]                                             │
└──────────────────────────────┬───────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────┐
│ Hidden layer: 8 neuron, tanh                                 │
│ [h1, h2, h3, h4, h5, h6, h7, h8]                             │
└──────────────────────────────┬───────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────┐
│ Output layer: 3 neuron, softmax                              │
│ setosa | versicolor | virginica                              │
│ [  p0  ] [    p1    ] [   p2   ]                              │
└──────────────────────────────┬───────────────────────────────┘
                               v
Kết luận: loài có xác suất lớn nhất
```

### Những gì người xem cần thấy sau khi bấm chạy

- **Input layer:** bốn số đo gốc và bốn số sau khi chuẩn hóa.
- **Hidden layer:** tám giá trị activation sau `tanh`. Có thể hiển thị bằng bảng hoặc bar chart, mỗi cột là một neuron.
- **Output layer:** ba xác suất tương ứng với ba loài. Tô nổi bật xác suất lớn nhất.
- **Kết luận:** tên loài được dự đoán, kèm input mẫu đã dùng.
- **Training:** loss curve và thông tin ANN tự viết sau khi train.

Nút nên đặt tên cụ thể là `Chạy forward pass` hoặc `Dự đoán mẫu này`. Khi bấm nút, app phải cập nhật cả ba khu vực layer. Nếu input chưa đủ hoặc có giá trị không hợp lệ, app báo lỗi ở phần input và không hiển thị output cũ như thể đó là kết quả mới.

## Hướng chuyển thành web app sau notebook

Sau khi notebook Iris chạy ổn định, có thể chuyển cùng logic sang Streamlit. App chỉ cần có:

- bốn ô nhập số đo của hoa;
- nút `Chạy forward pass`;
- panel Input layer, Hidden layer và Output layer;
- tên loài hoa được dự đoán;
- xác suất của ba loài;
- confusion matrix và loss curve của ANN tự viết.

Không cần cho người dùng chỉnh quá nhiều tham số trong bản đầu tiên. Nếu có phần mở rộng, có thể cho chọn Logistic Regression hoặc MLP để so sánh.

### Trạng thái cần dự kiến cho app

- Chưa có input: hướng dẫn nhập đủ bốn số đo.
- Input không hợp lệ: báo rõ giá trị thiếu hoặc không phải số.
- Đang train: hiển thị trạng thái xử lý nếu app train khi người dùng bấm nút.
- Đã dự đoán: hiển thị loài hoa và xác suất.
- Lỗi model hoặc dữ liệu: báo lỗi thay vì hiển thị kết quả giả.

## Thứ tự ưu tiên triển khai

1. Hoàn thành notebook Iris với training, dự đoán mẫu mới, loss curve và confusion matrix.
2. Chuẩn bị một slide ngắn giải thích input, hidden layer và output.
3. Dùng `make_moons` như hình minh họa bổ sung cho ranh giới phi tuyến.
4. Thêm XOR nếu cần mở đầu phần perceptron.
5. Chỉ sau đó mới chuyển notebook thành Streamlit.
6. Không thêm dataset khác nếu chưa có câu hỏi thuyết trình rõ ràng.
