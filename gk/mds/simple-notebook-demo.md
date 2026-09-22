# Thiết kế notebook demo ANN

File này là blueprint cho notebook. Ở bước hiện tại chưa tạo file `.ipynb`; người triển khai có thể chuyển từng cell dưới đây thành notebook sau khi kiểm tra môi trường Python.

## Mục tiêu notebook

Notebook cần trả lời một câu hỏi đơn giản:

> Khi dữ liệu có ranh giới phi tuyến, hidden layer của MLP giúp gì so với logistic regression?

Notebook không cố chứng minh MLP luôn tốt hơn mọi model. Nó chỉ tạo một thí nghiệm nhỏ, có thể nhìn thấy và giải thích được trong thời gian thuyết trình.

## Cell 1: import thư viện

```python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

Nếu notebook dùng môi trường tối thiểu, chỉ cần `numpy`, `matplotlib` và `scikit-learn`.

## Cell 2: tạo dữ liệu

```python
X, y = make_moons(
    n_samples=400,
    noise=0.20,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

Các giá trị trên là tham số cấu hình đề xuất cho demo. Chúng không phải số liệu lấy từ một nghiên cứu hay kết quả đánh giá.

## Cell 3: vẽ dữ liệu đầu vào

```python
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=35)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("make_moons")
plt.show()
```

Điểm cần nói: dữ liệu chỉ có hai feature nên ta có thể vẽ trực tiếp, không cần giảm chiều để minh họa.

## Cell 4: hàm vẽ decision boundary

```python
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probability = model.predict_proba(grid)[:, 1]
    zz = probability.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(
        xx,
        yy,
        zz,
        levels=np.linspace(0, 1, 11),
        cmap="coolwarm",
        alpha=0.25,
    )
    plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=35)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.show()
```

Đường đen biểu diễn vùng model chuyển từ dự đoán lớp này sang lớp kia. Nền màu thể hiện xác suất dự đoán của model.

## Cell 5: baseline bằng logistic regression

```python
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

print("Logistic regression test accuracy:", accuracy_score(y_test, logistic_pred))
plot_decision_boundary(
    logistic_model,
    X,
    y,
    "Logistic regression decision boundary",
)
```

Điểm cần nói: logistic regression là baseline tuyến tính. Nếu boundary không bám được hình dạng hai cung trăng, đó là giới hạn của model tuyến tính trên dataset này.

## Cell 6: huấn luyện MLP

```python
mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42,
        ),
    ),
])

mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)

print("MLP test accuracy:", accuracy_score(y_test, mlp_pred))
plot_decision_boundary(
    mlp_model,
    X,
    y,
    "MLP decision boundary",
)
```

Không điền sẵn một accuracy kỳ vọng. Khi chạy notebook, ghi lại kết quả thực tế cùng cấu hình và seed đã dùng.

## Cell 7: vẽ loss curve

```python
trained_mlp = mlp_model.named_steps["model"]

plt.figure(figsize=(7, 4))
plt.plot(trained_mlp.loss_curve_)
plt.xlabel("Training iteration")
plt.ylabel("Loss")
plt.title("MLP loss curve")
plt.grid(alpha=0.25)
plt.show()
```

Điểm cần nói: loss curve cho thấy cách objective thay đổi trong quá trình solver cập nhật tham số. Đây không phải đồ thị accuracy và cũng không đảm bảo test performance luôn tăng ở mọi bước.

## Cell 8: thay đổi activation

```python
def train_mlp(activation="tanh", hidden_layer_sizes=(8,)):
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver="adam",
                learning_rate_init=0.01,
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])
    model.fit(X_train, y_train)
    return model


for activation in ["tanh", "relu"]:
    candidate = train_mlp(activation=activation)
    score = accuracy_score(y_test, candidate.predict(X_test))
    print(activation, score)
    plot_decision_boundary(
        candidate,
        X,
        y,
        f"MLP with {activation}",
    )
```

Kết quả của hai activation có thể thay đổi theo dữ liệu, seed và tham số. Phần này dùng để mở câu hỏi, không dùng để tuyên bố activation nào luôn tốt hơn.

## Cell 9: thay đổi kích thước hidden layer

Thử từng cấu hình riêng:

```python
for hidden_layer_sizes in [(2,), (8,), (16, 8)]:
    candidate = train_mlp(
        activation="tanh",
        hidden_layer_sizes=hidden_layer_sizes,
    )
    score = accuracy_score(y_test, candidate.predict(X_test))
    print(hidden_layer_sizes, score)
    plot_decision_boundary(
        candidate,
        X,
        y,
        f"MLP hidden_layer_sizes={hidden_layer_sizes}",
    )
```

Khi trình chiếu, chỉ nên giữ lại một hoặc hai cấu hình để tránh làm người xem mất mạch. Có thể dùng cấu hình nhỏ để nói về underfitting và cấu hình lớn hơn để mở sang overfitting, nhưng chỉ kết luận sau khi so sánh train và test.

## Cell 10: thay đổi noise

Nếu có thời gian, tạo một dataset mới với `noise` khác rồi chạy lại cùng quy trình. Hãy giữ các tham số khác cố định để người xem biết thay đổi nào gây ra khác biệt.

```python
X_noisy, y_noisy = make_moons(
    n_samples=400,
    noise=0.35,
    random_state=42,
)
```

Đây là phần mở rộng. Kịch bản chính nên dùng một dataset và một cấu hình ổn định trước.

## Cell 11: kết luận notebook

Phần Markdown cuối notebook nên trả lời:

- Logistic regression gặp giới hạn gì trên `make_moons`?
- Hidden layer thay đổi decision boundary ra sao?
- Loss curve cho biết điều gì và không cho biết điều gì?
- Kết quả nào là quan sát từ lần chạy hiện tại?
- Tham số nào chỉ là lựa chọn cho demo?
- Vì sao cần đánh giá trên test set?

## Lưu ý khi chạy thật

- Có thể xuất hiện cảnh báo chưa hội tụ nếu thay đổi cấu hình hoặc môi trường. Không được xóa cảnh báo mà không kiểm tra nguyên nhân.
- Nếu `loss_curve_` không tồn tại, kiểm tra solver và phiên bản scikit-learn.
- Nếu kết quả thay đổi, ghi lại phiên bản thư viện, seed và cấu hình.
- Không so sánh hai model nếu chúng dùng cách chia dữ liệu khác nhau.

