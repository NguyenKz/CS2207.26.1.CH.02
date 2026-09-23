# Huấn luyện ANN như thế nào?

Huấn luyện ANN là quá trình tìm trọng số và bias để model tạo dự đoán phù hợp với dữ liệu. Phần này nối trực giác của một neuron với ANN tự viết bằng NumPy trong demo Iris.

## 1. Chuẩn bị dữ liệu

Quy trình cơ bản:

1. Có feature matrix và target labels.
2. Chia dữ liệu thành train set và test set.
3. Học tham số model chỉ trên train set.
4. Đánh giá một lần trên test set.

Trong notebook, demo tự chia dữ liệu theo từng lớp bằng NumPy. Mean và standard deviation chỉ được tính từ training set rồi áp dụng cho test set.

`random_state=42` trong phần khởi tạo NumPy chỉ làm cho lần chạy lại có cùng weights ban đầu và cùng cách chia dữ liệu. Đây là tham số tái lập demo, không phải một kết luận thống kê.

## 2. Chuẩn hóa feature

ANN thường nhạy với thang đo của feature. Nếu một feature có giá trị lớn hơn nhiều feature khác, nó có thể ảnh hưởng mạnh đến weighted sum và quá trình tối ưu.

Chuẩn hóa thủ công bằng mean và standard deviation của training set:

```python
training_feature_means = training_features_raw.mean(axis=0)
training_feature_stds = training_features_raw.std(axis=0)
training_features = (
    training_features_raw - training_feature_means
) / training_feature_stds
testing_features = (
    testing_features_raw - training_feature_means
) / training_feature_stds
```

Cách này tránh việc dùng thông tin của test set trong lúc chuẩn bị dữ liệu train.

## 3. Loss function

Loss đo mức độ khác nhau giữa nhãn thật và dự đoán. Với nhiều điểm dữ liệu, ta thường tối ưu một loss trung bình:

```text
J(theta) = (1/n) * sum(loss(y_i, y_hat_i))
```

`theta` đại diện cho toàn bộ trọng số và bias của mạng. Model không cần loss bằng 0 để có ích. Mục tiêu thực tế là tìm tham số giúp dự đoán tốt trên dữ liệu chưa thấy, chứ không chỉ ghi nhớ train set.

Trong `SimpleANN.fit`, code tự tính cross-entropy loss và lưu vào `loss_history` để vẽ loss theo từng epoch.

## 4. Gradient descent

Gradient cho biết loss thay đổi thế nào khi thay đổi một tham số nhỏ. Gradient descent cập nhật tham số theo hướng ngược với gradient:

```text
theta_new = theta_old - learning_rate * gradient
```

Learning rate quá nhỏ có thể làm quá trình học chậm. Learning rate quá lớn có thể làm loss dao động hoặc khó hội tụ. Với demo trên lớp, nên cố định một cấu hình dễ chạy trước, sau đó mới cho người xem thử thay đổi.

## 5. Backpropagation

Backpropagation dùng chain rule để truyền thông tin về sai số từ output layer về các layer trước đó.

Một vòng huấn luyện có thể mô tả bằng các bước:

1. Chạy forward pass để tạo `y_hat`.
2. Tính loss từ target labels và predicted probabilities.
3. Tính gradient ở output layer.
4. Truyền gradient ngược qua hidden layer.
5. Cập nhật trọng số và bias bằng gradient descent.

Backpropagation không phải một loại activation hay một loại model riêng. Nó là cách tính gradient hiệu quả cho các tham số trong mạng nhiều layer.

`chap4_ann.pdf`, slide 16 đến 19, mô tả vấn đề khi không biết error thật của hidden node và cách backpropagation dùng chain rule để tính gradient từ layer sau về layer trước. `lecture-21.pdf`, trang 3 đến 4, trình bày cùng ý tưởng bằng ký hiệu đạo hàm.

## 6. Epoch, batch và stopping condition

- Epoch là một lượt model xử lý dữ liệu train theo cách định nghĩa của thuật toán.
- Batch là nhóm mẫu được dùng trong một lần cập nhật.
- Batch gradient descent dùng toàn bộ dữ liệu cho một lần cập nhật.
- Stochastic gradient descent dùng từng mẫu hoặc nhóm nhỏ hơn.
- `epochs` giới hạn số lần ANN chạy qua toàn bộ training set, không phải số accuracy hay số neuron.
- Training có thể dừng khi đạt giới hạn lặp, khi loss hội tụ hoặc khi điều kiện stopping khác được thỏa mãn.

`chap4_ann.pdf`, slide 17 đến 19, phân biệt gradient descent, stochastic gradient descent và việc lặp cập nhật cho đến khi hội tụ.

## 7. Cấu hình ANN tự viết cho notebook

Đây là cấu hình khởi đầu để minh họa, không phải cấu hình tối ưu cho mọi dữ liệu:

```python
neural_network = SimpleANN(
    input_feature_count=4,
    hidden_neuron_count=8,
    output_class_count=3,
    learning_rate=0.05,
    random_seed=42,
)
neural_network.fit(
    training_features,
    training_one_hot_labels,
    epochs=2000,
)
```

Có thể thử các thay đổi sau trong phần mở rộng:

```python
hidden_neuron_count=4
hidden_neuron_count=8
hidden_neuron_count=16
learning_rate=0.01
learning_rate=0.05
epochs=1000
epochs=2000
```

Khi so sánh, chỉ thay đổi một yếu tố mỗi lần để dễ giải thích ảnh hưởng. Cần ghi lại cấu hình thực tế và kết quả thực tế sau khi chạy, không điền sẵn một accuracy chưa được đo.

## 8. Đánh giá model

Notebook nên báo cáo ít nhất:

```python
testing_predictions = neural_network.predict(testing_features)
testing_accuracy = np.mean(testing_predictions == testing_labels)
print(f"Test accuracy: {testing_accuracy:.3f}")
```

Accuracy chỉ là một chỉ số cho bài toán demo cân bằng hai lớp. Khi dữ liệu mất cân bằng hoặc chi phí của các lỗi khác nhau, cần xem thêm confusion matrix, precision, recall hoặc metric phù hợp.

Đồ thị decision boundary giúp giải thích bằng trực giác. Nó không thay thế việc đánh giá trên test set.

## 9. Overfitting và noise

Mạng có nhiều tham số có thể tạo decision boundary rất uốn lượn để bám vào train set. Khi đó train score có thể cao nhưng kết quả trên dữ liệu mới giảm.

Trong demo, có thể minh họa bằng cách:

- Tăng số neuron hoặc số hidden layer.
- Tăng số epoch.
- Thay đổi learning rate.

Không nên gọi một hình decision boundary là overfitting chỉ bằng mắt. Nếu muốn kết luận, cần so sánh train và test hoặc dùng validation phù hợp.

## 10. Checklist trước khi nói

- Giải thích được một neuron tính `w^T*x + b` rồi áp dụng activation.
- Nói được vì sao có thể dùng một model tuyến tính làm baseline, nhưng baseline không phải phần ANN tự viết.
- Phân biệt forward pass với backpropagation.
- Giải thích learning rate và `max_iter` bằng ngôn ngữ đơn giản.
- Không đọc accuracy như một sự thật độc lập với dataset và cấu hình.
- Nói rõ cấu hình nào là đề xuất và kết quả nào là kết quả thực tế sau khi chạy.
