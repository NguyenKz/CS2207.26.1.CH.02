# Tôi cần học gì trước khi làm demo?

Bạn không cần học toàn bộ deep learning để trình bày một demo ANN cơ bản. Cần hiểu một chuỗi ngắn từ neuron đến quá trình huấn luyện.

## Kiến thức nên có trước

- Python cơ bản và cách chạy notebook.
- NumPy array và phép nhân ma trận ở mức đọc được công thức.
- Feature, label, train set và test set.
- Ý nghĩa cơ bản của classification và accuracy.
- Biết vì sao cần tách dữ liệu để đánh giá model trên dữ liệu chưa dùng khi train.

Nếu chưa chắc phần nào, hãy đọc lại ghi chú môn học hiện có trước khi đi vào đạo hàm chi tiết.

## Lộ trình học ANN

### Bước 1: Neuron

Trả lời được:

```text
z = w^T*x + b
a = f(z)
```

- `x` là gì?
- `w` là gì?
- `b` thay đổi điều gì?
- `f` có vai trò gì?

Đọc [`very-simple-liner-model.md`](./very-simple-liner-model.md) rồi [`very-simple-ann.md`](./very-simple-ann.md).

### Bước 2: Perceptron

Hiểu rằng perceptron đơn tạo ranh giới tuyến tính. Dùng XOR để kiểm tra trực giác thay vì học thuộc kết luận.

- Vẽ bốn điểm XOR trên mặt phẳng.
- Thử tưởng tượng một đường thẳng tách hai lớp.
- Giải thích vì sao một đường thẳng không đủ.

### Bước 3: Hidden layer / n neuron

Hiểu hidden layer tạo ra các output trung gian. Những output này được output layer kết hợp để tạo quyết định cuối.

- Một hidden layer có bao nhiêu neuron?
- Một neuron trong hidden layer nhận input từ đâu?
- Nếu bỏ activation phi tuyến thì chuyện gì xảy ra?

Đọc [`very-simple-n-neurons.md`](./very-simple-n-neurons.md), rồi [`overview.md`](./overview.md) và [`how-it-works.md`](./how-it-works.md).

### Bước 4: Forward pass

Tự tính được output của một mạng nhỏ trên một input cụ thể. Không cần bắt đầu bằng mạng sâu. Chỉ cần biết:

1. Tính weighted sum.
2. Cộng bias.
3. Áp dụng activation.
4. Dùng output layer để tạo dự đoán.

### Bước 5: Loss và gradient descent

Hiểu vai trò của loss và ý nghĩa của learning rate. Ở buổi demo cơ bản, không cần tự đạo hàm toàn bộ MLP bằng tay.

- Loss đo điều gì?
- Gradient chỉ hướng thay đổi nào?
- Vì sao cập nhật có dấu trừ?
- Learning rate quá lớn hoặc quá nhỏ có thể gây vấn đề gì?

### Bước 6: Backpropagation

Nói được backpropagation dùng chain rule để tính gradient cho các layer phía trước dựa trên sai số ở output. Không cần trình bày toàn bộ ký hiệu nếu mục tiêu của nhóm là demo trực quan.

Đọc [`how-to-train.md`](./how-to-train.md) và đối chiếu `chap4_ann.pdf`, slide 16 đến 19.

### Bước 7: Chạy notebook

Chạy lần lượt các cell trong [`simple-notebook-demo.md`](./simple-notebook-demo.md). Sau mỗi cell, ghi lại một câu trả lời cho câu hỏi:

> Cell này đang minh họa thành phần nào của ANN?

## Thuật ngữ cần nhớ

| Thuật ngữ | Cách giải thích ngắn |
| --- | --- |
| Neuron | Đơn vị tính weighted sum, bias và activation |
| Weight | Mức ảnh hưởng học được của một input hoặc kết nối |
| Bias | Tham số giúp dịch ngưỡng hoạt động |
| Activation | Hàm biến tổng đầu vào thành output của neuron |
| Phi tuyến | Quan hệ không bị giới hạn bởi một đường thẳng; activation phi tuyến giúp mạng tạo ranh giới cong hoặc phức tạp hơn |
| Ngưỡng hoạt động | Mức để quyết định neuron được xem là hoạt động; step thường dùng điều kiện `z >= 0` |
| Linear | Activation giữ nguyên `z`, có dạng `f(z) = z` và không thêm tính phi tuyến |
| Step | Activation trả về `0` hoặc `1` tùy việc `z` có vượt ngưỡng hay không |
| Sign | Activation trả về `-1` hoặc `1` tùy việc `z` có vượt ngưỡng hay không |
| Sigmoid | Hàm đưa `z` về khoảng `0` đến `1`; `sigmoid(0) = 0.5` |
| Tanh | Hàm đưa `z` về khoảng `-1` đến `1`; `tanh(0) = 0` |
| ReLU | Hàm `max(0, z)`, giữ giá trị dương và đưa giá trị âm về `0` |
| Layer | Nhóm neuron cùng một bước tính toán |
| Hidden layer | Layer nằm giữa input và output |
| Forward pass | Tính dự đoán từ input đi về output |
| Loss | Mức sai khác giữa nhãn và dự đoán |
| Gradient | Thông tin về hướng thay đổi của loss theo tham số |
| Backpropagation | Cách truyền thông tin gradient từ sau về trước |
| Epoch | Một lượt xử lý dữ liệu theo cách định nghĩa của solver |
| Overfitting | Model bám quá sát dữ liệu train và kém trên dữ liệu mới |
| Decision boundary | Ranh giới giữa các vùng dự đoán lớp |

## Câu hỏi tự kiểm tra

Trước khi thuyết trình, hãy thử trả lời không nhìn tài liệu:

1. Vì sao một perceptron chỉ tạo ranh giới tuyến tính?
2. Vì sao XOR là ví dụ kinh điển cho giới hạn đó?
3. Hidden layer có tác dụng gì trong MLP?
4. Activation phi tuyến cần thiết ở đâu?
5. Forward propagation và backpropagation khác nhau thế nào?
6. Vì sao phải chia train và test?
7. Loss curve có phải accuracy curve không?
8. Nếu tăng số neuron, model có chắc chắn tốt hơn không?
9. Vì sao `make_moons` phù hợp để vẽ decision boundary?
10. Khi nào nên dùng dataset thực tế thay cho dataset tổng hợp?

## Checklist chuẩn bị thuyết trình

- [ ] Có thể nói một câu định nghĩa ANN bằng ngôn ngữ của mình.
- [ ] Có thể tính một neuron bằng tay.
- [ ] Có thể giải thích XOR bằng hình vẽ.
- [ ] Có thể nói vì sao logistic regression là baseline.
- [ ] Đã chạy notebook với cùng seed và ghi lại cấu hình.
- [ ] Đã kiểm tra decision boundary và loss curve.
- [ ] Biết accuracy thực tế của lần chạy, nếu muốn trình bày accuracy.
- [ ] Không đọc một accuracy trong tài liệu như một kết quả chắc chắn cho mọi môi trường.
- [ ] Có câu trả lời nếu người nghe hỏi về overfitting.
- [ ] Có câu trả lời nếu người nghe hỏi vì sao không dùng dataset chẩn đoán khối u làm demo chính.
