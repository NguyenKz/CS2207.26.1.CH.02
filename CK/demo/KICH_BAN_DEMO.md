# Kịch bản demo — Simplicial SMOTE

> **Một file duy nhất.** Phần A = đọc trên sân khấu. Phần B = tra cứu khi giáo viên hỏi.
>
> Bạn **demo app**, không thuyết lại slide. Tên method đúng paper: Imbalanced, Gaussian, Random, Global, SMOTE, Simplicial SMOTE.

---

# Phần A — Kịch bản lời thoại để đọc

**Thời lượng dự kiến:** khoảng 5–7 phút.  
**Dataset chính:** Gaussian in circle.  
**Thứ tự thao tác:** Imbalanced → Global → Gaussian → SMOTE → Simplicial SMOTE.

## A1. Chuẩn bị trước khi nói

1. Người trình bày chạy `./run.sh` hoặc `./share.sh` trong thư mục `CK/demo`.
2. Người trình bày mở landing page và chọn dataset **Gaussian in circle**.
3. Người trình bày không cần thay đổi seed, k hoặc p trong lần demo đầu tiên. Cấu hình mặc định là seed 2207, k = 5 và p = 3.
4. Người trình bày nên dừng animation ở một mẫu Simplicial có từ ba đỉnh trở lên để người xem nhìn thấy một tam giác.

## A2. Lời mở đầu

> “Xin chào thầy và các bạn. Trong phần này, em sẽ minh họa trực tiếp cách các phương pháp oversampling tạo thêm dữ liệu cho lớp thiểu số. Em sẽ tập trung vào điểm khác nhau về hình học giữa SMOTE và Simplicial SMOTE: SMOTE sinh điểm trên một đoạn thẳng, còn Simplicial SMOTE có thể sinh điểm bên trong một vùng simplex, ví dụ như một tam giác.
>
> Phần demo này không thay thế phần lý thuyết và phần thí nghiệm trong slide. Em chỉ dùng canvas để mọi người nhìn thấy dữ liệu được tạo ra như thế nào.”

## A3. Giới thiệu dataset trên landing page

**Thao tác:** Người trình bày đang ở landing page và chỉ vào các thẻ dataset.

> “Trên landing page, ứng dụng giới thiệu bốn bộ dữ liệu synthetic hai chiều được xây dựng theo nhóm dữ liệu synthetic trong paper: Two moons, Swiss roll, Gaussian in circle và Circle in circle.
>
> Các bộ này không phải dữ liệu giao dịch thật, cũng không phải bốn bộ dữ liệu UCI. Chúng được tạo ra để mô phỏng những hình dạng khác nhau của dữ liệu và để quan sát xem một phương pháp có giữ được cấu trúc hình học của lớp thiểu số hay không.
>
> Trong demo hôm nay, em chọn Gaussian in circle. Theo kết quả trong Table 1 của paper, đây là bộ mà Simplicial SMOTE đạt kết quả cao nhất trong bốn bộ synthetic và cao hơn SMOTE. Vì vậy, bộ này phù hợp để minh họa sự khác nhau giữa việc sinh điểm trên cạnh và sinh điểm bên trong simplex.”

**Thao tác:** Người trình bày chọn **Gaussian in circle**.

> “Trong bộ dữ liệu này, lớp thiểu số là một đám điểm theo phân phối Gaussian ở giữa. Lớp đa số nằm trên một vành tròn bao quanh đám điểm đó.
>
> Ở tập train, n⁺ bằng 50 là số điểm thiểu số, còn n⁻ bằng 300 là số điểm đa số. Vì vậy, tỉ lệ mất cân bằng IR bằng n⁻ chia cho n⁺, tức là 300 chia 50 bằng 6. Nếu muốn đưa hai lớp về cùng số lượng, ứng dụng cần sinh thêm m bằng n⁻ trừ n⁺, tức là 250 điểm synthetic cho lớp thiểu số.
>
> Trong code của ứng dụng, Gaussian in circle được sinh bằng NumPy. Phần thiểu số dùng `random.normal` để tạo đám Gaussian. Phần đa số dùng hàm `_disk` tự viết để tạo vành tròn. Riêng Swiss roll mới gọi hàm `make_swiss_roll` của scikit-learn.”

## A4. Cách app train và đánh giá model

**Thao tác:** Người trình bày giải thích luồng dữ liệu trước khi chạy các method.

> “Trước khi so sánh các method, em sẽ giải thích cách app huấn luyện và đánh giá model. App **không** dùng cùng một tập dữ liệu cho cả việc train và test. Nếu model đã nhìn thấy những điểm dùng để chấm điểm, kết quả sẽ quá lạc quan và không phản ánh khả năng dự đoán dữ liệu mới.
>
> App chia dữ liệu thành hai phần. Tập train được dùng để học, còn tập test được giữ riêng để kiểm tra sau cùng. Với dataset hôm nay, tập train có 50 điểm minority và 300 điểm majority. Tập test độc lập có 30 điểm minority và 120 điểm majority.
>
> Đầu tiên, app chỉ học cách chuẩn hóa từ tập train. Việc chuẩn hóa đưa hai trục về thang đo phù hợp để tính khoảng cách. Sau đó, app áp dụng đúng cách chuẩn hóa đó cho tập test. App không dùng test để tính trung bình hoặc độ lệch chuẩn, vì làm như vậy sẽ đưa thông tin test vào quá trình huấn luyện.
>
> Tiếp theo, app chỉ oversample tập train. Với full budget, app sinh thêm 250 điểm minority synthetic để số minority trong train tăng từ 50 lên 300, bằng số majority. Tập test vẫn giữ nguyên 30 điểm minority và 120 điểm majority; app không sinh thêm điểm synthetic trên test.
>
> Sau khi oversampling hoàn tất, app huấn luyện `KNeighborsClassifier(n_neighbors=5)` trên tập train mới. Vì vậy, demo hiện tại dùng KNN, không dùng Logistic Regression. KNN dự đoán một điểm mới dựa trên nhãn của những điểm train gần nó nhất.”

### A4.1. App tính metric như thế nào?

> “Sau khi model học xong, app đưa từng điểm trong tập test vào model để model dự đoán nhãn. Model chưa từng dùng những điểm test này để học.
>
> App so sánh nhãn dự đoán với nhãn thật của từng điểm test. Nếu một điểm minority được dự đoán đúng là minority, đó là true positive, viết tắt là TP. Nếu một điểm majority được dự đoán đúng là majority, đó là true negative, viết tắt là TN. Nếu majority bị dự đoán nhầm thành minority, đó là false positive, viết tắt là FP. Nếu minority bị dự đoán nhầm thành majority, đó là false negative, viết tắt là FN.
>
> Từ bốn loại kết quả này, app tính các chỉ số. Recall cho biết trong tất cả minority thật, model bắt được bao nhiêu phần trăm. Precision cho biết trong những điểm model dự đoán là minority, có bao nhiêu điểm thật sự là minority. F1 là điểm cân bằng giữa recall và precision. MCC tổng hợp cả TP, TN, FP và FN, nên hữu ích khi số lượng hai lớp không bằng nhau.
>
> Nói ngắn gọn, app đi theo chuỗi: dùng train để chuẩn hóa và huấn luyện, dùng test chưa từng được nhìn thấy để dự đoán, rồi so sánh dự đoán với nhãn thật để tính metric.”

### A4.2. App so sánh các method như thế nào?

> “Để so sánh công bằng, em giữ nguyên dataset, seed, cách chia train/test, số điểm synthetic cần sinh và classifier KNN. Mỗi lần em chỉ thay đổi cách tạo điểm minority synthetic: Imbalanced không tạo điểm, Random sao chép điểm, Global trộn một cặp bất kỳ, Gaussian lấy mẫu theo phân phối, SMOTE nội suy trên cạnh và Simplicial SMOTE nội suy trong simplex.
>
> Sau đó, app huấn luyện lại một model cho từng method và đánh giá tất cả model trên cùng tập test. Nhờ vậy, sự khác nhau về metric và hình dạng điểm trên canvas chủ yếu đến từ method oversampling, chứ không phải do đổi dataset hoặc đổi tập test.
>
> Cách này phù hợp để minh họa trong demo. Tuy nhiên, đây vẫn là một holdout cố định với một seed và một lần chạy. Nó không thay thế nested cross-validation, grid search nhiều giá trị k và p, hay thí nghiệm trên 21 dataset như phần Experiments của slide và paper.”

## A4. Giới thiệu canvas và các thành phần giao diện

**Thao tác:** Người trình bày chờ canvas tải xong, chưa chọn method khác.

> “Bây giờ em sẽ giới thiệu nhanh cách đọc giao diện trước khi chạy từng phương pháp.
>
> Ở bên trái là canvas chính. Canvas hiển thị vị trí của các điểm trên hai trục x và y, đồng thời hiển thị nền màu nhạt thể hiện vùng quyết định của classifier. Các điểm xám là lớp đa số trong tập train. Các điểm đỏ là lớp thiểu số trong tập train. Các điểm cam có kích thước lớn hơn là các điểm minority trong tập test. Những điểm xanh lá xuất hiện sau đó là các điểm synthetic do phương pháp oversampling tạo ra.
>
> Ở bên phải là bảng điều khiển. Phần đầu bảng hiển thị tên dataset, trạng thái chạy và nút Run. Bên dưới là legend, giúp em xác định màu của majority, minority, test và synthetic.
>
> Tiếp theo là danh sách method. Các tên Imbalanced, Gaussian, Random, Global, SMOTE và Simplicial SMOTE là tên được dùng trong paper và trong slide. Em sẽ bấm từng method để so sánh cách chúng tạo dữ liệu.
>
> Nhóm nút điều khiển cho phép em phóng to, thu nhỏ, đưa camera về khung ban đầu bằng Fit, tập trung vào minority bằng Focus, chuyển về mẫu trước hoặc mẫu sau, khởi động lại animation và điều chỉnh tốc độ. Trong lúc demo, em chủ yếu dùng Play, Stop, Next và Reset.
>
> Phần STEPS mô tả từng bước của mẫu synthetic hiện tại. Ví dụ, phần này có thể cho biết ứng dụng đang chọn parent, nối các điểm hàng xóm, tính trọng số hoặc commit điểm mới.
>
> Phần OUTCOME ở cuối bảng hiển thị kết quả trên tập test, gồm detection, recall, precision, F1, MCC, FN và FP. Những con số này giúp so sánh lần chạy hiện tại của app, nhưng không phải là kết quả trung bình trong Table 1 của paper.”

## A5. Chạy method Imbalanced

**Thao tác:** Người trình bày bấm **Imbalanced** và chờ kết quả tải xong.

> “Trước tiên, em chọn Imbalanced để làm mốc so sánh. Method này không sinh thêm điểm nào. Vì vậy, canvas chỉ hiển thị dữ liệu gốc với 50 điểm minority màu đỏ và 300 điểm majority màu xám.
>
> Em có thể thấy minority tạo thành một đám ở giữa, còn majority tạo thành một vành ở bên ngoài. Đây là tình trạng dữ liệu mất cân bằng trước khi oversampling. Classifier chỉ học từ tập train chưa được bổ sung điểm synthetic.”

**Thao tác:** Người trình bày chỉ vào canvas, legend và OUTCOME.

> “Ở trạng thái này, không có điểm xanh lá vì chưa có synthetic sample. Các điểm cam thuộc test và không được đưa vào quá trình sinh mẫu.”

## A6. Chạy method Global

**Thao tác:** Người trình bày bấm **Global**, sau đó bấm Play hoặc Next một vài bước.

> “Tiếp theo, em chọn Global. Global chọn hai điểm minority bất kỳ trong toàn bộ tập X⁺ rồi tạo điểm mới trên đoạn thẳng nối hai điểm đó. Hai điểm được chọn không bắt buộc phải là hàng xóm của nhau.
>
> Vì Global không dùng k-NN graph cục bộ, canvas không hiển thị web hàng xóm màu xám. Canvas chỉ hiển thị cặp điểm đang được chọn và quá trình nội suy giữa cặp đó.
>
> Cách chọn toàn cục này có thể tạo điểm ở những vùng trống giữa các cụm hoặc gần vùng majority. Đây là lý do paper phân biệt global sampling với các phương pháp giữ cấu trúc cục bộ.”

## A7. Chạy method Gaussian

**Thao tác:** Người trình bày bấm **Gaussian** và chờ ellipse xuất hiện.

> “Bây giờ em chọn Gaussian. Method này ước lượng trung tâm và độ phân tán của các điểm minority, sau đó lấy mẫu mới theo phân phối Gaussian đã ước lượng.
>
> Ellipse màu xanh nhạt trên canvas biểu diễn vùng hiệp phương sai của minority. Gaussian không xây dựng k-NN graph nên không biết mối quan hệ hàng xóm giữa từng điểm. Nó cũng không biết rằng bên ngoài đám minority có một vành majority, vì vậy đây là một baseline theo phân phối chứ không phải theo topology.”

## A8. Chạy method SMOTE

**Thao tác:** Người trình bày bấm **SMOTE**, chọn tốc độ chậm hoặc Stop, rồi dùng Next từng bước.

> “Tiếp theo, em chọn SMOTE. SMOTE chỉ xây dựng đồ thị k-NN trên các điểm minority. Với cấu hình hiện tại, mỗi điểm tìm 5 hàng xóm gần nhất. Canvas biểu diễn các quan hệ đó bằng những cạnh màu xám.
>
> Để tạo một điểm mới, SMOTE chọn một điểm minority và một hàng xóm của nó. Sau đó, thuật toán chọn một tỉ lệ trộn nằm giữa hai điểm và đặt điểm mới lên đoạn thẳng nối hai điểm đó.
>
> Vì SMOTE chỉ sử dụng hai điểm tại một thời điểm, mô hình hình học của SMOTE là một tập các đoạn thẳng. Trên canvas, em có thể thấy hai parent được highlight và điểm synthetic nằm trên cạnh được chọn.”

## A9. Chạy method Simplicial SMOTE

**Thao tác:** Người trình bày bấm **Simplicial SMOTE**, dùng Stop hoặc tốc độ chậm, rồi Next đến một mẫu có ít nhất ba vertices.

> “Cuối cùng, em chọn Simplicial SMOTE. Đây là phần chính của demo.
>
> Simplicial SMOTE vẫn bắt đầu từ các điểm minority và quan hệ hàng xóm cục bộ. Tuy nhiên, thuật toán không chỉ chọn một cặp điểm. Thuật toán tìm một nhóm điểm mà mọi cặp trong nhóm đều có quan hệ hàng xóm. Nhóm đó được gọi là clique trong đồ thị, và được xem như một simplex trong mô hình hình học.
>
> Khi nhóm có hai điểm, simplex là một đoạn thẳng. Khi nhóm có ba điểm, simplex là một tam giác. Canvas đang highlight nhóm được chọn và tô vùng simplex để em nhìn thấy vùng mà điểm mới có thể xuất hiện.
>
> Sau khi chọn simplex, thuật toán tạo các trọng số λ cho các đỉnh. Mỗi trọng số không âm và tổng các trọng số bằng 1. Điểm synthetic được tính bằng trung bình có trọng số của các đỉnh. Vì vậy, điểm mới nằm bên trong convex hull của nhóm, thay vì chỉ nằm trên một cạnh.
>
> Nếu giới hạn p bằng 1, thuật toán chỉ sử dụng các simplex có hai đỉnh, nên cách sinh trở thành SMOTE. Khi p lớn hơn 1 và đồ thị có đủ clique, thuật toán có thể sinh điểm bên trong tam giác hoặc simplex có chiều cao hơn. Trong cấu hình demo này, p được đặt là 3 để có cơ hội quan sát nhóm nhiều đỉnh.”

**Thao tác:** Người trình bày chỉ lần lượt vào parent points, vùng simplex, trọng số và điểm xanh.

> “Ở bước này, em muốn mọi người chú ý bốn thành phần. Các điểm đỏ là những parent minority. Các cạnh xám cho biết quan hệ k-NN. Vùng được tô là simplex đang được chọn. Điểm xanh là kết quả của phép trộn các parent bằng các trọng số λ.
>
> Đây là khác biệt trực quan quan trọng nhất: SMOTE sinh trên cạnh, còn Simplicial SMOTE có thể sinh trong vùng được tạo bởi nhiều điểm hàng xóm.”

## A10. Kết luận và giải thích metric

**Thao tác:** Người trình bày để animation dừng ở simplex đang highlight và chỉ vào OUTCOME.

> “Qua năm method, em có thể tóm tắt như sau. Imbalanced giữ nguyên tình trạng lệch lớp. Random chỉ sao chép những điểm minority đã có. Global trộn một cặp minority bất kỳ trên toàn bộ lớp. Gaussian bốc điểm theo một phân phối đã ước lượng. SMOTE trộn hai điểm minority gần nhau trên một cạnh. Simplicial SMOTE trộn nhiều điểm minority tạo thành một simplex cục bộ.
>
> Các chỉ số ở OUTCOME được tính trên tập test cố định của ứng dụng. App không dùng cùng một tập vừa để train vừa để test. App chỉ dùng train để chuẩn hóa, oversample và huấn luyện KNN; sau đó app dùng test độc lập để dự đoán và tính metric. Recall cho biết ứng dụng bắt được bao nhiêu phần trăm minority. Precision cho biết trong các điểm bị dự đoán là minority thì có bao nhiêu điểm đúng. F1 cân bằng recall và precision. MCC tổng hợp cả bốn loại kết quả đúng và sai. FN là số minority bị bỏ sót, còn FP là số majority bị báo nhầm.
>
> Em nhấn mạnh rằng đây là metric của một lần chạy demo với một holdout cố định và một KNN classifier. Đây không phải quy trình nested cross-validation trên 21 dataset trong phần Experiments của slide, nên các con số trên app không cần trùng với Table 1 của paper.
>
> Kết luận của phần demo là: SMOTE mở rộng minority bằng các đoạn nối hàng xóm, còn Simplicial SMOTE mở rộng bằng các vùng simplex cục bộ. Canvas giúp nhìn trực tiếp sự khác nhau đó.”

## A11. Cheat-sheet thao tác

| Bước | Thao tác | Câu chính cần nói |
|------|----------|-------------------|
| 1 | Chọn Gaussian in circle | Dataset có đốm Gaussian minority ở giữa và vành majority bên ngoài. |
| 2 | Giới thiệu canvas và panel phải | Canvas hiển thị dữ liệu; panel phải chứa method, điều khiển, STEPS và OUTCOME. |
| 3 | Chọn Imbalanced | Đây là dữ liệu gốc, chưa sinh synthetic. |
| 4 | Chọn Global | Global trộn một cặp minority bất kỳ, không dùng web k-NN. |
| 5 | Chọn Gaussian | Gaussian lấy mẫu theo phân phối minority và hiển thị ellipse. |
| 6 | Chọn SMOTE | SMOTE sinh điểm trên cạnh nối hai hàng xóm. |
| 7 | Chọn Simplicial SMOTE | Simplicial SMOTE sinh điểm trong simplex tạo bởi nhiều hàng xóm. |
| 8 | Chỉ vào OUTCOME | Số liệu live của app khác với kết quả trung bình trong paper. |

## A12. Nếu hết thời gian

> “Em đang minh họa một bộ Gaussian in circle có 50 điểm minority và 300 điểm majority. SMOTE sinh điểm trên các cạnh của đồ thị hàng xóm, còn Simplicial SMOTE có thể sinh điểm bên trong tam giác hoặc simplex cục bộ. Đây là demo trực quan, nên metric trên app chỉ là kết quả của một lần chạy cố định và không thay thế thí nghiệm trong paper.”

---

# Phần B — Chi tiết / tra cứu (khi giáo viên hỏi)

## 0. Các ký hiệu cần nhớ (n⁺, n⁻, …)

Bài toán có **hai lớp**. Paper gọi lớp hiếm là “positive / thiểu số”, lớp nhiều là “negative / đa số”. Dấu `+` / `−` chỉ là quy ước — **không** có nghĩa tốt/xấu.

### Giải thích từng cái

**n⁺ (đọc: n-plus)**  
= số điểm thuộc lớp **thiểu số** (ít hơn).  
Ví dụ: số giao dịch gian lận, số bệnh nhân mắc bệnh hiếm, …  
Trên canvas: điểm **đỏ** lúc train.  
Demo Gaussian in circle: **n⁺ = 50**.

**n⁻ (đọc: n-minus)**  
= số điểm thuộc lớp **đa số** (nhiều hơn).  
Trên canvas: điểm **xám**.  
Demo: **n⁻ = 300**.

**Vì sao gọi là mất cân bằng?**  
Vì thiểu số ít hơn đa số rất nhiều: n⁺ nhỏ hơn hẳn n⁻. Nhìn canvas sẽ thấy đỏ thưa, xám dày.

**IR (imbalance ratio — tỉ lệ mất cân bằng)**  
= “đa số gấp thiểu số bao nhiêu lần”  
= n⁻ chia cho n⁺.  
Ví dụ: 300 / 50 = **6** → đa số gấp thiểu số 6 lần.

**m (số điểm giả cần sinh)**  
Oversampling muốn **cân số lượng**: thiểu số sau cùng cũng có bằng đa số.  
Công thức: **m = n⁻ − n⁺**.  
Ví dụ: 300 − 50 = **250**.  
Sau khi thêm 250 điểm xanh: thiểu số = 50 thật + 250 giả = 300 = n⁻.

**X⁺**  
= tập các điểm thiểu số (các chấm đỏ dùng để học cách sinh điểm mới).  
Thuật toán SMOTE / Simplicial **chỉ nhìn X⁺**, không “vẽ thêm” điểm đa số.

**X⁻**  
= tập các điểm đa số (xám). Khi oversample thường **giữ nguyên**, không xóa, không nhân bản.

**x̂ (điểm synthetic / điểm giả)**  
= điểm mới do thuật toán tạo ra, vẫn gắn nhãn thiểu số.  
Trên canvas: điểm **xanh lá**.

**Train và test**  
- **Train:** dữ liệu dùng để chuẩn hóa, sinh điểm giả, huấn luyện classifier.  
- **Test:** dữ liệu để chấm điểm — **không** oversample.  
Trên canvas, thiểu số ở test tô **cam** (khác đỏ train).

**n⁺_test, n⁻_test**  
= số thiểu số / đa số trên tập test của demo.  
Demo: **30** thiểu số test, **120** đa số test.  
Đừng nhầm với n⁺, n⁻ của train.

### Nhớ nhanh

| Muốn nói | Nhớ |
|----------|-----|
| Thiểu số cần “bơm thêm” | dấu **+** → n⁺, X⁺, điểm đỏ / xanh |
| Đa số làm mốc số lượng | dấu **−** → n⁻, X⁻, điểm xám |
| Cần sinh bao nhiêu điểm giả? | m = n⁻ − n⁺ |
| Lệch bao nhiêu lần? | IR = n⁻ / n⁺ |

### Số liệu demo (Gaussian in circle)

```text
Train:  n⁺ = 50,  n⁻ = 300,  IR = 6,  m = 250
Test:   n⁺_test = 30,  n⁻_test = 120
```

---

## 1. Phạm vi

| Làm | Không làm |
|-----|-----------|
| Mở landing → chọn 1 dataset → so method trên canvas | Thuyết trình lại toàn bộ slide |
| Cho thấy Simplicial sinh điểm **trong tam giác/simplex**, khác SMOTE (trên cạnh) và Global (cặp bất kỳ) | Coi số F1 trên app = số Table 1 của paper |
| Giải thích n⁺, n⁻, m, IR nếu bị hỏi | Demo hết cả 4 dataset |
| Dùng phần Q&A bên dưới khi giáo viên hỏi sâu | Nhận demo này = tái hiện đủ thí nghiệm paper |

---

## 2. Chuẩn bị

```bash
cd CK/demo
./run.sh          # máy mình: giao diện cổng 5500, API cổng 8000
# hoặc
./share.sh        # chia sẻ qua ngrok (giữ cửa sổ terminal chạy)
```

- Vào thẳng canvas: `http://127.0.0.1:5500/demo.html?dataset=gaussian_circle`  
  hoặc từ trang landing chọn **Gaussian in circle**.
- Tham số mặc định app: seed = 2207, k = 5, p = 3 (đủ để thấy được tam giác, không chỉ đoạn thẳng).
- **Ngrok:** tắt `./share.sh` thì link public chết ngay.

---

## 4. Giải thích dataset (tiếng Việt dễ hiểu)

### 4.1 Vì sao phải oversample? (nếu giáo viên hỏi)

- Máy học trên dữ liệu lệch dễ **thiên về lớp đông** (xám nhiều).
- Độ chính xác tổng (accuracy) dễ đánh lừa: cứ đoán “đa số” mãi cũng ra số đẹp, nhưng **bắt trượt** hết thiểu số.
- **Oversample:** giữ nguyên điểm đa số, **thêm điểm giả** cho thiểu số cho gần bằng số lượng → máy có nhiều ví dụ thiểu số hơn để học.  
  Đổi lại: nếu sinh điểm giả kém chất lượng thì có thể thêm nhiễu.

### 4.2 Vì sao hôm nay chọn Gaussian in circle?

Trong paper có bảng F1 trên 4 bộ vẽ sẵn (Table 1). Tóm lại:

| Bộ dữ liệu | Simplicial so với SMOTE (theo paper) | Dùng để demo? |
|------------|--------------------------------------|---------------|
| moons | **Hòa** nhau (cùng điểm cao) | Không — khó thấy “Simplicial hơn” |
| swiss_rolls | **SMOTE cao hơn** | Không — dễ phản tác dụng |
| **g_circle** (Gaussian in circle) | **Simplicial cao nhất, hơn SMOTE rõ** | **Có — chọn bộ này** |
| circles | Simplicial cũng nhất nhưng cách biệt nhỏ hơn | Chỉ nhắc miệng |

Paper còn nói: khi hình dạng dữ liệu phức tạp, cách **Global / Gaussian** thường sinh điểm kém hơn cách **local** (SMOTE / Simplicial).

Số F1 chi tiết (copy từ paper — **không** phải số trên app):

| Dataset | Imbalanced | Gaussian | Random | Global | SMOTE | Simplicial |
|---------|------------|----------|--------|--------|-------|------------|
| moons | 0.9511 | 0.8830 | 0.9485 | 0.9348 | **0.9694** | **0.9694** |
| swiss_rolls | 0.5317 | 0.6673 | 0.7168 | 0.6774 | **0.7208** | 0.6823 |
| **g_circle** | 0.7129 | 0.6750 | 0.7089 | 0.6542 | 0.6937 | **0.7269** |
| circles | 0.6541 | 0.7060 | 0.6777 | 0.6356 | 0.7005 | **0.7139** |

### 4.3 Dataset lấy từ đâu? Sinh bằng thư viện gì? (có đoạn code để show)

**Nói rõ 3 tầng (đừng gộp làm một):**

1. **Paper §4.1:** bốn bộ synthetic (moons, swiss rolls, Gaussian trong vòng, circle-in-circle), mỗi bộ khoảng 50 thiểu số + 300 đa số. Paper viết là tạo **một phần** bằng **scikit-learn**, phần còn lại là hình học tự thiết kế — paper **không** phát hành sẵn file CSV cho bốn bộ này trong demo của mình.
2. **App của nhóm:** tự sinh lại cùng ý tưởng trong file `CK/demo/backend/server.py`, chủ yếu bằng **NumPy**. Chỉ **Swiss roll** gọi thẳng hàm có sẵn của scikit-learn.
3. **Không phải:** dữ liệu ngân hàng thật, cũng không phải 21 bộ UCI trong phần Experiments slide.

**Bảng “bộ nào — thư viện nào” (học thuộc):**

| Bộ | Gọi thư viện có sẵn? | Thực tế trong app |
|----|----------------------|-------------------|
| Gaussian in circle | **Không** dùng `sklearn.datasets` | **NumPy**: `normal` (đốm giữa) + hàm `_disk` tự viết (vòng ngoài) |
| Two moons | **Không** dùng `make_moons` | **NumPy**: hàm `_moon_arc` tự viết |
| Circle in circle | **Không** dùng `make_circles` | **NumPy**: `_disk` (đĩa trong / vành ngoài) |
| Swiss roll | **Có** — `sklearn.datasets.make_swiss_roll` | sklearn sinh cuộn 3D → lấy 2 trục; đa số = NumPy `uniform` |

**Câu nói miệng (rõ ràng):**  
“Paper bảo synthetic dựa một phần scikit-learn. Trên app của tụi em: chỉ Swiss roll gọi `make_swiss_roll` của sklearn; còn bộ demo chính Gaussian-in-circle, cũng như moons và circles, em sinh bằng NumPy — phân phối chuẩn và hình đĩa/vành — không gọi `make_moons` / `make_circles`.”

---

#### Đoạn code có thể bật lên demo (rút từ `server.py`)

**A) Import — thấy sklearn chỉ lấy Swiss roll**

```python
import numpy as np
from sklearn.datasets import make_swiss_roll  # chỉ dùng cho swiss roll
```

**B) Bộ demo chính — Gaussian in circle (thuần NumPy)**

Thiểu số = phân phối chuẩn quanh gốc; đa số = vòng ngoài (`_disk`).

```python
def make_gaussian_circle_data(seed: int = 2207):
    random = np.random.default_rng(seed)
    # Đa số: vành tròn (tự viết bằng NumPy)
    majority_train = _disk(random, n=300, radius=1.55, thickness=0.14)
    # Thiểu số: đốm Gaussian ở giữa (NumPy normal)
    minority_train = np.column_stack((
        random.normal(0.0, 0.28, 50),
        random.normal(0.0, 0.28, 50),
    ))
    # ... test tương tự, rồi ghép nhãn
```

Hàm vành/đĩa (cũng NumPy, không sklearn):

```python
def _disk(random, n, radius, thickness=0.0):
    if thickness <= 0:
        # đĩa đặc: bán kính ngẫu nhiên trong hình tròn
        r = radius * np.sqrt(random.uniform(0.0, 1.0, n))
    else:
        # vành: bán kính ≈ radius ± thickness
        r = radius + random.uniform(-thickness, thickness, n)
    angles = random.uniform(0.0, 2 * np.pi, n)
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))
```

**Khi show code B trên sân khấu, nói:**  
“Đây là chỗ sinh bộ Gaussian in circle: không có `sklearn.datasets`, chỉ `numpy.random`.”

**C) Swiss roll — chỗ thật sự gọi scikit-learn**

```python
def _swiss_2d(n, noise, seed):
    points, _ = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    return points[:, [0, 2]]  # chiếu 3D → 2D (x, z)

def make_swiss_roll_data(seed=2207):
    random = np.random.default_rng(seed)
    minority_train = _swiss_2d(50, noise=0.35, seed=seed)          # sklearn
    # Đa số: NumPy bốc đều trong khung bao quanh minority
    lo = minority_train.min(axis=0) - 1.5
    hi = minority_train.max(axis=0) + 1.5
    majority_train = random.uniform(lo, hi, size=(300, 2))          # NumPy
```

**D) Moons — NumPy tự viết (không `make_moons`)**

```python
def _moon_arc(random, n, shift, flip):
    angles = random.uniform(0.0, np.pi, n)
    radius = 1.0 + random.normal(0.0, 0.08, n)
    x = radius * np.cos(angles) + shift
    y = radius * np.sin(angles)
    if flip:
        x = 1.0 - x
        y = -y + 0.35
    y = y + random.normal(0.0, 0.06, n)
    return np.column_stack((x, y))
```

**E) Circle in circle — lại dùng `_disk` (NumPy)**

```python
def make_circles_data(seed=2207):
    random = np.random.default_rng(seed)
    majority_train = _disk(random, n=300, radius=0.55)                 # đĩa trong
    minority_train = _disk(random, n=50, radius=1.45, thickness=0.12)  # vành ngoài
```

**Gợi ý lúc demo code:** mở `server.py`, nhảy tới `make_gaussian_circle_data` (bộ đang chạy) → nếu GV hỏi “sklearn chỗ nào?” thì nhảy `_swiss_2d` / `make_swiss_roll`.

### 4.4 Phân bố điểm trông như thế nào?

Tưởng tượng mặt phẳng 2 trục x, y:

**Lớp thiểu số (đỏ, n⁺ = 50)**  
- Nằm quanh gốc tọa độ (0, 0).  
- Mỗi tọa độ được bốc theo phân phối chuẩn (hình chuông), độ lệch chuẩn khoảng 0.28.  
- Nhìn: một **đốm tròn đặc** ở giữa.

**Lớp đa số (xám, n⁻ = 300)**  
- Nằm trên một **vòng ngoài** (như vành bánh xe).  
- Góc quay ngẫu nhiên quanh vòng; bán kính khoảng 1.55, dao động nhẹ ±0.14.  
- Nhìn: vòng bao quanh đốm đỏ.

**Vì sao hợp để demo?**  
Đốm thiểu số ở trong, vòng đa số ở ngoài. Nếu sinh điểm “bừa” (Global / Gaussian) dễ đẩy điểm giả ra chỗ trống hoặc gần vòng xám. Cách local (SMOTE / Simplicial) cố gắng sinh gần đốm đỏ hơn.

**Tập test của app**  
- 30 điểm thiểu số + 120 điểm đa số, cùng kiểu phân bố.  
- Chỉ để chấm điểm; **không** đem đi oversample.

**Sau khi sinh đủ 250 điểm giả**

| Tập | Trước | Sau |
|-----|-------|-----|
| Thiểu số train | 50 đỏ | 50 đỏ + 250 xanh = 300 |
| Đa số train | 300 xám | vẫn 300 xám |
| Test | 30 cam + 120 xám | không thêm xanh |

### 4.5 Ba bộ kia là gì? (chỉ giới thiệu miệng)

Cả bốn bộ đều: train 50 thiểu số / 300 đa số (IR = 6).

| Tên trên app | Nhìn đại khái | Sinh bằng |
|--------------|---------------|-----------|
| Two moons | Hai vành trăng khuyết; thiểu số một vành, đa số vành kia | NumPy (`_moon_arc`) |
| Swiss roll | Thiểu số nằm trên “cuộn giấy” chiếu xuống 2D; đa số rải đều quanh đó | **sklearn** `make_swiss_roll` + NumPy (đa số) |
| Gaussian in circle | **Đốm giữa + vòng ngoài** (bộ demo chính) | NumPy (`normal` + `_disk`) |
| Circle in circle | Thiểu số vòng ngoài; đa số đĩa đặc bên trong | NumPy (`_disk`) |

### 4.6 App chạy những bước gì? (khác phần Experiments slide)

**Trên demo:**

1. Chuẩn hóa số liệu (kéo về cùng thang đo) — **chỉ học từ train**, rồi áp sang test.  
2. Sinh điểm giả trên train (tùy method).  
3. Huấn luyện classifier hàng xóm KNN (5 hàng xóm).  
4. Chấm trên test; vẽ vùng quyết định + các chỉ số bên phải.

**Trên slide Experiments:**  
21 bộ dữ liệu thật, chia train–test nhiều vòng lồng nhau, thử nhiều k và p.  
→ **Hai chuyện khác nhau**; đừng trộn số.

### 4.7 Các số bên phải Outcome nghĩa là gì?

Gọi “bắt đúng thiểu số” là điều ta quan tâm.

| Tên trên UI | Nói tiếng Việt dễ hiểu |
|-------------|-------------------------|
| Detection (vd. 30/30) | Trong 30 điểm thiểu số test, bắt đúng bao nhiêu |
| Recall | Trong **tất cả** thiểu số thật, bắt được bao nhiêu phần trăm (càng cao càng ít bỏ sót) |
| Precision | Trong những lần máy **bảo là** thiểu số, bao nhiêu lần đúng (càng cao càng ít báo nhầm) |
| F1 | Điểm cân giữa recall và precision |
| MCC | Điểm tổng hợp cả bốn ô đúng/sai; hữu ích khi hai lớp lệch nhau |
| FN | Thiểu số bị **bỏ sót** (nguy hiểm nếu lớp thiểu số quan trọng) |
| FP | Đa số bị **báo nhầm** thành thiểu số |

---

## 5. Tính toán từng method (chi tiết để trả lời giáo viên)

### 5.1 Luồng chung của một lần chạy

Mọi method trong app đều đi qua cùng một pipeline. Chỉ phần sinh dữ liệu synthetic thay đổi.

```text
1. Tạo dataset có train và test riêng.
2. Fit StandardScaler trên train בלבד.
3. Dùng scaler đó để transform train và test.
4. Chọn method và chỉ sinh synthetic từ train.
5. Nối synthetic vào train, không nối vào test.
6. Train KNeighborsClassifier(n_neighbors=5).
7. Dùng model dự đoán test chưa từng được nhìn thấy.
8. So sánh dự đoán với nhãn thật và tính metric.
```

Trong code, luồng chính tương ứng với:

```text
scaler.fit(raw_data["train_x"])
train_x = scaler.transform(raw_data["train_x"])
test_x = scaler.transform(raw_data["test_x"])
sampled_x, sampled_y = run_sampler(...)
model = train_model(sampled_x, sampled_y, seed)
result = metrics(model, test_x, test_y)
```

Với Gaussian in circle:

```text
Train ban đầu: 50 minority + 300 majority = 350 điểm
Full budget:   thêm 250 synthetic minority
Train sau đó:  300 minority + 300 majority = 600 điểm
Test:          30 minority + 120 majority = 150 điểm, giữ nguyên
```

**Câu trả lời quan trọng:** app không train rồi test trên cùng một tập. App tách test trước, không cho test tham gia vào việc chuẩn hóa, sinh synthetic hoặc train model.

### 5.2 Imbalanced

Imbalanced là mốc không oversampling.

1. App chuẩn hóa train và test bằng scaler fit trên train.
2. App không sinh synthetic.
3. `sampled_x` chính là `train_x` ban đầu.
4. `sampled_y` chính là nhãn train ban đầu.
5. KNN được train trên 50 minority và 300 majority.
6. KNN dự đoán 150 điểm test.
7. App đếm TP, TN, FP, FN rồi tính recall, precision, F1 và MCC.

Ở method này, model học từ dữ liệu lệch lớp nên đây là baseline để so sánh.

### 5.3 Random

Random tạo synthetic bằng cách sao chép minority train.

1. App lấy riêng `minority = train_x[train_y == 1]`, gồm 50 điểm.
2. Bộ sinh số ngẫu nhiên chọn 250 chỉ số minority, có hoàn lại.
3. “Có hoàn lại” nghĩa là một điểm minority có thể được chọn nhiều lần.
4. Mỗi synthetic point bằng đúng tọa độ của minority parent được chọn:

```text
x̂ = x_parent
```

5. App gắn nhãn `1` cho 250 bản sao.
6. App nối 250 bản sao vào train.
7. Model học trên 300 minority gồm 50 điểm gốc và 250 bản sao, cùng 300 majority.
8. Test vẫn giữ nguyên và không chứa các bản sao này.

Random làm tăng trọng số của các điểm minority đã có, nhưng không mở rộng vùng hình học của minority. Vì vậy, nhiều điểm xanh có thể nằm chồng lên điểm đỏ.

### 5.4 Gaussian

Gaussian tạo synthetic bằng phân phối chuẩn nhiều chiều được fit trên minority train.

1. App lấy 50 minority train.
2. App tính trung bình vector `μ` của 50 điểm.
3. App tính ma trận hiệp phương sai `Σ` của 50 điểm.
4. App thêm một lượng rất nhỏ `1e-6` vào đường chéo của `Σ` để ma trận ổn định khi lấy mẫu.
5. App sinh 250 điểm từ:

```text
x̂ ~ N(μ, Σ)
```

6. App gắn nhãn minority cho các điểm này và nối chúng vào train.
7. KNN được train trên train sau tăng cường.
8. Test độc lập được dùng để tính metric.

Ellipse trên canvas là cách minh họa trung tâm và hiệp phương sai của minority. Gaussian không dùng quan hệ k-NN và không biết cấu trúc vòng majority bao quanh.

### 5.5 Global

Global tạo synthetic bằng cách trộn một cặp minority bất kỳ trong toàn bộ lớp minority.

1. App gọi sampler `Mixup(k=2, random_state=seed)` từ vendor.
2. Sampler chọn hai điểm minority, không yêu cầu hai điểm phải là hàng xóm.
3. Sampler chọn trọng số trộn giữa hai parent.
4. Điểm synthetic có dạng:

```text
x̂ = λ₁x₁ + λ₂x₂
với λ₁ ≥ 0, λ₂ ≥ 0 và λ₁ + λ₂ = 1
```

5. Vì chỉ có hai parent, x̂ nằm trên đoạn nối x₁ và x₂.
6. App lấy 250 synthetic points đầu tiên cần dùng để cân bằng train.
7. App nối synthetic vào train và train KNN.
8. App giữ test nguyên vẹn rồi tính metric.

Global khác SMOTE ở điều kiện chọn parent. Global có thể chọn hai điểm ở xa nhau; SMOTE chỉ chọn cặp theo quan hệ local trên k-NN graph.

### 5.6 SMOTE

SMOTE tạo synthetic trên các cạnh của đồ thị k-NN minority.

1. App đưa train đã chuẩn hóa vào sampler SMOTE.
2. Sampler lấy riêng 50 minority train.
3. Với mỗi minority point, sampler tìm `k = 5` hàng xóm gần nhất theo khoảng cách Euclid.
4. Các quan hệ hàng xóm tạo thành đồ thị minority.
5. Sampler chọn một parent và một hàng xóm của parent.
6. Sampler chọn tỉ lệ `λ` giữa 0 và 1.
7. Điểm mới được tính:

```text
x̂ = x_parent + λ(x_neighbor − x_parent)
```

8. Vì `0 ≤ λ ≤ 1`, x̂ nằm trên đoạn nối hai parent.
9. Sampler lặp quá trình cho đến khi tạo đủ 250 synthetic points.
10. App nối synthetic vào train, train KNN và test trên holdout nguyên bản.

Canvas vẽ web k-NN màu xám để người xem thấy các cạnh ứng viên. Điểm xanh đang được tạo nằm trên một cạnh được highlight.

### 5.7 Simplicial SMOTE

Simplicial SMOTE mở rộng ý tưởng SMOTE từ một cạnh sang một nhóm nhiều điểm gần nhau.

1. App đưa train đã chuẩn hóa vào `SimplicialSMOTE(k=5, p=3, random_state=seed)`.
2. Sampler lấy riêng minority train, không lấy majority để dựng simplex.
3. Sampler dựng đồ thị k-NN trên 50 minority points.
4. Hai điểm được nối nếu một điểm nằm trong 5 hàng xóm của điểm kia hoặc ngược lại.
5. Sampler tìm các clique, tức các nhóm mà mọi cặp điểm đều có cạnh.
6. Với cấu hình demo, sampler xét nhóm có tối đa 3 đỉnh theo quy ước code/vendor. Nhóm 3 đỉnh tạo thành một tam giác trong không gian 2D.
7. Mỗi lần sinh mẫu, sampler chọn một simplex ứng viên, có thể chọn lại simplex cũ.
8. Sampler tạo các trọng số barycentric không âm và có tổng bằng 1:

```text
λ₁ ≥ 0, λ₂ ≥ 0, λ₃ ≥ 0
λ₁ + λ₂ + λ₃ = 1
```

9. Điểm mới được tính:

```text
x̂ = λ₁x₁ + λ₂x₂ + λ₃x₃
```

10. Vì các trọng số không âm và tổng bằng 1, x̂ nằm trong tam giác tạo bởi ba parent.
11. Sampler lặp lại cho đến khi tạo đủ 250 synthetic points.
12. App nối các điểm mới vào train, train KNN và chỉ đánh giá trên test độc lập.

Trong phần animation, app lưu parent indices và dựng metadata để giải thích. Các giá trị `exp_draws` và `weights` được dùng để minh họa cách tính trọng số trên canvas. Điểm synthetic dùng cho train và metric vẫn là điểm do sampler chính thức tạo.

### 5.8 Ví dụ tính một điểm Simplicial bằng tay

Giả sử sampler chọn ba parent:

```text
x₁ = (0.0, 0.0)
x₂ = (1.0, 0.0)
x₃ = (0.0, 1.0)
```

Giả sử trọng số là:

```text
λ₁ = 0.2
λ₂ = 0.5
λ₃ = 0.3
```

Tổng trọng số bằng `0.2 + 0.5 + 0.3 = 1`.

Khi đó:

```text
x̂ = 0.2(0.0, 0.0) + 0.5(1.0, 0.0) + 0.3(0.0, 1.0)
   = (0.5, 0.3)
```

Điểm `(0.5, 0.3)` nằm bên trong tam giác. Nếu chỉ dùng SMOTE với hai parent, điểm mới chỉ nằm trên một cạnh của tam giác, không tự do nằm trong phần diện tích bên trong.

### 5.9 Sau khi sinh mẫu: train, test và metric

Tất cả method đều kết thúc bằng cùng một quy trình:

1. App nối synthetic points vào `train_x`.
2. App nối nhãn minority `1` vào `train_y`.
3. App fit `KNeighborsClassifier(n_neighbors=5)` trên train đã tăng cường.
4. App dùng model dự đoán `test_x`.
5. App so sánh dự đoán với `test_y`.
6. App đếm:
   - TP: minority dự đoán đúng là minority;
   - TN: majority dự đoán đúng là majority;
   - FP: majority bị dự đoán nhầm là minority;
   - FN: minority bị dự đoán nhầm là majority.
7. App tính:

```text
recall    = TP / (TP + FN)
precision = TP / (TP + FP)
F1        = 2 × precision × recall / (precision + recall)
```

MCC được tính từ cả TP, TN, FP và FN. Nếu `TP + FN` bằng 0 hoặc `TP + FP` bằng 0, implementation dùng `zero_division=0` cho precision/recall để tránh lỗi.

### 5.10 So sánh các method công bằng

Một lần so sánh hợp lý phải giữ nguyên:

```text
dataset       = Gaussian in circle
seed          = 2207
train/test    = cùng một split cố định
scaler        = fit trên train
n_synthetic   = cùng budget
classifier    = KNN với 5 hàng xóm
test set      = cùng một test nguyên bản
```

Chỉ method oversampling được thay đổi. Vì vậy, khi so sánh, ta quan sát cả hai mặt:

- **Hình học:** synthetic points có nằm đúng vùng minority không, có đi vào vùng majority không, có phủ được vùng bên trong simplex không.
- **Đánh giá:** recall, precision, F1, MCC, FN và FP trên cùng test set.

Không nên kết luận một method luôn tốt hơn chỉ từ một lần chạy demo. Muốn kết luận theo paper cần nhiều seed, cross-validation, chọn siêu tham số và nhiều dataset.

---

## 6. Thuật toán Simplicial — giải thích dễ hiểu (để trả lời, không đọc nguyên)

Phần này khớp slide 10–14, nhưng viết lại cho dễ nói.

### Ý chính một câu

SMOTE sinh điểm trên **đoạn nối hai điểm gần nhau**.  
Simplicial SMOTE sinh điểm trong **nhóm nhiều điểm gần nhau** (tam giác, …) — nên phủ được vùng.

### Đầu vào cần gì?

- Các điểm thiểu số X⁺ (đã chuẩn hóa trong demo).  
- m = cần bao nhiêu điểm giả (thường = n⁻ − n⁺).  
- **k** = mỗi điểm nhìn bao nhiêu hàng xóm gần nhất.  
- **p** = cho phép nhóm lớn tới mức nào (slide: p là “chiều”; nhóm p có p+1 đỉnh).  
  Ví dụ: p = 1 → chỉ cạnh (2 điểm) = SMOTE; p = 2 → cho phép tam giác (3 điểm).

### Bước 1 — Vẽ “lưới hàng xóm”

1. Với mỗi điểm đỏ, tìm k điểm đỏ gần nó nhất.  
2. Nối cạnh nếu A là hàng xóm của B hoặc B là hàng xóm của A.  
3. Trên canvas: lưới xám chính là các cạnh này (chỉ hiện ở SMOTE / Simplicial).

**k lớn:** lưới dày, dễ có nhóm lớn.  
**k nhỏ:** chỉ nối người rất gần.

### Bước 2 — Từ cạnh thành nhóm (clique / simplex)

Nói đơn giản:

| Tên học thuật | Nói dễ hiểu | Ví dụ |
|---------------|-------------|--------|
| Clique | Một nhóm điểm **đôi một đều nối được** | 3 điểm tạo thành tam giác đầy đủ cạnh |
| Cạnh (1-simplex) | Nhóm 2 điểm | Đoạn thẳng — chỗ SMOTE sinh điểm |
| Tam giác (2-simplex) | Nhóm 3 điểm đôi một nối | Vùng bên trong tam giác — hay thấy trên demo |
| p-skeleton | Chỉ xét các nhóm không lớn quá mức p cho phép | p = 1 chỉ cạnh; p = 2 thêm tam giác |
| Maximal simplex | Nhóm “đã lớn hết mức” trong giới hạn đó | Đã có tam giác ABC thì không cần coi cạnh AB là nhóm đích riêng nữa |

**Vì sao lấy nhóm maximal?**  
Sinh điểm trong tam giác ABC đã bao gồm việc nội suy trên cạnh AB. Nếu còn sample riêng AB thì dễ bị trùng ý / thiên về cạnh nhỏ.

### Bước 3 — Lặp m lần để đủ điểm giả

Mỗi lần:

1. **Chọn một nhóm** (maximal) trong danh sách — paper chọn đều, có thể chọn lại nhóm cũ.  
2. **Bốc trọng số** λ cho từng đỉnh trong nhóm: mỗi λ ≥ 0 và **cộng lại = 1**.  
   Cách bốc “công bằng” trên cả vùng nhóm gọi là Dirichlet(1,1,…,1) — hiểu đơn giản: điểm mới có thể rơi **bất kỳ đâu trong nhóm**, không thiên góc nào.  
3. **Điểm mới** = trộn các đỉnh theo trọng số đó  
   (ví dụ 3 điểm A,B,C với trọng số 0.2, 0.5, 0.3 → điểm nằm trong tam giác, gần B hơn).

Gắn nhãn thiểu số cho điểm mới; làm vậy đủ m lần.

### Trên giao diện bước “show-weights”

App minh họa cách bốc trọng số bằng vài số ngẫu nhiên rồi chia cho tổng (tương đương Dirichlet).  
Nhưng điểm thật dùng để huấn luyện / tính metric vẫn do **sampler backend** tạo — đừng nghĩ tự tính tay trên UI mới ra F1.

### Ba câu hỏi slide — trả lời miệng

1. **Nhóm hàng xóm là gì?**  
   Nhóm điểm thiểu số đôi một đều là hàng xóm trên lưới k-NN.

2. **Sinh từ nhóm nào?**  
   Các nhóm lớn nhất được phép (maximal) trong giới hạn p.

3. **Điểm mới nằm đâu?**  
   Bên trong nhóm: trung bình có trọng số của các đỉnh; trọng số cộng = 1.

### So các method trên canvas cho dễ nhớ

| Method | Làm gì với điểm thiểu số? | Điểm mới thường nằm đâu? | Nhìn trên hình |
|--------|---------------------------|---------------------------|----------------|
| Imbalanced | Không sinh | Không có | Không xanh, không lưới |
| Random | Copy điểm sẵn có | Trùng chỗ điểm cũ | Sáng điểm cha |
| Global | Lấy **hai điểm bất kỳ** trong cả lớp | Trên đoạn nối — có thể xa / vào chỗ trống | Chỉ một cạnh cặp |
| Gaussian | Fit hình chuông rồi bốc | Theo ellipse ước lượng | Ellipse xanh |
| SMOTE | Lấy **hai điểm gần nhau** | Trên đoạn hàng xóm | Lưới xám + cạnh |
| Simplicial SMOTE | Lấy **nhóm gần nhau** (đôi một nối) | **Trong** tam giác / simplex | Lưới + tô đầy tam giác |

### Vì sao hay thấy tam giác trên demo 2D?

- Không bắt buộc phải 3 đỉnh — phụ thuộc p và lưới có tạo được nhóm lớn không.  
- Trên mặt phẳng 2D, tam giác là hình “đầy” dễ nhìn nhất.  
- p quá nhỏ ≈ giống SMOTE; p quá lớn đòi hỏi nhóm rất đông → ít nhóm đủ điều kiện, kết quả kém ổn định. Slide khuyên thử nhiều p giống như thử nhiều k.

---

## 6. Hỏi–đáp sẵn (trả lời ngắn, dễ hiểu)

**Q: n⁺ và n⁻ là gì?**  
A: n⁺ = số điểm thiểu số; n⁻ = số điểm đa số. Train demo: 50 và 300.

**Q: m tính sao?**  
A: m = n⁻ − n⁺ = 300 − 50 = 250 — số điểm giả cần thêm để hai lớp bằng số lượng.

**Q: IR là gì?**  
A: Đa số gấp thiểu số bao nhiêu lần = n⁻ / n⁺ = 6.

**Q: Oversample có đụng điểm xám không?**  
A: Không. Giữ nguyên đa số; chỉ thêm điểm giả cho thiểu số.

**Q: Điểm cam có tính vào n⁺ không?**  
A: Không. n⁺ là thiểu số **train**. Cam là thiểu số **test**, chỉ để chấm điểm.

**Q: Sao không demo moons / swiss roll?**  
A: moons thì Simplicial hòa SMOTE; swiss roll thì SMOTE còn cao hơn. Gaussian in circle là chỗ Simplicial thắng rõ trên bảng paper.

**Q: Dataset sinh bằng thư viện gì?**  
A: Paper: synthetic §4.1, một phần nhờ scikit-learn. App (`server.py`):  
- **Gaussian in circle** (bộ demo): thuần **NumPy** — `random.normal` + `_disk` (xem `make_gaussian_circle_data`).  
- **moons / circles:** cũng NumPy (`_moon_arc` / `_disk`), **không** gọi `make_moons` / `make_circles`.  
- **Swiss roll:** có gọi **`sklearn.datasets.make_swiss_roll`**, đa số quanh đó bằng NumPy `uniform`.  
Có thể mở đúng các hàm trên để show code.

**Q: F1 trên app khác paper?**  
A: Bình thường. Paper chạy nhiều lần / nhiều lần chia dữ liệu. App chỉ một lần cố định để nhìn hình.

**Q: Phần Experiments slide khác demo chỗ nào?**  
A: Slide = 21 bộ dữ liệu thật + đánh giá lồng nhiều vòng. Demo = hình 2D synthetic. Không thay thế nhau.

**Q: Global khác Simplicial thế nào?**  
A: Global lấy cặp **bất kỳ** trong cả lớp. Simplicial chỉ lấy nhóm điểm **thật sự gần nhau**. Global dễ sinh điểm vào chỗ trống; Simplicial bám cụm local.

**Q: Random khác Imbalanced?**  
A: Random **có** thêm điểm nhưng chỉ **copy** điểm cũ → số lượng tăng, hình dạng vùng thiểu số hầu như không mở rộng.

**Q: Clique khác simplex?**  
A: Trong paper/slide coi gần như một: nhóm điểm đôi một nối được trên đồ thị; về hình học gọi là simplex (2 điểm = đoạn, 3 điểm = tam giác).

**Q: Maximal simplex là gì?**  
A: Nhóm đã “lớn hết mức” trong giới hạn cho phép — không còn nhét thêm điểm nào mà vẫn đôi một nối được. Sinh từ nhóm này để khỏi lặp lại các cạnh con.

**Q: k và p khác nhau sao?**  
A: k = nhìn bao nhiêu hàng xóm (lưới dày/thưa). p = cho phép nhóm lớn tới đâu (1 = chỉ cạnh như SMOTE; lớn hơn = vào được trong tam giác…). Nên chỉnh cả hai.

**Q: Dirichlet / trọng số λ là gì?**  
A: Các tỉ lệ trộn điểm cha; không âm; cộng lại = 1. Cách bốc “công bằng” giúp điểm mới có thể nằm bất kỳ đâu trong nhóm, không chỉ gần một góc.

**Q: Demo có Borderline / Safe-level / ADASYN simplicial không?**  
A: Không. Canvas chỉ các baseline bảng synthetic + Simplicial SMOTE. Các biến thể nằm ở slide.

**Q: Chuẩn hóa (StandardScaler) để làm gì?**  
A: Đưa các trục về cùng tầm đo trước khi tính khoảng cách / hàng xóm. Chỉ học từ train rồi áp sang test — tránh “nhìn trước” test.

**Q: Classifier trên demo là gì?**  
A: KNN 5 hàng xóm — cùng họ classifier paper hay dùng trên bảng synthetic, nhưng cách chấm điểm của app đơn giản hơn paper.

**Q: Chữ pool / cluster trên bước UI?**  
A: Là danh sách nhóm (cạnh / simplex) đang được chọn để sinh điểm. Không phải thuật toán clustering riêng.

**Q: Có được nói Simplicial luôn hơn SMOTE?**  
A: Không. Paper/slide: trung bình thường tốt hơn, nhưng **không phải lúc nào cũng thắng**. Demo chọn Gaussian in circle vì đó là case thắng rõ trên Table 1.

---

## 7. Checklist trước giờ demo

- [ ] Giải thích được **n⁺ / n⁻ / m / IR** bằng tiếng Việt trong ~20 giây
- [ ] `./run.sh` hoặc `./share.sh` đang chạy
- [ ] Chọn **Gaussian in circle** (50 / 300 / cần 250 điểm giả)
- [ ] Thứ tự click: Imbalanced → Global → Gaussian → SMOTE → **Simplicial SMOTE**
- [ ] Dừng được một mẫu Simplicial có ≥ 3 đỉnh (thấy tam giác)
- [ ] Nhớ câu đóng: “p = 1 thì đúng SMOTE; p lớn hơn mới vào trong simplex”
- [ ] Nếu hỏi số paper → lấy Table 1; **đừng** lấy số live trên app làm bằng chứng
- [ ] Phân biệt màu: đỏ = thiểu số train · cam = thiểu số test · xanh = điểm giả
