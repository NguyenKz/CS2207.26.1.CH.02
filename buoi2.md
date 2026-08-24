
# 


## Characterization and Distribution

### Data Characterization
- Phân phối chuẩn: Mean, Variance, Skewness, Kurtosis
    - Tại sao phân phối chuẩn quan trọng?
        - 1 đại lượng ngẩu nghiên sẽ có 1 phân phối, vd, nhiệt độ, chiều cao.... -> Thường các đại lượng này tuân theo phân phối chuẩn nên phân phối chuẩn quan trọng. 
        - Các mô hình thường phải xây dựng trên 1 phân phối  nào đó, nên chings ta phải biết ít nhất moo hình hoạt động tốt trên phân phối nào -> có thể áp dụng với data này không 
            - VD: model chạy tốt trên phân phối chuẩn, và data điểm thi cũng uân theo phân phối chuẩn.

    - Vd Điểm thi của Chuyên Tuyên Quang 2026, lệch về phi điểm cao -> bất thường hoặc học sinh giỏi.

->Tìm các phân phối của data.

### Distribution Identification
- Trích xuất đặt trưng của data để sử dụng.


## Mining Frequent Patterns and Associations
### Frequent Patterns
- Itemsets: tập các items có tần suất xuất hiện lớn hơn một ngưỡng nào đó.
- Sub-sequence: tập các items có tần suất xuất hiện lớn hơn một ngưỡng nào đó.
- sub-structure: tập các items có tần suất xuất hiện lớn hơn một ngưỡng nào đó.

### Associations
- 100 người vào của hảng, có 30 người mua máy tính, vậy trong 30 người đó có bao nhiêu người mua tai nghe?

Cách tính:
All = 100
A = event (by computer) = 30
B = event (by headphone) = 20  
Xác xuât 1 người vừa mua máy tính và tai nghe.
- support = A giao B = 10 (giả sử)
- confidence  = P(B|A) (xác xuất có điều kiện) = P(A giao B) / P(A) = 10 / 30 = 0.33

VD: Xác xuất số 1 xẩy ra khi tung xúc xắc và biết số này lẻ = 1/3 = P(1|lẻ)

### Other examples


## Phân loại và hồi quy: Classification and Regression

- Xác định được bài toán của mình là gì? Phân loại hay hồi quy.

- Classification: Phân loại vào 1 trong nhiều lớp. -> rời rac,
    - Cần có Rule base để phân loại.
    - 2 loại rule: Tree based, Rule based.

    
- Regression: Dự đoán giá trị của một đại lượng dựa trên các đại lượng khác. -> liên tục,

## Phân nhóm: Clustering

- Tôi muốn dùng máy để gom cụm data này thành x cụm vì tôi không thể làm tay được, quá phức tạp, quá nhiều data.

## Outlier Analysis

- Phát hiện bất thường: VD, máy tính bị mh xanh là bất thường, lon nước bị móp thì bất thường....

## Technologies:

- Nhiều kỹ thuật, chỉ cần dùng và ứng dụng được trên 1 tập dữ liệu lướn lại nhỏ được.


## Quiz:
1. 
- Data analysis không phải là 1 buociws kha phá dữ liệu.
- Data clearning, data mining, data integration: Là các bước khai phá dữ liêu.

# Tiền xử lý dữ liệu: Data Cleaning, Data Integration, Data Transformation

## Data Pre-processing 

- Tăng chất lượng data, vd xóa bớt ngoại lệ, xóa bớt noise, không nhất quan, vd, điểm số >10 (data sai)
- Data quality:
    - Accuracy: Độ chính xác của dữ liệu.
    - Completeness: Độ đầy đủ của dữ liệu.
    - Consistency: Độ nhất quán của dữ liệu.
    - Timeliness: Độ tức thời của dữ liệu.
        - Chuyến bay, điểm số, thời tiết, giá cả,....
    - Believability: Độ tin cậy của dữ liệu.
        - Sức khỏe của người dùng, có thể dựa vào các dữ liệu khác để xác định độ tin cậy của dữ liệu.
    - Interpretability: Độ dễ hiểu của dữ liệu.


## Statistical Description of data:

- Central Tendency:
    - Mean: Trung bình cộng của dữ liệu. (average)
        - Vì sao nó được gọi là trung bình công? Công thức này rất đơn giản, nhưng giải thích và tìm ra thì rất khó, nhiều nguời giải tích khác nhau, tùy vào cách hiểu. 
            - VD: Tìm 1 điểm C mà tổng bình phương khoản cách tất cả các điểm đến C là nhỏ nhất. -> Lấy đạo hàm cấp 1 là ra trung bình công.
            - VD: Dùng hình học euclid và hàm mật độ bằng phương pháp moment của phân phối chuẩn..... 
    - Weighted average: Trung bình cộng của dữ liệu, nhưng mỗi điểm có trọng số khác nhau.
    - Median: Trung vị của dữ liệu. (middle)
        - VD: 1,2,4,6,8 -> Median = 4 Không quan tâm giá trị.
        - VD: 3,5,2,6,9 -> 2,3,5,6,9 -> Median = 5
        - VD: 1,2,3,4,5,6,7,8,9,10 -> Median = (5+6)/2 = 5.5
    - Mode: Giá trị xuất hiện nhiều nhất của dữ liệu. (mode)

- Mean và Median đều biểu diễn central tendency thì tại sao lại cần cả 2. Vì mean rất nhạy với outlier, còn median thì không. VD: lương 3 người: 10tr, 20tr, 1tỷ -> mean = 1ty30tr/3 = 333tr333 -> không phản ánh đúng tình hình.

- Measuring the dispersion:
    - Range: Khoảng cách giữa giá trị lớn nhất và giá trị nhỏ nhất của dữ liệu.
    - Outlier: Giá trị ngoại lai, giá trị bất thường, giá trị không phản ánh đúng tình hình.
    - Interquartile Range: Khoảng cách giữa phần tư thứ nhất và phần tư thứ ba của dữ liệu. (Python Bosplot )
    - Variance: Phương sai của dữ liệu.

## Data Cleaning

- Xử lý dữ liệu bất thường, dữ liệu nhiễu, dữ liệu thiếu, dữ liệu sai, dữ liệu trùng lặp...
- Filling missing values:
    - Filling missing values, bắt buộc phải fill vào, nếu không sẽ không dùng được.
    - Cách fill:
        - Dễ nhất: Thiếu thì bỏ đi. -> Mất data rất phí, trừ khi có qua nhiều data :)))
        - Làm được mà k ai làm: Điền tay.
        - Điền tự động:
            - Mặc định 1 con số nào đó, tùy chuyên gia.
            - Sửa dụng mean hoặc median hoặc mode.
            

- Smoothing:


## Data Integration

- Tích hợp dữ liệu từ nhiều nguồn khác nhau.

## Data Transformation

- Biến đổi dữ liệu từ dạng này sang dạng khác.