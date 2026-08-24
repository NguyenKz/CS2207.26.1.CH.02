
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
                - Chỉ dùng của data lân cận, không lấy của các class trái ngược nhau.
    - Missing value không có nghĩa là data lỗi, vd, khách không dùng thẻ tín ddụng thì ngày dùng thẻ trống là đúng.

- Smoothing:
    - Binning
        - Thay thế giá trị bằng trung bình cộng trong khoảng giá trị lân cận. 
        - vd: 1,10,5,30,20,60--> 1, 7.5, 25.....

## Data Integration

- Có nhiều data từ nhièu đơn vị thì làm sao tích hợp về cùng 1 db?
- Entity Integration:
    - Match các filed cùng ý nghĩa với nhau.
        VD: Cốt taskid ở cty a, cột runid ở công ty b là 1. -> merge thành 1 cột là taskid ahy run id.
- Redundancy Reduction and Correlation Analysis:
    - Phân tích tương quan để giảm chiều dữ liệu.
        vd: x1 = 1,2,3
            x2 = 2, 4, 6 
            -> x2  = 2*x1 => chỉ cần giữ x1 hoặc x2 là đủ. có thể nội suy ra cái còn lại.
    - Vì sau cần phải xử lý: Vì trong quá trình làm ML, các thuật toán ma trận nghịch đảo, ma trận tạo thành từ x1,x2 có thể không nghịch đảo -> không tính được.
    - Có 2 cách tổng quát:
        - Correlation Coefficient (Hệ số tương quan)
        - Covariance (Hiêu phương sai)

    1. Covariance and Correlation Coefficient:
        - Tương quan, đồng biến: x tăng, y tăng
            - Đồng biết mạnh hay nhẹ.... vd x tăng 100 y tăng 0.01 -> có thể bỏ qua vì đồng biến rất nhẹ
        - Tương quan, nghịch biến: x tăng, y giảm
        - Không tương quan: x tăng, y không tăng không giảm.


VD:

|A|B|
|-|-|
|6|20|
|5|10|
|4|14|
|3|5|
|2|5|

n = 5
Ā = (6+5+4+3+2)/5 = 4
B̄ = (20+10+14+5+5)/5 = 10.8

Covariance = (1/n) Σ (Aᵢ-Ā)(Bᵢ-B̄)
= 1/5 * [(6-4)(20-10.8) + (5-4)(10-10.8) + (4-4)(14-10.8) + (3-4)(5-10.8) + (2-4)(5-10.8)]
= 1/5 * (18.4 - 0.8 + 0 + 5.8 + 11.6)
= 1/5 * 35
= 7

σ_A² = 1/5 * [(6-4)² + (5-4)² + (4-4)² + (3-4)² + (2-4)²] = 1/5 * 10 = 2  → σ_A = √2
σ_B² = 1/5 * 162.8 = 32.56  → σ_B = √32.56

Correlation Coefficient = Cov / (σ_A * σ_B) = 7 / (√2 * √32.56) ≈ 0.867

(r = 1 chỉ khi B = aA + b, data này không thẳng hàng.)

-> Đồng biến


## Data Transformation

- Biến đổi dữ liệu từ dạng này sang dạng khác.