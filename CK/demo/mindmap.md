# Sơ đồ tư duy: Simplicial SMOTE

Nguồn: [Simplicial SMOTE (arXiv:2503.03418)](../papers/2503.03418v1.pdf). Số trang trong ngoặc là số trang PDF.

Paper chỉ đề xuất một phương pháp tạo mẫu (sampler), không đề xuất mô hình mới. Phần "train" chỉ dùng để đo hiệu quả: k-NN và Gradient Boosting được train trên dữ liệu đã tăng cường, rồi so sánh F1 và MCC.

```mermaid
flowchart LR
    R["Simplicial SMOTE<br/>(KDD '25)"]

    R --> P["1. Vấn đề (tr. 2)"]
    P --> P1["Dữ liệu mất cân bằng,<br/>bộ phân loại lệch về lớp đa số"]
    P --> P2["SMOTE chỉ sinh điểm<br/>trên đoạn thẳng 1 chiều: độ phủ thấp"]
    P --> P3["Phương pháp toàn cục:<br/>chất lượng mẫu thấp"]

    R --> I["2. Ý tưởng (tr. 2-3)"]
    I --> I1["Thay đoạn thẳng bằng simplex<br/>(đoạn, tam giác, tứ diện)"]
    I --> I2["Điểm mới = tổ hợp barycentric<br/>của nhiều điểm lân cận"]
    I --> I3["Điểm sinh gần lớp đa số hơn:<br/>khoảng cách 0.577 so với 0.707"]

    R --> M["3. Phương pháp (tr. 3-5)"]
    M --> M1["Algorithm 1 (tr. 4):<br/>k-NN, clique, chọn simplex, Dirichlet"]
    M --> M2["Biến thể (tr. 5):<br/>Borderline, Safe-level, ADASYN"]
    M --> M3["Độ phức tạp (tr. 5):<br/>không phụ thuộc số chiều"]

    R --> E["4. Thực nghiệm (tr. 5-7)"]
    E --> E1["4.1 Dữ liệu tổng hợp (tr. 5-6):<br/>moons, swiss rolls, 2 dạng hình tròn"]
    E --> E2["4.2 Dữ liệu thật (tr. 6):<br/>21 dataset, tỉ lệ mất cân bằng 9 đến 130"]
    E --> E3["4.3 Thời gian chạy (tr. 6-7):<br/>chậm hơn SMOTE khoảng 1.4 lần"]
    E2 --> T["Phần train:<br/>k-NN và Gradient Boosting<br/>đánh giá bằng F1 và MCC"]
    E2 --> T2["Kiểm định: Friedman + Conover"]

    R --> K["5. Kết quả chính (tr. 2, 6)"]
    K --> K1["F1 so với SMOTE:<br/>+4.5% (k-NN), +5% (GB)"]
    K --> K2["Mọi biến thể simplicial<br/>tốt hơn bản dùng đồ thị"]

    R --> C["6. Kết luận (tr. 7-8)"]
    C --> C1["Nên grid search p:<br/>p nhỏ hoặc lớn quá đều kém"]

    R --> A["7. Phụ lục (tr. 9-14)"]
    A --> A1["A: bảng dataset"]
    A --> A2["B: độ nhạy theo k, p"]
    A --> A3["C: MCC"]
    A --> A4["D: p-value kiểm định"]
```

## Ba đóng góp của paper ([trang 2](../papers/2503.03418v1.pdf#page=2))

1. Đề xuất cách tạo mẫu Simplicial SMOTE.
2. Chứng minh phương pháp này cải thiện hiệu năng cho nhiều bộ phân loại.
3. Tổng quát hóa Borderline, Safe-level và ADASYN theo cách lấy mẫu trên simplex.

## Phần train và đánh giá ([trang 6](../papers/2503.03418v1.pdf#page=6))

- Bộ phân loại: k-NN với tham số mặc định, Gradient Boosting với độ sâu cây tối đa là 2.
- Đánh giá: F1 và MCC, trung bình trên 5 lần lặp của 5-fold cross-validation.
- Kiểm định thống kê: Friedman test với Conover post-hoc.

Ghi chú: sơ đồ này là tóm tắt của mình, không phải nội dung trực tiếp của paper. Các ý được gắn số trang để tiện kiểm tra lại.
