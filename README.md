# 1.	Tìm hiểu, so sánh các phương pháp Optimizer trong huấn luyện mô hình học máy
## 1.1	  Optimizer là gì ?
Trong mô hình học máy, "optimizer" là một thành phần quan trọng của quá trình huấn luyện. Optimizer là một thuật toán được sử dụng để tối ưu hóa hàm mất mát (loss function) bằng cách điều chỉnh các tham số của mô hình. Mục tiêu của quá trình này là tìm ra các giá trị tham số mà giảm thiểu hàm mất mát, từ đó cải thiện khả năng dự đoán của mô hình trên dữ liệu mới.
Cụ thể, trong quá trình huấn luyện, mô hình học máy tạo ra dự đoán và so sánh chúng với giá trị thực tế từ dữ liệu đào tạo bằng cách sử dụng hàm mất mát. Sau đó, optimizer được sử dụng để điều chỉnh các tham số của mô hình sao cho giảm thiểu giá trị của hàm mất mát.
## 1.2	  Các thuật toán tối ưu
### 1.2.1	Gradient Descent
Gradient Descent có thể được coi là đứa trẻ nổi tiếng trong lớp tối ưu hóa. Thuật toán tối ưu hóa này sử dụng phép tính để sửa đổi các giá trị một cách nhất quán và đạt được mức tối thiểu cục bộ.
Công thức tính gradient descent:

![image](https://github.com/nguyenbathanh37/Final-ML-Question1/assets/89689892/ffb6808e-4a0b-4e99-abbf-bead5f72d2d7)

Ở đây alpha là kích thước bước biểu thị khoảng cách di chuyển so với từng gradient với mỗi lần lặp.
Gradient descent thực hiện theo các bước sau:
-	Nó bắt đầu với một số hệ số, xem chi phí của chúng và tìm kiếm giá trị chi phí thấp hơn giá trị hiện tại.
-	Nó di chuyển về phía trọng số thấp hơn và cập nhật giá trị của các hệ số.
-	Quá trình lặp lại cho đến khi đạt được mức tối thiểu cục bộ. Mức tối thiểu cục bộ là một điểm vượt quá mức đó nó không thể tiếp tục.

**Ưu điểm:**
-	Cơ bản, dễ hiểu
-	Giải quyết được vấn đề tối ưu bằng cách cập nhật trọng số sau mỗi vòng lặp

**Nhược điểm:**
-	Phụ thuộc vào nghiệm khởi tạo ban đầu.
-	Tốc độ học quá lớn sẽ khiến cho thuật toán không hội tụ.
-	Không tốt đối với tập dữ liệu lớn

### 1.2.2	Stochastic Gradient Descent
Thay vì lấy toàn bộ tập dữ liệu cho mỗi lần lặp như Gradient Descent thì Stochastic Gradient Descent sẽ chọn ngẫu nhiên một vài mẫu từ tập dữ liệu.

Công thức: 

![image](https://github.com/nguyenbathanh37/Final-ML-Question1/assets/89689892/24ab57ee-4f04-41c7-90a2-476f99a1f371)

Quy trình đầu tiên là chọn các tham số ban đầu w và tốc độ học n. Sau đó xáo trộn ngẫu nhiên dữ liệu ở mỗi lần lặp để đạt mức tối thiểu gần đúng. 

**Ưu điểm:**
-	Giải quyết được đối với tập dữ liệu lớn

**Nhược điểm:**
-	Có thể hội tụ chậm và dễ bị mắc kẹt ở các điểm tối ưu cục bộ

### 1.2.3	Stochastic Gradient Descent with Momentum
Momentum giúp hàm mất mát hội tụ nhanh hơn bằng cách giảm dần độ dốc ngẫu nhiên dao động giữa một trong hai hướng của độ dốc và cập nhật trọng số tương ứng.

**Ưu điểm:**
-	Giải quyết được vấn đề GD không tiến được tới điểm global minimum mà chỉ dừng lại ở local minimum

**Nhược điểm:**
-	Khi gần tới điểm global minimum vẫn mất khá nhiều thời gian trước khi dừng hẳn

### 1.2.4	Mini Batch Gradient Descent
Trong biến thể giảm độ dốc này, thay vì lấy tất cả dữ liệu huấn luyện, chỉ một tập hợp con của tập dữ liệu được sử dụng để tính hàm mất mát.

**Ưu điểm:**
-	Giảm phức tạp tính toán so với SGD và tận dụng tốt các tài nguyên tính toán

**Nhược điểm:**
-	Vẫn có thể bị mắc kẹt ở các điểm cục bộ

### 1.2.5	Adagrad
Thuật toán giảm độ dốc thích ứng hơi khác so với các thuật toán giảm độ dốc khác. Điều này là do nó sử dụng các tốc độ học tập khác nhau cho mỗi lần lặp. Sự thay đổi tốc độ học phụ thuộc vào sự khác biệt của các tham số trong quá trình huấn luyện. Các tham số càng thay đổi thì tốc độ học càng thay đổi nhỏ. Việc sửa đổi này rất có lợi vì các bộ dữ liệu trong thế giới thực chứa các tính năng thưa thớt cũng như dày đặc. Vì vậy, không công bằng khi có cùng một giá trị tốc độ học cho tất cả các đặc tính. Thuật toán Adagrad sử dụng công thức dưới đây để cập nhật trọng số. Ở đây alpha(t) biểu thị tốc độ học khác nhau ở mỗi lần lặp, n là hằng số và E là giá trị dương nhỏ để tránh chia cho 0.

![image](https://github.com/nguyenbathanh37/Final-ML-Question1/assets/89689892/10695e31-a453-47c5-b5b5-562ece55738c)

**Ưu điểm:**
-	Hiệu suất tốt cho những mô hình có tham số thưa thớt

**Nhược điểm:**
-	Có thể điều chỉnh learning rate nhanh dẫn tới việc hội tụ sớm

### 1.2.6	RMSprop
RMSprop giải quyết vấn đề tỷ lệ học giảm dần của Adagrad bằng cách chia tỷ lệ học cho trung bình của bình phương gradient.

![image](https://github.com/nguyenbathanh37/Final-ML-Question1/assets/89689892/47b4cc99-4b0e-4289-8bdd-96b49bbdea4f)

**Ưu điểm:**
-	Giải quyết được vấn đề tốc độ học giảm dần của Adagrad

**Nhược điểm:**
-	Có thể chỉ cho kết quả nghiệm chỉ là local minimum chứ không đạt được global minimum như momentum.

### 1.2.7	AdaDelta
AdaDelta có thể được coi là phiên bản mạnh mẽ hơn của trình tối ưu hóa AdaGrad. Nó dựa trên phương pháp học tập thích ứng và được thiết kế để giải quyết những hạn chế đáng kể của trình tối ưu hóa hỗ trợ AdaGrad và RMS. Vấn đề chính với hai trình tối ưu hóa ở trên là tốc độ học ban đầu phải được xác định theo cách thủ công. Một vấn đề khác là tốc độ học giảm dần, tại một thời điểm nào đó nó trở nên cực kỳ nhỏ. Do đó, sau một số lần lặp nhất định, mô hình không thể học được kiến thức mới nữa.

**Ưu điểm:**
-	Giảm vấn đề của Adagrad bằng cách giữ learning rate ổn định

**Nhược điểm:**
-	Đòi hỏi lưu trữ thông tin lớn và có thể cần điều chỉnh hyperparameters

### 1.2.8	Adam
Adam là sự kết hợp của Momentum và RMSprop. Nếu giải thích theo hiện tượng vật lí thì Momentum giống như 1 quả cầu lao xuống dốc, còn Adam như 1 quả cầu rất nặng có ma sát, vì vậy nó dễ dàng vượt qua local minimum tới global minimum và khi tới global minimum nó không mất nhiều thời gian dao động qua lại quanh đích vì nó có ma sát nên dễ dừng lại hơn.

**Ưu điểm:**
-	Hiệu suất tốt trong nhiều loại dữ liệu, tự điều chỉnh learning rate cho từng tham số

**Nhược điểm:**
-	Cần lưu trữ thêm thông tin (Momentum và RMSprop) nên yêu cầu bộ nhớ lớn
