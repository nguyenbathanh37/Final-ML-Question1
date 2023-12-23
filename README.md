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

# 2.	Tìm hiểu về Continual Learning và Test Production khi xây dựng một giải pháp học máy để giải quyết một bài toán nào đó
## 2.1	  Continual Learning
### 2.1.1	Continual Learning là gì ?
Học liên tục là ý tưởng về việc cập nhật mô hình khi dữ liệu mới xuất hiện; điều này giúp mô hình duy trì sự tương ứng với phân phối dữ liệu hiện tại.
### 2.1.2	Tại sao phải Continual Learning ?
Lý do cơ bản là giúp mô hình bắt kịp với sự thay đổi trong phân phối dữ liệu. Có một số trường hợp sử dụng trong đó việc thích ứng nhanh chóng với sự thay đổi trong phân phối là quan trọng.
Ví dụ :
-	Các trường hợp sử dụng trong đó có thể xảy ra những thay đổi đột ngột và nhanh chóng. Có thể có một buổi concert ở một khu vực ngẫu nhiên vào một ngày Thứ Hai ngẫu nhiên, và "mô hình ML giá Thứ Hai" có thể không được trang bị đầy đủ để xử lý nó.
-	Các trường hợp sử dụng trong đó không thể có được dữ liệu đào tạo cho một sự kiện cụ thể. Một ví dụ về điều này là mô hình thương mại điện tử trong ngày Black Friday. Rất khó để thu thập dữ liệu lịch sử để dự đoán hành vi người dùng trong Black Friday, vì vậy mô hình của bạn phải thích ứng trong suốt ngày.
-	Các trường hợp mà mô hình phải dự đoán cho một người dùng mới mà không có dữ liệu lịch sử hoặc dữ liệu đã lỗi thời
### 2.1.3	Stateless retraining VS Stateful training
#### 2.1.3.1	Stateless retraining
Đào tạo lại mô hình từ đầu mỗi lần, sử dụng trọng số được khởi tạo ngẫu nhiên và dữ liệu mới hơn.
-	Có thể có một số sự chồng chéo với dữ liệu đã được sử dụng để đào tạo phiên bản mô hình trước đó.
-	Hầu hết bắt đầu thực hiện học liên tục bằng cách sử dụng quá trình đào tạo không lưu trạng thái.
#### 2.1.3.2	Stateful training
Khởi tạo mô hình với trọng số từ vòng đào tạo trước đó và tiếp tục đào tạo bằng cách sử dụng dữ liệu mới chưa được nhìn thấy.
-	Điều này cho phép mô hình cập nhật với lượng dữ liệu ít hơn đáng kể
-	Cho phép mô hình hội tụ nhanh hơn và sử dụng ít công suất tính toán hơn
-	Đôi khi sẽ cần chạy quá trình đào tạo không lưu trạng thái với một lượng lớn dữ liệu để hiệu chỉnh lại mô hình
-	Đào tạo có lưu trạng thái chủ yếu được sử dụng để tích hợp dữ liệu mới vào một kiến trúc mô hình hiện tại và cố định (tức là lặp lại dữ liệu). Nếu muốn thay đổi các đặc trưng hoặc kiến trúc của mô hình, sẽ cần thực hiện một vòng đào tạo không lưu trạng thái trước đó.
### 2.1.4	Những thách thức của Continual Learning
#### 2.1.4.1	Truy cập dữ liệu mới
Nếu muốn cập nhật mô hình mỗi giờ, thì ta cần dữ liệu đào tạo chất lượng được gắn nhãn mỗi giờ. Khoảng thời gian cập càng ngắn thì càng thách thức.
-	Tốc độ lưu trữ dữ liệu vào kho dữ liệu
-	Tốc độ ghi nhãn
#### 2.1.4.2	Đánh giá
Việc áp dụng việc học tập liên tục như một phương pháp thực hành có nguy cơ dẫn đến những thất bại của mô hình. Việc cập nhật mô hình càng thường xuyên thì càng có nhiều cơ hội để mô hình thất bại.
#### 2.1.4.3	Mở rộng quy mô dữ liệu
Tính toán tính năng thường yêu cầu Chia tỷ lệ. Chia tỷ lệ yêu cầu quyền truy cập vào số liệu thống kê dữ liệu global như tối thiểu, tối đa, trung bình và phương sai.
Một kỹ thuật phổ biến để thực hiện việc này là tính toán hoặc ước tính các thống kê này tăng dần khi quan sát dữ liệu mới
#### 2.1.4.4	Thuật toán
Thách thức này xuất hiện khi chúng ta sử dụng một số loại thuật toán nhất định và muốn cập nhật chúng rất nhanh.
Thử thách chỉ xảy ra khi chúng ta cần cập nhật chúng thật nhanh vì không thể chờ thuật toán xem hết toàn bộ tập dữ liệu.
### 2.1.5	Bốn giai đoạn của Continual Learning
#### 2.1.5.1	Đào tạo lại thủ công, không lưu trạng thái
Các mô hình chỉ được đào tạo lại khi đáp ứng hai điều kiện:
-	hiệu suất của mô hình đã suy giảm đến mức hiện tại nó gây hại nhiều hơn là có lợi
-	có thời gian để cập nhật mô hình
#### 2.1.5.2	Đào tạo lại tự động không lưu trạng thái theo lịch trình cố định
Giai đoạn này thường xảy ra khi các mô hình chính của một lĩnh vực đã được phát triển và do đó, ưu tiên của chúng ta không còn là tạo ra các mô hình mới, mà là duy trì và cải tiến những mô hình hiện tại.
#### 2.1.5.3	Đào tạo có lưu trạng thái tự động theo lịch trình cố định
Để đạt được điều này, chúng ta cần cấu hình lại đoạn mã và cách theo dõi dòng dữ liệu cũng như mô hình
#### 2.1.5.4	Continual learning
Trong giai đoạn này, phần lịch trình cố định của các giai đoạn trước được thay thế bằng một số cơ chế kích hoạt đào tạo lại. Các tác nhân kích hoạt có thể là:
-	Thời gian
-	Hiệu suất
-	Khối lượng
-	Sự trôi dạt
### 2.1.6	Tần suất cập nhật mô hình
Để trả lời câu hỏi này, trước tiên chúng ta cần hiểu và xác định mức lợi ích bạn nhận được khi cập nhật mô hình của mình với dữ liệu mới. Càng đạt được nhiều thì càng phải đào tạo lại thường xuyên.
### 2.1.7	Khi nào thì thực hiện lặp lại mô hình
Hầu hết chương này cho đến nay đều đề cập đến việc cập nhật mô hình với dữ liệu mới. Tuy nhiên, trong thực tế, đôi khi chúng ta cũng có thể cần thay đổi kiến trúc mô hình của mình.
## Test Production
Để kiểm thử đầy đủ mô hình trước khi đưa chúng ra sử dụng rộng rãi, bạn cần cả các đánh giá offline trước triển khai VÀ kiểm thử trong sản xuất. Đánh giá offline một mình không đủ.
Lý tưởng nhất, mỗi nhóm đều đặt ra một quy trình rõ ràng về cách mô hình được đánh giá: những kiểm thử nào được chạy, ai chạy chúng và ngưỡng áp dụng để thăng mô hình lên giai đoạn tiếp theo. Điều tốt nhất là nếu các quy trình đánh giá này được tự động hóa và bắt đầu khi có cập nhật mô hình mới. Các bước thăng cấp nên được xem xét tương tự như cách CI/CD được đánh giá trong kỹ thuật phần mềm.
### 2.2.1	Đánh giá ngoại tuyến trước khi triển khai
Có hai cách phổ biến nhất:
-	Sử dụng phần tách thử nghiệm để so sánh với đường cơ sở
-	Chạy thử nghiệm ngược
### 2.2.2	Thử nghiệm trong chiến lược sản xuất
#### 2.2.2.1	Triển khai bóng
Là việc triển khai một phiên bản mới của mô hình mà không ảnh hưởng đến quá trình sản xuất chính. Phiên bản mới này thường được triển khai song song với mô hình hiện tại và nhận được dữ liệu đầu vào thực tế, nhưng kết quả dự đoán lại không được sử dụng hoặc ảnh hưởng đến quyết định cuối cùng

**o	Ưu điểm:**
-	Đây là cách triển khai an toàn nhất cho mô hình. Ngay cả khi mô hình mới có lỗi, các dự đoán sẽ không được phục vụ
-	Đây là một khái niệm đơn giản
-	Cuộc thử nghiệm sẽ thu thập đủ dữ liệu để đạt được ý nghĩa thống kê nhanh hơn so với tất cả các chiến lược khác vì tất cả các mô hình đều nhận toàn bộ lưu lượng.
-	
**o	Nhược điểm:**
-	Kỹ thuật này không thể được sử dụng khi đo lường hiệu suất của mô hình phụ thuộc vào quan sát cách người dùng tương tác với các dự đoán
-	Kỹ thuật này tốn kém vì nó làm tăng gấp đôi số lượng dự đoán và do đó làm tăng số lượng tính toán cần thiết
#### 2.2.2.2	Thử nghiệm A/B
A/B testing là một phương pháp thử nghiệm thường được sử dụng trong việc đánh giá hiệu suất của các mô hình trong môi trường sản xuất. Phương pháp này giúp so sánh giữa hai (hoặc nhiều hơn) phiên bản của mô hình hoặc hệ thống để xem phiên bản nào mang lại hiệu suất tốt hơn.

**o	Ưu điểm:**
-	Vì dự đoán được cung cấp cho người dùng nên kỹ thuật này cho phép nắm bắt đầy đủ cách người dùng phản ứng với các mô hình khác nhau.
-	Thử nghiệm A/B rất dễ hiểu và có rất nhiều thư viện cũng như tài liệu xung quanh nó.
-	Việc chạy này rẻ vì chỉ có một dự đoán cho mỗi yêu cầu.
-	Chúng ta sẽ không cần phải xem xét các trường hợp đặc biệt phát sinh từ các yêu cầu suy luận song song cho các chế độ dự đoán trực tuyến

**o	Nhược điểm:**
-	Nó kém an toàn hơn so với triển khai bóng. 
#### 2.2.2.3	Phát hành Canary
Canary Release là một chiến lược triển khai phần mềm trong môi trường sản xuất, bao gồm cả triển khai và kiểm thử mô hình máy học, để giảm rủi ro và đảm bảo tính ổn định của hệ thống. 

**o	Ưu điểm:**
-	Dễ hiểu
-	Đơn giản nhất trong tất cả các chiến lược để triển khai 
-	So với việc triển khai bóng thì việc chạy sẽ rẻ hơn
-	Nếu kết hợp với thử nghiệm A/B, nó cho phép thay đổi linh hoạt lượng lưu lượng truy cập mà mỗi mô hình đang sử dụng.

**o	Nhược điểm:**
-	Nó mở ra khả năng không khắt khe trong việc xác định sự khác biệt về hiệu suất.
-	Đây được cho là lựa chọn kém an toàn nhất nhưng lại rất dễ bị hủy bỏ.
#### 2.2.2.4	Thí nghiệm xen kẽ
Là một phương pháp thử nghiệm thường được sử dụng để so sánh hiệu suất giữa hai hoặc nhiều mô hình trong môi trường sản xuất. Phương pháp này được thiết kế để giảm thiểu các yếu tố ngoại vi và đảm bảo sự công bằng trong việc so sánh bằng cách xen kẽ các trải nghiệm giữa các nhóm.

**o	Ưu điểm:**
-	Netflix đã phát hiện bằng thực nghiệm rằng việc xen kẽ sẽ xác định mô hình tốt nhất một cách đáng tin cậy với kích thước mẫu nhỏ hơn đáng kể so với thử nghiệm A/B truyền thống.
-	Ngược lại với việc triển khai theo dõi, chiến lược này cho phép nắm bắt cách người dùng hành xử trái với dự đoán.

**o	Nhược điểm:**
-	Việc triển khai phức tạp hơn thử nghiệm A/B
-	Mất nhiều thời gian
-	Nó tăng gấp đôi sức mạnh tính toán cần thiết vì mọi yêu cầu đều nhận được dự đoán từ nhiều mô hình.
