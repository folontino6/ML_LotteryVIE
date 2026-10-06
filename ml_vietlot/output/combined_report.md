# Báo cáo tổng hợp các game

## 1. Tóm tắt hỗ trợ game
- power655: đã phát hiện dữ liệu tại power655.csv; đánh giá thực nghiệm được thực hiện theo mô hình chung.
- mega645: đã phát hiện dữ liệu tại mega645.csv; đánh giá thực nghiệm được thực hiện theo mô hình chung.
- lotto535: đã phát hiện dữ liệu tại lotto535.csv; đánh giá thực nghiệm được thực hiện theo mô hình chung.
- lotto645: Dữ liệu của game này hiện chưa có trong thư mục dự án nên chưa thực hiện đánh giá thực nghiệm.

## 2. Kiến trúc chung
- Tất cả game sử dụng cùng một kiến trúc: kiểm tra dữ liệu -> thống kê tham chiếu -> feature engineering -> mô hình ML -> ensemble -> walk-forward -> ranking -> đánh giá tổ hợp.
- Kiến trúc này cho phép game khác nhau có cấu hình riêng về số lượng số, không gian số, và hằng số lý thuyết, đồng thời giữ nguyên quy trình phân tích chung.

## 3. Phương pháp thống kê và ML
- Xác suất lý thuyết: tính theo cấu hình từng game, giữ rõ ràng giữa probability và model score.
- Phân tích thống kê: tần suất, EWMA, gap, recency, Markov, pair, triple, structural, Monte Carlo, và hiệu chỉnh nhiều phép thử.
- Mô hình ML: train/test theo walk-forward và ưu tiên đánh giá không rò rỉ dữ liệu tương lai.

## 4. Kết quả theo game
### power655
- Dữ liệu: power655.csv
- Nếu cần, báo cáo chi tiết có thể xem tại file tương ứng trong cùng thư mục output.
### mega645
- Dữ liệu: mega645.csv
- Nếu cần, báo cáo chi tiết có thể xem tại file tương ứng trong cùng thư mục output.
### lotto535
- Dữ liệu: lotto535.csv
- Nếu cần, báo cáo chi tiết có thể xem tại file tương ứng trong cùng thư mục output.
### lotto645
- Dữ liệu của game này hiện chưa có trong thư mục dự án nên chưa thực hiện đánh giá thực nghiệm.

## 5. Giới hạn
- Một game vẫn là trò chơi ngẫu nhiên; mô hình không làm thay đổi xác suất toán học của vé fair-play.
- Chỉ các game có dữ liệu thực tế mới được đánh giá; các game thiếu dữ liệu không được tạo dữ liệu giả.

## 6. Kết luận
- Kiến trúc chung đang ở trạng thái hoạt động với dữ liệu có sẵn và sẵn sàng cho game mới khi thêm CSV tương ứng.