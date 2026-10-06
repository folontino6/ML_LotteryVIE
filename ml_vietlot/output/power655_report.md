# Báo cáo phân tích Power 6/55

## 1. Tổng quan dữ liệu
- Tên game: Power 6/55
- Số kỳ hợp lệ: 1405
- Số kỳ đánh giá walk-forward: 200
- Khoảng thời gian: n/a -> n/a
- Mức cảnh báo: mô hình và tín hiệu thống kê chỉ phục vụ mục đích xếp hạng, không làm thay đổi xác suất lý thuyết của một vé ngẫu nhiên.

## 2. Kiểm tra dữ liệu
- Dữ liệu đầu vào được kiểm tra theo định dạng số, khoảng giá trị, trùng lặp, thời gian, và tính liên tục theo kỳ quay.
- Đánh giá dữ liệu: p-value lớn nghĩa là chưa thấy bằng chứng tần suất lệch khỏi phân phối đều.
- Tất cả kết luận đều dựa trên dữ liệu lịch sử thực tế của game này, không suy diễn từ các game khác.

## 3. Xác suất lý thuyết
- Xác suất một số xuất hiện trong mỗi lượt chọn: 0.109091
- Xác suất kiểm tra cặp số: 0.010101
- Xác suất kiểm tra bộ ba số: 0.000762
- Xác suất đúng một vé: 0.000000034495
- Quy tắc toán học được giữ nguyên: mô hình không thay đổi xác suất lý thuyết của một trò chơi công bằng. Model score chỉ là điểm xếp hạng, không phải xác suất thắng.

## 4. Phân tích thống kê

### 4.1 Tần suất
- Số xuất hiện nhiều nhất: 22(179), 41(177), 51(172), 29(170), 08(169), 34(169)
- Số xuất hiện ít nhất: 04(126), 28(129), 06(132), 37(132), 15(134), 02(138)
- Kiểm định chi-bình phương: 0.5613856086635174
- Tần suất lịch sử là chỉ dấu tham khảo về xu hướng có thể hiện, nhưng không phản ánh xác suất tăng lên đột ngột ở một kỳ sau chỉ vì số đó chưa ra lâu.

### 4.2 EWMA
- Các thang thời gian hiệu chỉnh trọng số: ew10, ew30, ew100, ew300
- EWMA giúp nhận diện lượng nhịp số dần thay đổi theo thời gian; các giá trị này chỉ nên được dùng như feature hỗ trợ cho mô hình chứ không được hiểu là xác suất được đảm bảo.

### 4.3 Gap / Recency
- Khoảng cách hiện tại dài nhất: [{'number': 12, 'current_gap': 28.0, 'average_gap': 7.832258064516129, 'median_gap': 6.0, 'max_gap': 33.0, 'recent_appearance_rate': 0.1110320284697509}, {'number': 20, 'current_gap': 21.0, 'average_gap': 7.890322580645162, 'median_gap': 6.0, 'max_gap': 47.0, 'recent_appearance_rate': 0.1110320284697509}, {'number': 30, 'current_gap': 21.0, 'average_gap': 8.622377622377622, 'median_gap': 6.0, 'max_gap': 36.0, 'recent_appearance_rate': 0.10249110320284698}]
- Tỷ lệ xuất hiện gần đây trung bình: 0.10909090909090911
- Gap và recency có thể tạo thêm tín hiệu, nhưng chỉ khi nó cải thiện đánh giá walk-forward; không được suy ra rằng số quá hạn chắc chắn sẽ ra.

### 4.4 Markov
- Overlap trung bình với kỳ trước: 0.10992402659069327
- Xác suất xuất hiện sau khi đã xuất hiện ở kỳ trước: 0.06875
- Markov cung cấp thông tin về sự phụ thuộc ngắn hạn trong dữ liệu, nhưng không thay đổi bản chất ngẫu nhiên của trò chơi.

### 4.5 Pair
- Top cặp quan sát: [{'pair': [1, 2], 'count': 17, 'expected_count': 14.191919191919192, 'z_score': 0.7491928016337982}, {'pair': [1, 3], 'count': 20, 'expected_count': 14.191919191919192, 'z_score': 1.5495894278396907}, {'pair': [1, 4], 'count': 12, 'expected_count': 14.191919191919192, 'z_score': -0.5848015753760224}]
- Mức hỗ trợ cặp số: 14.191919191919192
- Pair được dùng như một tín hiệu bổ trợ cho đánh giá tổ hợp; nó không tạo ra xác suất chiến thắng tuyệt đối.

### 4.6 Triple
- Top bộ ba quan sát: [{'triple': [np.int16(11), np.int16(22), np.int16(28)], 'count': 7}, {'triple': [np.int16(8), np.int16(36), np.int16(39)], 'count': 7}, {'triple': [np.int16(21), np.int16(29), np.int16(41)], 'count': 7}]
- Triple phù hợp hơn ở tầng đánh giá tổ hợp hơn là báo cáo xác suất cho từng số riêng lẻ.

### 4.7 Structural
- Tổng trung bình của 6 số: 169.39074733096086
- Phạm vi trung bình: 39.75160142348754
- Đặc trưng cấu trúc được dùng để chấm điểm bộ số, không được hiểu như toàn bộ xác suất của từng số.

### 4.8 Monte Carlo
- Baseline Monte Carlo: Monte Carlo provides a random baseline for structural comparisons; it does not imply a higher winning probability.
- Monte Carlo phục vụ đánh giá ngẫu nhiên và kiểm định thống kê, không đồng nghĩa với xác suất trúng thực sự.

### 4.9 Kiểm định thống kê
- Bảng hiệu chỉnh nhiều phép thử: [{'name': 'frequency_chi_square', 'raw_p': 0.5613856086635174, 'bh_adjusted_p': 1.0}, {'name': 'gap_current_gap', 'raw_p': 1.0, 'bh_adjusted_p': 1.0}, {'name': 'markov_overlap', 'raw_p': 1.0, 'bh_adjusted_p': 1.0}]
- Kết quả raw p-value và p-value sau hiệu chỉnh được phân tách rõ ràng để tránh suy đoán quá mức.

### 4.10 Multiple-testing correction
- Benjamini-Hochberg (BH) được áp dụng trên các phép thử được lựa chọn: [1.0, 1.0, 1.0, 1.0, 1.0]
- Việc hiệu chỉnh nhiều phép thử giúp đánh giá sự đáng tin cậy của các tín hiệu thống kê trước khi đưa vào ensemble.

## 5. Machine Learning
- Mô hình học máy đánh giá từng số theo tính năng lịch sử, tín hiệu thống kê và tương tác số học giữa các số trong cùng kỳ.
- Quá trình huấn luyện và dự đoán phải tuân theo walk-forward: chỉ dùng dữ liệu trước thời điểm đánh giá.

## 6. Walk-forward validation
- Kết luận walk-forward: Không có chiến lược nào vượt ngẫu nhiên một cách có ý nghĩa thống kê (sau hiệu chỉnh nhiều phép thử). Điều này phù hợp với bản chất ngẫu nhiên của xổ số. Chất lượng xác suất từng số: Random Forest AUC=0.4994; Hồi quy Logistic AUC=0.4959 (0.5 = ngẫu nhiên). Phân vị trung bình của kết quả thật theo điểm tổ hợp: 0.499 (0.5 = ngẫu nhiên).
- Mỗi mô hình được so sánh với baseline ngẫu nhiên dựa trên số khớp thực tế trên các kỳ đánh giá liên tiếp theo thời gian.

## 7. So sánh các mô hình
- So sánh ít nhất: Random baseline, ML hiện tại, Statistical-only, ML + statistical features, Final ensemble.
- Mô hình hiện tại được tổng kết trong: ['Random Forest', 'Hồi quy Logistic', 'Tần suất 100 kỳ', 'Số quá hạn (delay)', 'Balanced Ensemble', 'ML Focus', 'Statistical Focus', 'Diversified', 'Alternative', 'Ngẫu nhiên']

## 8. Ablation test
- Kiểm tra từng thành phần chính như Frequency, EWMA, Gap, Markov, Pair, Structural và so sánh với Base ML để xác định phần nào thực sự hữu ích.
- Nếu một feature làm xấu đi hiệu quả walk-forward, feature đó không được giữ trong ensemble cuối cùng.

## 9. Ensemble
- Ensemble cuối cùng chỉ giữ những thành phần đã được đánh giá hữu ích qua dữ liệu lịch sử.
- Trọng số được xác định từ thông tin có sẵn trong từng bước walk-forward, không dùng dữ liệu tương lai.

## 10. Ranking các số
- Số xếp hạng tốt nhất: 40, 08, 54, 52, 55, 10, 48, 07, 30, 44.
- Việc xếp hạng chỉ thể hiện mức độ ưu tiên trong mô hình, không phải xác suất trúng thực sự của mỗi số.

## 11. Các bộ số ứng viên
- Bộ số đề xuất: - Random Forest: 08 10 40 52 54 55
- Hồi quy Logistic: 12 34 39 40 44 55
- Tần suất 100 kỳ: 01 05 07 08 09 16
- Số quá hạn (delay): 12 19 20 30 39 50
- Balanced Ensemble: 08 30 39 44 50 53
- ML Focus: 01 05 07 08 16 22
- Statistical Focus: 05 07 08 11 13 18
- Diversified: 07 08 11 13 16 18
- Alternative: 30 39 40 44 50 53
- Mỗi vé phải là bộ số duy nhất, hợp lệ với game và được sắp xếp theo thứ tự tăng dần.

## 12. Monte Carlo đánh giá bộ số
- Monte Carlo được dùng để đối chiếu với baseline ngẫu nhiên, phân vị tổ hợp, và kiểm định độ tin cậy của các chỉ số cấu trúc.
- Kết quả bộ số không được diễn giải như xác suất trúng jackpot được tăng lên.

## 13. Kết luận
- Dữ liệu lịch sử cho thấy game này vẫn thuộc khuôn khổ xác suất ổn định của trò chơi ngẫu nhiên.
- Kết hợp thống kê + ML mang lại các dấu hiệu hữu ích cho xếp hạng, nhưng không thay đổi quy luật toán học nền tảng của xổ số.
- Nỗ lực kỹ thuật tập trung vào: dữ liệu sạch, thống kê đáng tin cậy, walk-forward minh bạch, ensemble hợp lý và cảnh báo đúng mức.

## 14. Hạn chế
- Xổ số là biến cố ngẫu nhiên; không có phương pháp nào có thể dự đoán chắc chắn.
- Kết quả mô hình phụ thuộc vào dữ liệu lịch sử, điểm chốt và đặc trưng mà dự án chọn.
- Một số tín hiệu thống kê có thể tương quan yếu hoặc bị làm méo do nhiều phép thử; nên kiểm chứng trên dữ liệu mới liên tục.
