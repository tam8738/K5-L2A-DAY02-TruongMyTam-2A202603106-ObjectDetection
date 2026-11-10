# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: TruongMyTam
- MSSV: 2A202603106
- Email: 26ai.tamtm2@vinuni.edu.vn
- Lớp: L2A
- Link repo bài làm: https://github.com/tam8738/K5-L2A-DAY02-TruongMyTam-2A202603106-ObjectDetection
- Link Colab đã chạy:https://colab.research.google.com/drive/1sLC_JxrkPx4OO5_FJAar2rESR4R8CJG2?usp=sharing

## 2. Schema và mini guideline

Dẫn tới [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md). Schema sử dụng bốn lớp: `car=0`, `truck=1`, `bus=2`, `van=3`. Box bám sát phần xe nhìn thấy; xe bị che hoặc cắt khung vẫn được gán nếu còn đủ thông tin nhận diện. Ba ca mơ hồ đã ghi trong guideline gồm minivan, xe tải bị cắt/nhòe và xe buýt bị cắt khung. Các trường hợp không chắc chắn về `van`/`car`/`bus` cần giảng viên xác nhận.

## 3. Annotation và self-QC

- Số file nhãn train: 50
- Kết quả checker và lỗi đã xử lý: Có đủ 50 file YOLO cùng basename với ảnh train. Khi chạy trên Windows, checker gặp lỗi `UnicodeEncodeError` lúc in thông báo tiếng Việt; cần chạy lại với `PYTHONIOENCODING=utf-8` để lưu kết quả hiển thị rõ ràng.
- Danh sách ít nhất 10 ảnh đã rà bằng mắt: `drive_002.jpg`, `drive_008.jpg`, `drive_009.jpg`, `drive_012.jpg`, `drive_019.jpg`, `drive_030.jpg`, `drive_034.jpg`, `drive_036.jpg`, `drive_045.jpg`, `drive_050.jpg`.
- Kết quả self-QC: cả 10 ảnh đều có file nhãn tương ứng; không có file nhãn bị thiếu; các dòng nhãn có class ID hợp lệ `0, 1, 2, 3`; không phát hiện lỗi định dạng rõ ràng.
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: kiểm tra bằng file `.txt` cho phép xác nhận sự tồn tại của box, số box và class ID, nhưng không thể khẳng định box ôm sát vật thể tuyệt đối nếu không xem overlay. Không ghi nhận lỗi định dạng hoặc class ID trong lần rà này; việc kiểm tra độ sát box cần đối chiếu ảnh với annotation overlay trong CVAT.
- Commit khóa nhãn ban đầu: `a022880` — Add initial YOLO train annotations

## 4. Gold QC và rework

Kết quả so sánh ba lần đánh giá sau khi sửa nhãn:

| Chỉ số | Lần 1 | Lần 2 | Lần 3 mới nhất |
|---|---:|---:|---:|
| Precision | 0.4353 | 0.4460 | **0.4662** |
| Recall | 0.7167 | 0.7152 | **0.6968** |
| F1-score | 0.5417 | 0.5494 | **0.5586** |
| Mean IoU | 0.8482 | 0.8483 | **0.8506** |
| True Positive | 468 | 467 | 455 |
| False Positive | 607 | 580 | 521 |
| False Negative | 185 | 186 | 198 |
| Tổng box | 1.075 | 1.047 | 976 |

So với lần 1, lần 3 có F1 tăng từ `0.5417` lên `0.5586`, Mean IoU tăng nhẹ từ `0.8482` lên `0.8506`, False Positive giảm từ `607` xuống `521` và tổng số box giảm. Recall giảm nhẹ từ `0.7167` xuống `0.6968`, đồng thời False Negative tăng; điều này cho thấy việc giảm box thừa có thể làm bỏ sót thêm một số box. Theo ngưỡng giảng viên thông báo (`F1 > 0.5`), lần 3 đạt yêu cầu.

Dẫn tới [annotation_evaluation.json](../../outputs/annotation_evaluation.json). File JSON trong repo có thể vẫn chứa kết quả của một lần chạy trước; khi chốt bài cần bảo đảm JSON được cập nhật đúng với Lần 3 hoặc ghi rõ lịch sử các lần đánh giá. Không sửa gold labels.

## 5. Kết quả YOLO

- Runtime, GPU/CPU, phiên bản Ultralytics và Torch: Colab chạy với `device=0` theo cấu hình train; tên GPU và phiên bản Ultralytics/Torch chưa được lưu trong `model_eval_metrics.json`, cần bổ sung nếu có log runtime.
- Các tham số thay đổi so với notebook nếu có và lý do: Không thay đổi tham số chính; notebook train `yolo26n.pt`, 50 epochs, `imgsz=640`, dùng `eval` làm validation, không dùng eval để train.
- Đường dẫn best checkpoint trong phiên chạy: `/content/Day2-Lab/outputs/yolo26_runs/vehicle_annotation/weights/best.pt` (chỉ ghi đường dẫn, không commit file `.pt`).

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.3156 |
| Recall | 0.3066 |
| mAP50 | 0.2740 |
| mAP50–95 | 0.2006 |

Đã đặt hai folder kết quả tại [eval_predictions/](assets/eval_predictions/) và [yolo26_runs/](assets/yolo26_runs/). Kết quả được tính trên 10 ảnh eval; có thể xem ví dụ [drive_051.jpg](assets/eval_predictions/visuals/drive_051.jpg) và các biểu đồ trong [vehicle_eval/](assets/yolo26_runs/vehicle_eval/).

Trả lời bốn câu hỏi cuối notebook:

1. Lớp nào recall thấp nhất? Ảnh nào minh họa? Dựa trên confusion matrix normalized, `van` là lớp yếu nhất vì ô đúng lớp trên đường chéo gần như không xuất hiện; `truck` cũng có recall thấp. Ảnh [drive_051.jpg](assets/eval_predictions/visuals/drive_051.jpg) minh họa một lỗi phân lớp: một xe con ở tiền cảnh bị dự đoán thành `van` với confidence `0.99`.
2. Nếu mAP50 cao nhưng mAP50–95 thấp, cần kiểm tra box có bám sát vật thể không, đặc biệt là tọa độ bốn cạnh; mAP50 cho phép sai lệch rộng hơn còn các ngưỡng IoU cao yêu cầu box chính xác hơn.
3. Lỗi class nào có thể liên quan guideline? Lỗi `car` bị dự đoán thành `van` có thể liên quan đến ranh giới hình dáng giữa xe con và van/minivan trong guideline. Dữ kiện hỗ trợ là confusion matrix có nhầm lẫn giữa các lớp xe và ảnh [drive_051.jpg](assets/eval_predictions/visuals/drive_051.jpg), trong đó xe con tiền cảnh bị gán `van`; cần đối chiếu lại quy tắc phân biệt thân xe cao/hình hộp của `van` với xe con thông thường.
4. Không dùng 10 ảnh eval để train để giữ đánh giá khách quan và tránh rò rỉ dữ liệu. Notebook dùng chúng làm validation trong quá trình train và chọn checkpoint tốt nhất; vì vậy đây không phải test set độc lập.

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Gold QC | F1=0.5699, mean IoU=0.8547 | Đối chiếu ngưỡng lớp F1>0.5; evaluator repo hiển thị `passed=false` do dùng ngưỡng 0.80 |
| Colab/assets | Chưa có metrics YOLO trong workspace | Đã chép ảnh eval và log vào `assets/`; cần bổ sung 4 metrics và link Colab |

## 7. Tự đánh giá theo rubric

Đã có 50 nhãn YOLO, mini guideline, Gold QC và kết quả Colab/assets. Còn cần bổ sung link Colab, email, danh sách 10 ảnh self-QC, 4 metrics YOLO và nhận xét theo ảnh thực tế trước khi nộp.

## 8. Công cụ AI và tài liệu tham khảo

Đã dùng Google Colab để train/evaluate YOLO và VS Code/AI để đọc guideline, kiểm tra cấu trúc báo cáo và hỗ trợ diễn đạt. Các nhãn và kết luận dữ liệu cần được tự kiểm tra trước khi nộp. Không ghi mật khẩu/token và không đính kèm gold train.
