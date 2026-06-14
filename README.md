# Kiểm duyệt bình luận độc hại tiếng Việt

Project này xây dựng hệ thống phân loại và kiểm duyệt bình luận tiếng Việt, tập trung vào phát hiện bình luận an toàn, xúc phạm và thù ghét. Ngoài mô hình phân loại, project có demo Streamlit mô phỏng một bảng điều khiển kiểm duyệt với chuẩn hóa văn bản, bằng chứng cụm độc hại, điểm rủi ro và hành động đề xuất.

## Điểm nổi bật

- Phân loại bình luận tiếng Việt bằng PhoBERT sequence classification, kèm đối chiếu trực tiếp với TF-IDF + SVM, BiLSTM và XLM-R khi artifact có sẵn.
- Dùng kết quả Note 8 để so sánh PhoBERT với XLM-R đa ngôn ngữ và đánh giá augmentation trên văn bản nhiễu.
- Chuẩn hóa văn bản nhiễu, teencode, lặp ký tự và cách viết né lọc.
- Tô sáng cụm độc hại (toxic span) từ tài nguyên VIHOS/rules để giải thích vì sao bình luận bị đánh dấu.
- Tính điểm rủi ro minh bạch từ nhãn, độ tin cậy, xác suất lớp, bằng chứng và tín hiệu chuẩn hóa.
- Đề xuất hành động kiểm duyệt: cho phép, cảnh báo, chuyển duyệt thủ công, ẩn, chặn hoặc chuyển xử lý nâng cao.
- Hỗ trợ kiểm duyệt từng bình luận, so sánh nhiều mô hình trên cùng đầu vào, xử lý CSV hàng loạt, kiểm thử độ bền và thống kê kết quả.

## Demo giao diện

Chạy demo Streamlit:

```powershell
streamlit run app.py
```

Nếu muốn chỉ định rõ thư mục project và model:

```powershell
$env:PROJECT_ROOT = "D:\vietnamese-toxic-comment-classification_full_20260610_125602\vietnamese-toxic-comment-classification"
$env:PHOBERT_MODEL_DIR = "$env:PROJECT_ROOT\outputs\models\note08b\phobert_augmented_mixed"
streamlit run app.py
```

Sau khi chạy, mở:

```text
http://127.0.0.1:8501
```

Demo hiện có 5 tab:

- `Kiểm duyệt`: nhập một bình luận hoặc chọn ca demo nhanh, chạy PhoBERT cho quyết định chính, đối chiếu SVM/BiLSTM/XLM-R/PhoBERT, xem nhãn, độ tin cậy, rủi ro, hành động và bằng chứng được tô sáng.
- `Hàng loạt`: tải file CSV mẫu hoặc CSV riêng, chọn cột văn bản và xuất kết quả kiểm duyệt nhiều bình luận.
- `Độ bền nhiễu`: so sánh dự đoán khi bình luận bị bỏ dấu, lặp ký tự, che ký tự, teencode hoặc nhiễu tổng hợp.
- `Dashboard`: xem phân bố nhãn, mức rủi ro, hành động và các bình luận rủi ro cao.
- `Model card`: model card, so sánh SVM/BiLSTM/PhoBERT/XLM-R, độ bền augmentation, vai trò VIHOS, audit artifact/leakage/failure analysis, hình notebook và hạn chế.

## Vai trò của Note 8, XLM-R và VIHOS

- **PhoBERT** là mô hình chính trong demo kiểm duyệt vì kết quả tổng thể tốt hơn trong các bảng đánh giá của project.
- **XLM-R/mô hình đa ngôn ngữ ở Note 8** là baseline nghiên cứu, không phải mô hình triển khai chính. Metric clean của XLM-R thấp hơn PhoBERT khá rõ, nhưng vẫn được đưa vào demo để chứng minh lựa chọn PhoBERT và cho thấy augmentation cải thiện độ bền trên dữ liệu nhiễu như bỏ dấu, lặp ký tự, mixed noise.
- **VIHOS** được dùng cho lớp giải thích toxic span. Từ dữ liệu span và các rule rút ra, app tô sáng cụm độc hại như `ngu`, `con chó`, `rácccc`, đồng thời đo độ phủ bằng chứng trên tập kiểm thử.
- **SVM/BiLSTM/XLM-R** là nhóm đối chiếu/baseline. Bảng demo dùng chúng để xem đồng thuận hoặc lệch nhãn trên cùng một câu; quyết định kiểm duyệt vẫn lấy từ PhoBERT.

## Cài đặt

Tạo môi trường ảo và cài dependency:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Các package chính:

- `streamlit`
- `torch`
- `transformers`
- `pyvi`
- `pandas`
- `numpy`
- `scikit-learn`

Nếu có `underthesea`, app sẽ ưu tiên dùng để tách từ. Nếu không có, app tự fallback sang `pyvi`.

## Artifact cần có để chạy demo

Demo cần model PhoBERT đã huấn luyện ở:

```text
outputs/models/note08b/phobert_augmented_mixed
```

Thư mục model nên có các file như:

```text
config.json
model.safetensors
tokenizer_config.json
vocab.txt
bpe.codes
added_tokens.json
```

App cũng sử dụng các tài nguyên giải thích và kiểm duyệt:

```text
outputs/resources/toxic_span_highlighter_rules.json
outputs/resources/span_explainer_config.json
outputs/resources/toxic_phrases_candidates_train.csv
outputs/resources/moderation_policy.json
outputs/resources/app_model_config.json
outputs/results/note08/multilingual_vs_phobert_comparison.csv
outputs/results/note08/augmentation_vs_original_comparison.csv
outputs/results/vihos_span_baseline_metrics.csv
outputs/results/vihos_phrase_coverage_on_vihsd_test.csv
outputs/results/extension_artifact_status.csv
outputs/results/leakage_overlap_report.csv
outputs/results/note08b/bilstm_failure_summary.csv
outputs/results/note09/moderation_case_analysis.csv
outputs/figures/model_comparison_f1.png
outputs/figures/phobert_confusion_matrix_recheck.png
outputs/figures/note08/robustness_f1_by_noise_type.png
```

Nếu các tài nguyên toxic span không đầy đủ, app vẫn có danh sách cụm độc hại dự phòng để demo phần tô sáng bằng chứng.

Để phần đối chiếu nhiều mô hình chạy đầy đủ, project nên có thêm:

```text
outputs/models/tfidf_vectorizer.joblib
outputs/models/svm_model.joblib
outputs/models/note08b/bilstm_augmented_mixed/bilstm_best.pt
outputs/models/note08/xlmr_augmented_balanced_sample
```

Nếu thiếu SVM, BiLSTM hoặc XLM-R, app vẫn chạy PhoBERT và hiển thị trạng thái của artifact phụ trong bảng đối chiếu. XLM-R là model lớn; trên máy thiếu RAM/pagefile, dòng XLM-R có thể hiện `Không chạy` kèm lý do thay vì làm sập demo.

## Cấu trúc project

```text
.
├── app.py
├── requirements.txt
├── README.md
├── run_streamlit_cloudflare_colab (1).ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   ├── robustness/
│   └── vihos/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_baseline_tfidf_svm.ipynb
│   ├── 03_bilstm_train_evaluate.ipynb
│   ├── 04_phobert_train_evaluate.ipynb
│   ├── 05_results_error_analysis.ipynb
│   ├── 06_extension_data_contract_and_vihos_readiness_local.ipynb
│   ├── 07_vihos_span_explanation_token_model_local.ipynb
│   ├── 08_robustness_normalization_and_augmentation.ipynb
│   └── 09_external_test_moderation_and_app_artifacts_final_expert (1).ipynb
├── src/
│   ├── app_utils.py
│   ├── inference.py
│   ├── moderation.py
│   ├── normalization.py
│   ├── risk_scoring.py
│   └── span_explainer.py
└── outputs/
    ├── models/
    ├── resources/
    ├── results/
    └── figures/
```

## Dữ liệu đầu vào

Notebook tiền xử lý kỳ vọng các file sau trong `data/raw`:

```text
train_raw.csv
val_raw.csv
test_raw.csv
```

Mỗi file nên có tối thiểu:

```text
free_text
label_id
```

Với tab `CSV` trong demo, file tải lên nên có một trong các cột:

```text
text
comment
content
original_text
```

Nếu không có các tên cột trên, app sẽ mặc định dùng cột đầu tiên.

## Pipeline notebook

Nếu cần chạy lại toàn bộ quá trình huấn luyện và đánh giá, chạy notebook theo thứ tự:

1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_baseline_tfidf_svm.ipynb`
3. `notebooks/03_bilstm_train_evaluate.ipynb`
4. `notebooks/04_phobert_train_evaluate.ipynb`
5. `notebooks/05_results_error_analysis.ipynb`
6. `notebooks/06_extension_data_contract_and_vihos_readiness_local.ipynb`
7. `notebooks/07_vihos_span_explanation_token_model_local.ipynb`
8. `notebooks/08_robustness_normalization_and_augmentation.ipynb`
9. `notebooks/08b_bilstm_phobert_augmentation_and_failure_analysis_colab_final (1).ipynb`
10. `notebooks/09_external_test_moderation_and_app_artifacts_final_expert (1).ipynb`

Các notebook ghi kết quả vào:

```text
outputs/models
outputs/results
outputs/resources
outputs/figures
```

## Chạy trên Colab với Cloudflare Tunnel

File:

```text
run_streamlit_cloudflare_colab (1).ipynb
```

dùng để chạy app Streamlit trên Colab và mở public URL qua Cloudflare Tunnel. Notebook này không tạo `app.py`; cần đặt sẵn `app.py` ở project root.

Notebook thực hiện các bước chính:

- Mount Google Drive.
- Kiểm tra `PROJECT_ROOT`, `app.py` và thư mục PhoBERT model.
- Cài package cần thiết.
- Chạy `streamlit run app.py --server.port 8501`.
- Chạy `cloudflared tunnel --url http://localhost:8501`.
- In URL dạng `https://...trycloudflare.com`.

## Kiểm thử nhanh

Kiểm tra cú pháp Python:

```powershell
python -m py_compile app.py
```

Kiểm tra app có phản hồi:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8501
```

Một case demo nên thử:

```text
bạn thật ngu ngốc
```

Kỳ vọng giao diện hiển thị:

- Nhãn: `HATE`
- Mức rủi ro: `Nghiêm trọng`
- Hành động: `Chặn hoặc chuyển duyệt`
- Bằng chứng: tô sáng cụm công kích như `ngu`

Với các câu viết lạ hoặc né lọc, ví dụ kéo dài ký tự, bỏ dấu hoặc biến dạng từ, lớp bằng chứng chạy theo 3 tầng:

1. Tìm toxic span trên văn bản gốc bằng VIHOS/rules.
2. Nếu chưa thấy, chuẩn hóa văn bản rồi tìm lại toxic span.
3. Nếu model vẫn dự đoán độc hại nhưng VIHOS/rules chưa khớp cụm chắc chắn, app hiển thị `tín hiệu gợi ý` từ phần bị chuẩn hóa, token nhiễu hoặc token có khả năng đóng góp vào dự đoán.

Tầng thứ 3 giúp demo giải thích các câu khó hơn, nhưng được ghi rõ là tín hiệu gợi ý, không phải bằng chứng toxic span chắc chắn.

## Kết quả và hình ảnh QA

Trong quá trình kiểm thử demo, ảnh chụp màn hình được lưu ở:

```text
qa_screenshots/
```

Một số ảnh QA cuối:

```text
qa_screenshots/24_final_review_result.png
qa_screenshots/25_case_weird_evidence_fixed.png
qa_screenshots/26_model_comparison_added.png
qa_screenshots/27_model_comparison_visible.png
qa_screenshots/28_case_raw_label_and_fuzzy_evidence.png
qa_screenshots/30_multimodel_comparison_table.png
qa_screenshots/31_note8_vihos_role_model_tab.png
qa_screenshots/32_multimodel_with_xlmr.png
qa_screenshots/34_tabs_purpose_batch.png
qa_screenshots/35_tabs_purpose_model_card.png
qa_screenshots/41_final_demo_quick_cases.png
qa_screenshots/42_final_demo_model_status.png
qa_screenshots/43_final_demo_sample_csv.png
qa_screenshots/44_model_card_notebook_output_audit.png
qa_screenshots/45_fallback_evidence_signal.png
qa_screenshots/21_audit_batch_csv.png
qa_screenshots/18_audit_robustness.png
qa_screenshots/19_audit_analytics.png
qa_screenshots/20_audit_model_card.png
```

## Lưu ý diễn giải

- Hệ thống hỗ trợ kiểm duyệt, không thay thế hoàn toàn người duyệt.
- Các ca ranh giới, có ảnh hưởng cao hoặc liên quan quyết định nhạy cảm nên được con người xem lại.
- Tiếng lóng và cách viết né lọc trong tiếng Việt thay đổi nhanh, cần cập nhật dữ liệu và luật bằng chứng định kỳ.
- Tô sáng cụm độc hại giúp giải thích quyết định, nhưng không phải lúc nào cũng bao phủ hết độc hại theo ngữ cảnh.
- Khi giao diện ghi `tín hiệu gợi ý`, đó là fallback giải thích cho câu khó; không nên coi như nhãn span chuẩn từ VIHOS.
- Điểm rủi ro là cơ chế hỗ trợ quyết định, không phải xác suất tuyệt đối của mức nguy hiểm ngoài đời thực.

## Ghi chú phát triển

- Không commit môi trường ảo `.venv`, cache notebook, checkpoint tạm hoặc log chạy local.
- Giữ artifact cần thiết trong `outputs/models` nếu muốn repo có thể chạy demo ngay.
- Khi chỉnh UI, nên kiểm tra cả desktop và mobile.
- Khi chỉnh model hoặc tokenizer, cần kiểm tra lại pipeline chuẩn hóa và tách từ để tránh lệch so với lúc huấn luyện.
