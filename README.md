# 1. Các bước triển khai
## Data Preparation
- Đọc dữ liệu:
  + Dùng pandas.read_csv() để load train, validation và test set.
  + Đặt tên cột là text và intent.
- Label Encoding:
  + Dùng LabelEncoder để biến nhãn text thành nhãn số (intent_label).
  + Số lớp (num_classes) được xác định từ số nhãn unique.

## Model 1: TF-IDF + Logistic Regression
- Pipeline:
  + Dùng TfidfVectorizer(max_features=5000) để chuyển text thành TF-IDF vector.
  + LogisticRegression với max_iter=1000 cho multi-class classification.

## Model 2: Word2Vec (Avg) + Dense Neural Network
- Train Word2Vec:
  + Tokenize text (lowercase, split words).
  + Huấn luyện Word2Vec với vector_size=50, window=4, min_count=2.
- Chuyển câu thành vector trung bình:
  + sentence_to_avg_vector() tính trung bình vector các từ trong câu.
  + Nếu câu không có từ trong vocab, dùng vector 0.
- Xây dựng model Dense:
  + Input: kích thước embedding 50.
  + Dense units 128 + ReLU + Dropout 0.5.
  + Output layer: softmax.
 
## Model 3: Pretrained Word2Vec + LSTM
- Tokenization & Padding:
  + Tokenizer với num_words=2000 và oov_token=UNK.
  + Chuyển câu thành sequences và pad đến max_len=20.
- Prepare embedding layer:
  + Dùng Word2Vec đã train từ model 2.
  + Tạo embedding_matrix dựa trên vocab của tokenizer.
  + Embedding layer: trainable=False (không update embeddings).
- Xây dựng LSTM model:
  + LSTM 128 units, dropout 0.2, recurrent_dropout 0.2.
  + Dense softmax output layer.
 
## Model 4: Embedding train from scratch + LSTM
- Tokenization & Padding:
  + Giống Model 3.
- Embedding layer học từ đầu:
  + embedding_dim=400, trainable=True.
  + Không dùng pretrained embeddings.
- Xây dựng LSTM model:
  + LSTM 128 units, dropout 0.2, recurrent_dropout 0.2.
  + Dense softmax output layer.
 
## Evaluation
- Tất cả model đều được đánh giá bằng:
  + Accuracy trên test set.
  + F1 macro để cân nhắc dataset không cân bằng.
  + Test sample để quan sát dự đoán của mô hình trên câu cụ thể.

# 2. Hướng dẫn chạy code
- Thay đổi đường dẫn data nếu cần thiết.
- Chạy file "Lab_05/src/lab5_text_classification.ipynb" để thấy kết quả chạy chương trình chi tiết.

# 3. Bảng so sánh định lượng

| Pipeline | F1-score (Macro) | Test Loss |
| :------- | :------: | :-------: |
| TF-IDF + Logistic Regression | 0.84 | N/A |
| Word2Vec (Avg) + Dense | 0.11 | 3.27 |
| Embedding (Pre-trained) + LSTM | 0.12 | 3.18 |
| Embedding (Scratch) + LSTM | 0.84 | 0.62 |

- Nhận xét:
  - TF-IDF + Logistic Regression: Khả năng generalization tốt với dữ liệu nhỏ, đơn giản, không cần train mạng nơ-ron.
  - Word2Vec (Avg) + Dense: Trung bình vector từ mất toàn bộ thông tin thứ tự từ, dẫn đến mô hình không học được ngữ cảnh câu.
  - Embedding (Pre-trained) + LSTM: Dù có LSTM, nhưng pre-trained embedding không train lại, nên mô hình không học được đặc trưng cho task hiện tại.
  - Embedding (Scratch) + LSTM: Khi embedding được train từ đầu, LSTM học được đặc trưng task-specific, dẫn đến hiệu suất tương đương TF-IDF + LR.

# 4. Phân tích định tính
- TF-IDF + Logistic Regression:
  - Text: can you remind me to not call my mom
  - -> Predicted intent: calendar_set
  - -> True intent: reminder_create
  - --

  - Text: is it going to be sunny or rainy tomorrow
  - -> Predicted intent: weather_query
  - -> True intent: weather_query
  - --

  - Text: find a flight from new york to london but not through paris
  - -> Predicted intent: general_negate
  - -> True intent: flight_search
 
- Word2Vec (Avg) + Dense
  - Text: can you remind me to not call my mom
  - -> Predicted intent: general_explain
  - -> True intent: reminder_create
  - --

  - Text: is it going to be sunny or rainy tomorrow
  - -> Predicted intent: alarm_query
  - -> True intent: weather_query
  - --

  - Text: find a flight from new york to london but not through paris
  - -> Predicted intent: email_sendemail
  - -> True intent: flight_search
  
- Embedding (Pre-trained) + LSTM
  - Text: can you remind me to not call my mom
  - -> Predicted intent: takeaway_order
  - -> True intent: reminder_create
  - --

  - Text: is it going to be sunny or rainy tomorrow
  - -> Predicted intent: email_query
  - -> True intent: weather_query
  - --

  - Text: find a flight from new york to london but not through paris
  - -> Predicted intent: transport_ticket
  - -> True intent: flight_search

- Embedding (Scratch) + LSTM
  - Text: can you remind me to not call my mom
  - -> Predicted intent: calendar_set
  - -> True intent: reminder_create
  - --

  - Text: is it going to be sunny or rainy tomorrow
  - -> Predicted intent: weather_query
  - -> True intent: weather_query
  - --

  - Text: find a flight from new york to london but not through paris
  - -> Predicted intent: transport_ticket
  - -> True intent: flight_search

- Nhận xét:
  - Chỉ model TF-IDF và Scratch LSTM dự đoán đúng câu số 2 (weather_query).
  - Các câu còn lại đều bị nhầm, đặc biệt câu số 1 và 3.
  - LSTM giữ thứ tự từ -> lý thuyết có thể hiểu “not call” trong câu reminder, hay “from … to … but not through …” trong flight. Tuy nhiên, dữ liệu sample quá ít và embedding chưa đủ mạnh -> mạng không học được các pattern phức tạp.
  - Word2Vec pretrained giúp hiểu nghĩa từ, nhưng không đủ context cho intent phức tạp.
  - Embedding scratch + LSTM có lợi thế học task-specific, nhưng nếu dataset nhỏ thì dễ underfit -> dự đoán sai.
 
# 5. Ưu và nhược của từng phương pháp
- TF-IDF + Logistic Regression:
  - Ưu điểm:
    - Cực nhanh, triển khai đơn giản.
    - Hoạt động tốt với các câu chứa từ khóa trực tiếp.
    - Không cần huấn luyện embedding, ít tốn bộ nhớ.
  - Nhược điểm:
    - Không capture ngữ nghĩa giữa các từ.
    - Không giữ thứ tự từ, không hiểu phủ định hay mối quan hệ phức tạp.
    - Dễ nhầm intent khi câu phức tạp hoặc nhiều entities.
   
- Word2Vec (trung bình) + Dense NN:
  - Ưu điểm:
    - Biểu diễn ngữ nghĩa từ (Word2Vec) -> giúp nhận dạng từ đồng nghĩa.
    - NN học được các mối quan hệ phi tuyến giữa các feature vector.
  - Nhược điểm:
    - Mất thứ tự từ (tính trung bình) -> không hiểu phủ định, quan hệ thứ tự.
    - Dataset nhỏ dễ underfit, dẫn đến dự đoán sai.
    - Thường kém hơn LSTM khi câu dài hoặc phức tạp.
   
- Pretrained Word2Vec + LSTM
  - Ưu điểm:
    - Giữ thứ tự từ -> lý thuyết hiểu ngữ cảnh tốt hơn.
    - Pretrained embeddings giúp mạng nhanh học ý nghĩa từ.
    - Phù hợp cho các câu dài, có nhiều thông tin.
  - Nhược điểm:
    - Dự đoán thực tế chưa chắc đúng nếu dữ liệu training quá ít.
    - Không train embedding task-specific -> một số từ khóa quan trọng cho intent chưa được mạng hiểu chính xác.
    - Huấn luyện lâu hơn model 1 & 2.

- Embedding train từ đầu + LSTM
  - Ưu điểm:
    - Embedding task-specific -> có thể học biểu diễn từ phù hợp dataset.
    - Giữ thứ tự từ -> LSTM hiểu ngữ cảnh.
    - Thích hợp với dữ liệu đủ lớn, các intent phức tạp.
  - Nhược điểm:
    - Dataset nhỏ -> dễ underfit hoặc dự đoán sai.
    - Huấn luyện lâu hơn, tốn tài nguyên hơn model 1 & 2.
    - Embedding mới cần nhiều dữ liệu để học tốt, nếu ít data thì hiệu quả kém.
