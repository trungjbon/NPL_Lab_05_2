# NPL_Lab_05_2

- Bảng so sánh định lượng

| Pipeline | F1-score (Macro) | Test Loss |
| :------- | :------: | :-------: |
| TF-IDF + Logistic Regression | 0.84 | N/A |
| Word2Vec (Avg) + Dense | 0.11 | 3.27 |
| Embedding (Pre-trained) + LSTM | 0.12 | 3.18 |
| Embedding (Scratch) + LSTM | 0.84 | 0.62 |

- Phân tích định tính
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
