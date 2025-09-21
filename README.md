# 📰Fake News Detection Using RNN

This project implements a **fake news detection system** using a Recurrent Neural Network (RNN) with Keras and TensorFlow. The model is trained on a dataset containing news articles labeled as *real* or *fake*, and predicts whether a given news article is genuine or fake.

---

## 📊Dataset

- The project uses two CSV files:
  - `True.csv` — contains real news articles✅
  - `Fake.csv` — contains fake news articles❌
- Each dataset contains the columns:
  - `title` — title of the news📝
  - `text` — content of the news📰
- Labels are assigned as follows:
  - `1` → Real news✅
  - `0` → Fake news❌
- The datasets are concatenated, shuffled, and preprocessed for training.

---

## ✨Features

- Combines `title` and `text` into a single `content` column.
- Text cleaning and preprocessing:
  - Converts to lowercase🔤
  - Removes URLs🌐
  - Removes non-word characters❌
  - Removes extra spaces➖
  - Removes stopwords🛑
  - Lemmatizes words🧠
- Converts text into sequences of integers using **Tokenization**🔢.
- Pads sequences to a fixed length for model input📏.

---

## ⚙️Installation

1. Clone the repository:
```bash
git clone <repository_link>
cd fake-news-detection
```

2. Install required Python packages:
```bash
pip install pandas numpy tensorflow scikit-learn nltk
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## ▶️How to Run

1. Load the dataset (`True.csv` and `Fake.csv`) into the `/content/` directory.
2. Run the Python script (`code.ipynb`) or Jupyter Notebook.
3. The code will:
   - Preprocess the dataset🧹
   - Train the RNN model🏋️‍♂️
   - Evaluate the accuracy on the test set📈
4. Make predictions on new news articles by providing raw text✍️.

---

## 🏗️Model Architecture

- **Embedding Layer** — Converts words to dense vectors (`input_dim=5000`, `output_dim=64`)📚
- **SimpleRNN Layer** — 120 units, processes sequential data🔄
- **Dropout Layers** — Reduces overfitting (0.5 and 0.3)💧
- **Dense Layer** — 64 neurons with ReLU activation⚡
- **Output Layer** — Single neuron with sigmoid activation for binary classification🎯
- **Loss function:** Binary Crossentropy🔻
- **Optimizer:** Adam⚙️
- **Metrics:** Accuracy✅  

The model is trained for **5 epochs** with a batch size of 64🕒.

---

## 🖥️Usage Example

```python
# Sample news text
sample_news = """Jeffrey Toobin chime protect Hillary Clinton poor taste mostly untrue..."""
cleaned = clean_text(sample_news)
seq = tokenizer.texts_to_sequences([cleaned])
padded = pad_sequences(seq, maxlen=200)
prediction = model.predict(padded)

if prediction < 0.5:
    print("Fake news")
else:
    print("Real news")
```

---

## 🚀Result

Training Accuracy : 97%
<img width="1014" height="145" alt="Accuracy" src="https://github.com/user-attachments/assets/9067a877-42ec-44fd-8502-327da972b109" />
<img width="1270" height="391" alt="Result" src="https://github.com/user-attachments/assets/994bea81-be0d-4567-b6f3-0601732df1f5" />


## Future Enhancements

- Increase dataset size for better accuracy📈.
- Experiment with LSTM or GRU layers for improved sequence learning🔄.
- Add sentiment analysis features😊😡.
- Deploy the model as a web application using Flask or Streamlit🌐.
- Implement multi-class classification for different types of misinformation🕵️‍♂️.

---

## 👩‍💻Author

**Linda Lance** 
[LinkedIn](https://www.linkedin.com/in/linda--lance/) 
[GitHub](https://github.com/Linda-Lance)

---

## 📄License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


