## 📰 Fake News Detection (Real/Fake)
---
"Spot fake news before it spots you!"

Fake news spreads fast, but now we can fight back. This project uses Machine Learning and Natural Language Processing (NLP) to automatically detect whether a news article is real or fake.

## 🚀 Features
---
  🔹Smart Text Analysis – Cleans and processes news text using NLP techniques like tokenization, lemmatization, and stopword removal.
  
  🔹Feature Extraction – Turns words into meaningful numbers with TF-IDF and embeddings.
  
  🔹Multiple ML Models – Logistic Regression, Naive Bayes, and Support Vector Machines to find the best predictor.
  
  🔹Performance Metrics – Accuracy, precision, recall, and F1-score to evaluate model strength.
  
  🔹Optional Web Interface – Test real-time news detection with a simple web app.

🛠️ Technologies Used
---
  🔸Python – The language of choice for AI
  
  🔸Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn

📂 Dataset
----
The dataset contains news articles labeled as Real or Fake

Dataset : [Kaggle](https://www.kaggle.com/datasets/fillerink/mohanlal-mammooty-images)

◦ True\

◦ Fake\

## ⚡ How to Run in Google Colab
---
1. Open this notebook in Colab: [Open in Colab](https://colab.research.google.com/drive/12m4cbxG3Qv7gxyAWWtsN-R4RQW3WmXre) ↗

2. Make sure you have **internet connection**.

3. Run the notebook cells in order. Libraries will be installed automatically:

```python
!pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

4.Train, evaluate, and predict fake news—all within Colab!

💡 How It Works
---
  1. Preprocessing – Cleans the text and removes noise.
  2. Feature Extraction – Converts text to numbers ML models can understand.
  3. Prediction – ML model predicts if news is Real ✅ or Fake ❌.
  
🌟 Results
---
  ✨Model Accuracy: ~97%
  <img width="1014" height="145" alt="Accuracy" src="https://github.com/user-attachments/assets/4e7c8046-5046-4d2e-a163-ca7263cf0823" />
  <img width="1270" height="391" alt="Result" src="https://github.com/user-attachments/assets/e8afdd27-2d57-4813-9de1-cb0048e032a6" />


## 🚀 Future Enhancements
---
⚡ Performance Optimization – Reduce prediction time for faster results on large datasets.

🌐 API Integration – Provide an API so other applications can use the fake news detection model.

🛡️ Spam & Bot Detection – Integrate with social media to detect bot-generated fake news.

📊 Topic Modeling – Automatically categorize fake news by topic for better analysis.

🤖 Continuous Learning – Update the model automatically with new data to improve accuracy over time.

## 📜 License
---
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

