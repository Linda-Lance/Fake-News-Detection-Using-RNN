## 📰 Fake News Detector
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

⚡ How to Run in Google Colab
----
1. Open the notebook: Fake News Detection Notebook

Make sure you have internet connection to install dependencies.

2. Run the cells in order. The notebook will automatically install required libraries using:

!pip install pandas numpy scikit-learn nltk matplotlib seaborn


That’s it! The model will train, evaluate, and predict fake news directly in Colab.

💡 How It Works
---
  1. Preprocessing – Cleans the text and removes noise.
  2. Feature Extraction – Converts text to numbers ML models can understand.
  3. Prediction – ML model predicts if news is Real ✅ or Fake ❌.
  
🌟 Results
---
  ✨Model Accuracy: ~97%
  
<img width="1014" height="145" alt="Screenshot 2025-09-21 195428" src="https://github.com/user-attachments/assets/a611f3fd-a33d-48da-9dcc-a4618781836c" />
<img width="1270" height="391" alt="Screenshot 2025-09-21 195450" src="https://github.com/user-attachments/assets/d8d5e0c2-9a74-465b-9e6e-d928e0378228" />

📜 License
---
MIT License – check the LICENSE file for details.
