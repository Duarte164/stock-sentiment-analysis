# 📈 Stock Sentiment Analysis — Predicting Market Behaviour from Tweets

NLP project that classifies financial tweets as **Bearish**, **Bullish**, or **Neutral**, using classical text-mining techniques and a fine-tuned **BERT** transformer model.

## 🎯 Objective

Financial markets are increasingly influenced by social media sentiment. This project explores whether the language used in finance-related tweets can be used to predict market sentiment (bearish / bullish / neutral), combining traditional NLP preprocessing with a deep learning classifier.

## 🧠 Approach

1. **Exploratory Data Analysis** — class distribution, tweet length, word frequency, word clouds
2. **Text Preprocessing** — ticker symbol removal, URL stripping, stopword removal (keeping key negations like *not*, *no*, *but*), lemmatization
3. **Feature Engineering** — Bag-of-Words and TF-IDF (with bigrams) for baseline comparison
4. **Modelling** — fine-tuned **BERT** (`bert-base-uncased`) for sequence classification
5. **Evaluation** — accuracy and macro F1-score on a stratified validation split

## 📊 Results

| Metric | Score |
|---|---|
| Training Accuracy | 0.97 |
| Validation Accuracy | 0.83 |
| Validation Macro F1-score | 0.77 |

The dataset showed class imbalance (tweets skew neutral/bullish), which was accounted for using a stratified train/validation split and macro-averaged F1 as the primary evaluation metric — accuracy alone would be misleading for the minority class.

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **NLP** — NLTK (tokenization, stopwords, lemmatization), WordCloud
- **ML** — scikit-learn (CountVectorizer, TfidfVectorizer, train/test split)
- **Deep Learning** — TensorFlow / Hugging Face Transformers (BERT)
- **Visualisation** — matplotlib, seaborn

## 📁 Repository Structure

```
stock-sentiment-analysis/
├── notebook/
│   └── stock_sentiment_analysis.ipynb   # full analysis & modelling
├── data/
│   └── README.md                        # dataset source & access instructions
├── outputs/
│   └── predictions_sample.csv           # sample of model predictions
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/stock-sentiment-analysis.git
cd stock-sentiment-analysis
pip install -r requirements.txt
jupyter notebook notebook/stock_sentiment_analysis.ipynb
```

## 📌 Dataset

Financial tweets labelled by sentiment (Bearish / Bullish / Neutral). See `data/README.md` for source and access details.

## 🔍 Key Takeaways

- Transformer-based models (BERT) substantially outperform Bag-of-Words/TF-IDF baselines on short, informal financial text
- Domain-specific preprocessing (e.g. removing `$TICKER` symbols, preserving negation words) materially affects downstream model quality
- Macro F1, not raw accuracy, is the right metric under class imbalance

## 👤 Author

**Duarte Queiroz Miguel** — [LinkedIn](https://www.linkedin.com/in/duarte-queiroz-miguel-1b626a273/)
