# 📰 Fake News Detection using DistilBERT, RoBERTa, and Ensemble Learning

## 📘 Overview
This project detects **fake news** using advanced **transformer-based NLP models** — specifically **DistilBERT** and **RoBERTa** — fine-tuned for binary text classification.  
An **ensemble method** combines predictions from both models for improved accuracy and generalization.

---

## 📂 Dataset
The dataset consists of labeled news articles, preprocessed and tokenized using Hugging Face Transformers.  
You can replace it with any binary-labeled text dataset (e.g., `FakeNewsNet`, `LIAR`, or Kaggle Fake News Dataset).

---

## 🧠 Model Architecture
- **DistilBERT**: A lightweight version of BERT, faster but still high-performing.  
- **RoBERTa**: A robustly optimized BERT variant for better language understanding.  
- **Ensemble Model**: Combines both model logits for more stable predictions.

---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run preprocessing & tokenization
```python
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("your_dataset_name")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

### 2. Train the models
```python
trainer1.train()
trainer2.train()
```

### 3. Ensemble predictions
```python
texts = ["Breaking: Major policy changes announced...", "Scientists confirm alien life."]
predictions = ensemble_predict(texts, model1, tokenizer1, model2, tokenizer2)
print(predictions)
```

---

## 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── raw/               # Original dataset files
│   ├── real/              # Preprocessed real news
│   └── fake/              # Preprocessed fake news
│
├── models/
│   ├── distilbert/        # Saved DistilBERT fine-tuned weights
│   └── roberta/           # Saved RoBERTa fine-tuned weights
│
├── notebooks/
│   └── FakeNews_Analysis.ipynb   # Main notebook for training & analysis
│
├── requirements.txt
├── README.md
└── ensemble_predict.py
```

---

## 📊 Results (Example)
| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| DistilBERT | 0.96 | 0.95 |
| RoBERTa | 0.97 | 0.96 |
| Ensemble | **0.98** | **0.97** |

---

## 🧾 Citation
If you use this project or any part of its code, please cite:
```
@software{fake_news_detection,
  author = {Your Name},
  title = {Fake News Detection using Transformer Models},
  year = {2025},
  url = {https://github.com/yourusername/fake-news-detection}
}
```

---

## 💡 Future Work
- Add multilingual fake news detection.
- Integrate explainable AI (SHAP/Grad-CAM).
- Deploy the model as an interactive web app using Streamlit.

---

## 👩‍💻 Author
**Your Name**  
AI/ML Engineer | NLP Researcher  
[GitHub](https://github.com/yourusername)
