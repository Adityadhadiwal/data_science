

---
 🧠 Project Title: student marks  Prediction Using Machine Learning

### 📌 Overview
This project aims to predict student marks using historical data and machine learning techniques. By identifying patterns in student behavior, the model helps businesses proactively retain student and reduce churn rates.

---

### 🚀 Features
- End-to-end pipeline: data ingestion → preprocessing → modeling → evaluation
- Exploratory Data Analysis (EDA) with visual insights
- Multiple ML models compared (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning with GridSearchCV
- Model evaluation using precision, recall, F1-score, ROC-AUC
- FastAPI endpoint for real-time predictions
- GitHub push protection and secret scanning enabled

---

### 📂 Project Structure
```bash
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── logger.py
├── api/
│   └── app.py
├── models/
│   └── churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---
 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn, xgboost, FastAPI
- **Version Control**: Git & GitHub (with push protection and secret scanning)
- **Deployment**: FastAPI + Uvicorn (local or cloud-ready)

---
 📊 Results
- Best model: XGBoost with ROC-AUC of 0.89
- Feature importance revealed contract type, tenure, and monthly charges as key drivers
- API tested with sample inputs for real-time predictions

---

 🔒 Security & Repo Hygiene
- Secrets removed from history using `git filter-repo`
- `.env` used for sensitive configs
- `.gitignore` includes model artifacts, logs, and environment files
- GitHub push protection enabled to block accidental secret commits

---

📈 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn api.app:app --reload
   ```

---

 📬 Contact
Made with ❤️ by [Aditya](https://github.com/Adityadhadiwal)  
Feel free to reach out for collaboration, feedback, or questions!

---

Want me to tailor this for a different project — maybe one with time series, NLP, or deep learning? Or help you write a `requirements.txt` and `.gitignore` to match? Just say the word.
