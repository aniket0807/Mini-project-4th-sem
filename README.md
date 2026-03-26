# 🎓 Personalized Study Plan Generator using Linear Regression

> **BCA 4th Semester — AI/ML Mini Project**

An AI-powered web application that analyzes student habits (study hours, sleep, social media usage, attendance, etc.) and predicts academic performance using **Linear Regression**. Based on the predicted score, it generates a **personalized study plan** with actionable recommendations.

---

## 📸 Features

- 🤖 **AI Score Prediction** — Predicts exam score based on 15+ student habit features
- 📋 **Personalized Study Plans** — 3-tier system (Intensive / Moderate / Advanced)
- 💡 **Habit Recommendations** — Specific tips for sleep, study, social media, attendance
- 📊 **Model Insights Dashboard** — Correlation heatmap, feature importance, performance metrics
- 🌙 **Premium Dark Theme** — Modern glassmorphism UI with smooth animations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    STREAMLIT UI                      │
│  (Home | Predict & Plan | Model Insights | About)    │
├─────────────────────────────────────────────────────┤
│              STUDY PLAN GENERATOR                    │
│  (3-tier plans + habit-specific recommendations)     │
├─────────────────────────────────────────────────────┤
│            LINEAR REGRESSION MODEL                   │
│  (Training | Evaluation | Prediction)                │
├─────────────────────────────────────────────────────┤
│                  DATA LAYER                          │
│  (CSV Dataset → Cleaning → Encoding → Features)     │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
anti/
├── dataset/
│   └── student_habits_performance.csv   # Training dataset (1000+ records)
├── saved_model/
│   ├── linear_regression_model.pkl      # Trained model
│   ├── encoders.pkl                     # Label encoders
│   ├── features.pkl                     # Selected feature names
│   ├── metrics.pkl                      # Evaluation metrics
│   └── plots/                           # Generated charts
├── .streamlit/
│   └── config.toml                      # Theme configuration
├── venv/                                # Virtual environment
├── model.py                             # ML pipeline (train, evaluate, predict)
├── study_plan.py                        # Study plan generation logic
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🚀 How to Run Locally

### 1. Clone / Navigate to the project
```bash
cd "d:\Personal Files\BCA\4TH SEM\Mini-project\anti"
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment
```bash
# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the model (first time only)
```bash
python model.py
```
This will train the Linear Regression model and save it to `saved_model/`.

### 6. Launch the web app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🤔 Why Linear Regression?

| Reason | Explanation |
|--------|-------------|
| **Interpretability** | Each coefficient tells us exactly how much each habit affects scores |
| **Continuous Target** | Exam scores (0-100) are continuous — perfect for regression |
| **Linear Relationships** | Study hours & attendance have ~linear effects on performance |
| **Simplicity** | Easy to understand, implement, and explain for a BCA project |
| **Speed** | Trains instantly, predicts in milliseconds |

---

## 📊 Model Performance

| Metric | Description |
|--------|-------------|
| **R² Score** | How much variance the model explains (closer to 1 = better) |
| **MAE** | Average prediction error in marks |
| **MSE** | Mean Squared Error (penalizes large errors) |
| **RMSE** | Root MSE — error in the same unit as scores |

---

## 📋 Study Plan Logic

```
if predicted_score < 50:
    → 🔴 Intensive Improvement Plan (7 hrs/day study schedule)
elif predicted_score <= 75:
    → 🟡 Moderate Improvement Plan (5 hrs/day study schedule)
else:
    → 🟢 Advanced Optimization Plan (4 hrs/day smart study)
```

**Additional recommendations based on habits:**
- 😴 Sleep < 6 hrs → Recommend 7-8 hours
- 📱 Social media > 3 hrs → Recommend reducing usage
- 📖 Study hours < 2 → Suggest increasing study time
- 🏫 Attendance < 75% → Emphasize attending classes
- 🧠 Low mental health → Recommend meditation & counseling

---

## 🛠️ Tech Stack

- **Python 3.x** — Core language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Linear Regression model
- **Matplotlib & Seaborn** — Data visualization
- **Streamlit** — Web application framework
- **Joblib** — Model serialization

---

## 🚀 Future Improvements

- 📈 Try advanced models (Random Forest, XGBoost)
- 🔐 Add user authentication & prediction history
- 📱 Mobile-responsive design / PWA
- 🗃️ Database integration for storing results
- 📧 Email study plans to students
- 🏆 Gamification (badges, achievement streaks)

---

## 📄 License

This project is built for academic/educational purposes as a BCA mini-project.

---

**Built with ❤️ using Python, Scikit-learn & Streamlit**
