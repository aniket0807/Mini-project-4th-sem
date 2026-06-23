# 🎓 StudyGenie — AI-Powered Study Intelligence

> **BCA 4th Semester — AI/ML & Web Integration Mini Project**

StudyGenie is a premium, feature-rich web application designed to help students optimize their learning workflow. It analyzes daily study habits to predict exam scores using **XGBoost** (tuned via **Optuna**), automatically extracts topics from syllabus PDFs using the **Google Gemini API**, schedules reviews via the **SM-2 Spaced Repetition** algorithm, gamifies learning with XP and badges, and delivers personalized HTML study plans via **SMTP Email**.

---

## 📸 Features

- 🔮 **XGBoost AI Score Prediction** — Uses an optimized XGBoost model to predict exam scores based on 15+ student habits (study hours, sleep, screen time, attendance, diet, etc.).
- 📄 **Gemini PDF Syllabus Parser** — Upload a syllabus PDF to automatically extract course units and topics using `gemini-2.5-flash` (with a multi-model fallback chain and a regex-heuristic local parser fallback).
- 🧠 **SM-2 Spaced Repetition** — Input subjects/topics to get structured flashcard-style review intervals with daily due items and retention analytics (Avg Ease, Mastered, Mastery Ratio).
- 🏆 **Gamified Achievements** — Earn XP for logging study sessions and completing reviews. Track your level progress, active streak, and unlock 12+ achievement badges.
- 📧 **SMTP Email Plan Delivery** — Generates a beautiful HTML study plan complete with daily schedules, weekly goals, and syllabus topics, and emails it directly to the user.
- 🔑 **Secure Authentication** — Full user register/login wall powered by SQLite and `bcrypt` password hashing.
- 🎨 **Premium Glassmorphism UI** — Stunning dark/light mode responsive dashboard with interactive charts, sleek animations, and tailored color schemes.

---

## 🏗️ System Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │                 STREAMLIT UI                 │
                      │  (Home | Predict | Review | Badges | Insights)│
                      └──────┬──────────────┬──────────────┬─────────┘
                             │              │              │
        ┌────────────────────▼───┐  ┌───────▼──────┐  ┌────▼─────────────────┐
        │  GEMINI SYLLABUS API   │  │  XGBOOST AI  │  │  SPACED REPETITION   │
        │  (PDF Text → JSON)     │  │  (Prediction)│  │  (SM-2 Scheduler)    │
        └────────────────────┬───┘  └───────┬──────┘  └────┬─────────────────┘
                             │              │              │
        ┌────────────────────▼──────────────▼──────────────▼─────────────────┐
        │                        SQLITE / CSV DATABASE                       │
        │      (User Profiles | Streaks & XP | SM-2 Reviews | Dataset)       │
        └───────────────────────────────────┬────────────────────────────────┘
                                            │
                                    ┌───────▼──────┐
                                    │  SMTP EMAIL  │
                                    │  (HTML Plan) │
                                    └──────────────┘
```

---

## 📁 Folder Structure

```
anti/
├── dataset/
│   └── student_habits_performance.csv   # Training dataset (1000+ records)
├── saved_model/
│   ├── xgboost_model.pkl                # Trained XGBoost pipeline
│   ├── encoders.pkl                     # Category label encoders
│   ├── features.pkl                     # Selected feature list
│   ├── metrics.pkl                      # XGBoost evaluation metrics
│   └── plots/                           # Correlation/Feature Importance plots
├── app.py                               # Main Streamlit web application
├── auth.py                              # SQLite & bcrypt user authentication
├── database.py                          # SQLite connection & path helpers
├── email_service.py                     # SMTP HTML email template & sender
├── gamification.py                      # Streaks, Level ups, XP & Badges logic
├── model.py                             # XGBoost & Optuna training pipeline
├── requirements.txt                     # Python packages list
├── spaced_repetition.py                 # SM-2 Spaced Repetition engine
├── study_plan.py                        # Rule-based study plan generator
├── syllabus_parser.py                   # Gemini API & Regex syllabus extractor
└── README.md                            # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone / Navigate to the project
```bash
cd "d:\Personal Files\BCA\4TH SEM\anti"
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
EMAIL_SENDER=your_gmail_sender@gmail.com
EMAIL_PASSWORD=your_gmail_app_password_here
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
APP_SECRET=your_secret_string
```

### 4. Train the AI Model
```bash
python model.py
```
This runs an Optuna search to find the best hyperparameters, trains the final XGBoost model, evaluates metrics against a baseline Linear Regression model, and saves the binary artifacts and plots.

### 5. Launch the Web Application
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

---

## ⚙️ Methodology

The StudyGenie workflow operates through a structured system pipeline:

```
  [ PDF Syllabus ]       [ User Habit Inputs ]      [ User Review Recall (0-5) ]
         │                        │                             │
         ▼                        ▼                             ▼
 ┌───────────────┐        ┌───────────────┐             ┌───────────────┐
 │ PyMuPDF Text  │        │Categorical Enc│             │  SM-2 Spaced  │
 │  Extraction   │        │(Ordinal Code) │             │  Repetition   │
 └───────┬───────┘        └───────┬───────┘             └───────┬───────┘
         │                        │                             │
         ▼                        ▼                             ▼
 ┌───────────────┐        ┌───────────────┐             ┌───────────────┐
 │Gemini LLM /   │        │ Optuna-Tuned  │             │ SQLite DB     │
 │Regex Parser   │        │ XGBoost Model │             │ Updates       │
 └───────┬───────┘        └───────┬───────┘             └───────┬───────┘
         │                        │                             │
         └─────────┬──────────────┘                             │
                   ▼                                            ▼
        ┌────────────────────┐                        ┌────────────────────┐
        │Personalised Study  │                        │ XP Logs, level-ups │
        │ Plan Generator     │                        │ & Badge Triggering │
        └──────────┬─────────┘                        └────────────────────┘
                   │
                   ▼
         [ SMTP Email Dispatch ]
```

### 1. Data Processing & ML Pipeline
* **Data Ingestion**: The system reads student habit surveys (`student_habits_performance.csv`) containing 1,000+ records and 15+ behavioral attributes.
* **Feature Processing**:
  - Drops identifier columns (`student_id`) and duplicates.
  - Imputes missing categorical items using mode values.
  - Applies scikit-learn `OrdinalEncoder` within an inference pipeline to parse non-numeric columns.
* **Hyperparameter Optimization**:
  - Integrates **Optuna** to execute hyperparameter search trials (tuning learning rate, tree depth, subsampling rates, and regularisation weights).
  - Validation metrics evaluate Root Mean Squared Error (RMSE) at each trial, selecting the model configurations that generalise best to unseen test datasets.

### 2. Syllabus Parsing Engine (Hybrid AI/Deterministic Pipeline)
* **Text Extraction**: PyMuPDF reads raw text streams from user-uploaded PDFs.
* **LLM Orchestration**:
  - The client formats prompt guidelines instructing the model to yield structured, valid JSON matching a predefined curriculum schema.
  - A fallback model sequence checks model availability in order of preference (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.0-flash-lite` → `gemini-1.5-flash`).
  - Catches 503/UNAVAILABLE or rate-limiting errors to seamlessly shift down the model list.
* **Deterministic Fallback**: If all API keys or models fail, a regex-based heuristic extractor parses units and topic hierarchies locally to guarantee continuous operation.

### 3. Study Plan Synthesis & Email Service
* Predicted scores are categorized into study plan tiers (Intensive, Moderate, or Advanced).
* Rules examine habit deficits (e.g. low sleep or high social media hours) to overlay tailored recommendations.
* An SMTP client compiles the plan metrics, daily schedules, weekly goals, and extracted syllabus topics into a responsive HTML email and dispatches it.

### 4. Spaced Repetition (SM-2) & Gamification Engine
* **Spaced Repetition**: Tracks topic memory retention via SQLite database queries. Users grade their recall (0-5), updating the repetitions count, current interval, and ease factor.
* **Gamification Logic**: Study logs and completed reviews query SQLite databases to calculate current streak counts, award XP, compute level caps, and check badge criteria.

---

## 🔬 Model & Algorithms

### Why XGBoost?
We migrated from **Linear Regression** to **XGBoost Regressor** to handle:
- **Non-linear interactions** (e.g., how sleep quality moderates high study hours).
- **Automatic regularisation** (L1/L2 penalties) to prevent overfitting on habit surveys.
- **Optuna Tuning** — Automates hyperparameter optimization (max depth, learning rate, subsampling, etc.) to minimize validation RMSE.

### SM-2 Spaced Repetition
Calculates review intervals using:
$$I(1) = 1,\quad I(2) = 6,\quad I(n) = I(n-1) \times EF$$
Where the Ease Factor ($EF$) adjusts based on user recall quality ($q \in [0, 5]$):
$$EF' = EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))$$

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Custom HTML/CSS (Glassmorphism, Dark/Light mode)
- **Database**: SQLite (Auth, Spaced Repetition, Gamification)
- **Machine Learning**: XGBoost, Optuna, Scikit-learn
- **GenAI**: Google Gemini API (`google-genai` client SDK)
- **Visualization**: Matplotlib, Seaborn
- **Security**: bcrypt password hashing

---

## 🔮 Future Steps & Roadmap

To evolve StudyGenie into a complete academic companion, the following enhancements are planned:
- 💬 **Syllabus-aware RAG Chat** — Add a conversational chatbot allowing students to upload study material and ask questions tailored specifically to their parsed syllabus topics (using FAISS vector store).
- 📈 **Retention Decay Curves** — Implement graphical visualization of user forgetfulness curves using the SM-2 data to prompt reviews more interactively.
- 📆 **Calendar Integrations** — Synchronize generated schedules and study session goals directly with Google Calendar or Outlook APIs.
- 🔔 **Web Push Notifications** — Real-time browser push notifications or SMS alerts notifying users when spaced repetition cards are due for review.
- 👥 **Peer Leaderboards** — Introduce student study groups, shared study rooms, and public high-score leaderboards based on XP and streak achievements.

---

**Built with ❤️ for BCA 4th Semester Mini Project**
