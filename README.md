# Email Spam Detector

A machine learning-based email spam detection system built with Python, Flask, and scikit-learn. The model uses a Naive Bayes classifier trained on an email dataset, falling back to the Groq AI API (LLM inference) when the Naive Bayes model has low confidence.

## Features

- **Hybrid Detection:** Uses a local Naive Bayes classifier for fast, offline predictions.
- **AI Fallback:** Submits low-confidence predictions to the Groq API (Llama 3) for deep structural and contextual analysis.
- **RESTful API:** Exposes endpoints to classify single or batched messages via a Flask server.
- **Detailed Explanations:** Provides structural signals (word count, ALL-CAPS, URLs), spam/ham probabilities, risk factors, and human-readable explanations.
- **Web UI:** Includes an `index.html` frontend for interactive testing.

## Prerequisites

- Python 3.8+
- Scikit-learn, Flask, and Pandas
- A valid Groq API key (if you want the AI fallback feature)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohit7739/email_spam.git
   cd email_spam
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   ```

3. **Train the Model:**
   ```bash
   cd backend
   python train.py
   ```
   This reads the `data/spam.csv` dataset, trains the Naive Bayes model, and saves `spam_model.pkl` and `vectorizer.pkl`.

4. **Add your Groq API Key:**
   In `backend/spam_processor.py`, replace `GROQ_API_KEY = "______"` with your actual key if you wish to use the LLM fallback component.

## Running the API Server

1. **Start Flask Server:**
   ```bash
   cd backend
   python app.py
   ```
   The backend will start running on `http://localhost:5000`.

2. **API Endpoints:**
   - `GET /health` - Check if the API is running and model is loaded.
   - `POST /predict` - Send a single text message for analysis.
     ```json
     { "message": "Congratulations! You won a free lottery!" }
     ```
   - `POST /predict/batch` - Send multiple messages at once.
     ```json
     { "messages": ["Hello, how are you?", "URGENT! Transfer money now!"] }
     ```

## Usage (Frontend UI)

Just open `backend/index.html` in your browser. It includes a chat-like interface to enter messages and review detailed spam analysis responses from the API server (must be running).