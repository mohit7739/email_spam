import os
import re
import string
import pickle
import json
import urllib.request
import urllib.error

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'spam_model.pkl')
VECTOR_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# ── Put your Groq API key here ────────────────────────────────────────────────
GROQ_API_KEY = ""
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama3-8b-8192"

# Confidence threshold — below this, Groq will be called
CONFIDENCE_THRESHOLD = 0.70

# ── Spam keyword categories ───────────────────────────────────────────────────
SPAM_KEYWORDS = {
    "Prize / Lottery":    ["free", "win", "winner", "won", "prize", "lucky",
                           "congratulations", "selected", "chosen", "award",
                           "reward", "giveaway", "jackpot"],
    "Money / Financial":  ["cash", "money", "earn", "income", "profit", "rich",
                           "wealth", "investment", "loan", "credit", "bank",
                           "dollar", "million", "billion"],
    "Urgency / Pressure": ["urgent", "hurry", "act now", "call now", "limited",
                           "expires", "last chance", "dont miss", "immediately",
                           "asap", "today only", "fast", "quick"],
    "Offers / Deals":     ["offer", "deal", "discount", "sale", "cheap", "save",
                           "bonus", "extra", "special", "exclusive", "100%",
                           "guarantee", "risk free", "no cost"],
    "Click / Links":      ["click", "visit", "subscribe", "buy", "order",
                           "purchase", "download", "apply", "register",
                           "signup", "join now", "get now", "claim"],
    "Health / Pills":     ["viagra", "pills", "weight loss", "diet",
                           "enlargement", "cure", "pharmacy", "supplement"],
    "Scam / Fraud":       ["nigerian", "prince", "inheritance", "transfer",
                           "confidential", "secret", "overseas", "unclaimed",
                           "beneficiary", "diplomat"],
    "Account / Security": ["password", "verify", "account", "login", "ssn",
                           "social security", "pin", "otp", "confirm",
                           "suspend", "blocked", "locked"],
    "Adult Content":      ["xxx", "adult", "dating", "singles", "nude",
                           "explicit", "18+", "hookup"],
}

HAM_KEYWORDS = [
    "meeting", "schedule", "tomorrow", "today", "call", "talk", "hello",
    "hi", "hey", "dear", "thanks", "thank you", "please", "could you",
    "project", "report", "update", "status", "review", "lunch", "dinner",
    "coffee", "see you", "let me know", "attached", "document", "regards",
    "sincerely", "team", "office", "work", "colleague", "manager"
]


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def deep_analysis(text, label, spam_prob, ham_prob):
    lower = text.lower()
    analysis = {}

    keyword_hits = {}
    for category, words in SPAM_KEYWORDS.items():
        found = [w for w in words if w in lower]
        if found:
            keyword_hits[category] = found
    analysis["spam_keyword_categories"] = keyword_hits

    urls       = re.findall(r'http\S+|www\S+', text, re.IGNORECASE)
    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    exclaims   = text.count('!')
    money_syms = re.findall(r'[\$\£\€]\d+[\d,]*|\d+\s*%', text)
    phone_nums = re.findall(r'\b\d{10,}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', text)
    word_count = len(text.split())
    char_count = len(text)
    cap_ratio  = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    sentences  = re.split(r'[.!?]+', text)
    avg_sent   = word_count / max(len([s for s in sentences if s.strip()]), 1)

    analysis["structural_signals"] = {
        "urls_found":           urls,
        "all_caps_words":       caps_words,
        "exclamation_marks":    exclaims,
        "money_symbols_found":  money_syms,
        "phone_numbers_found":  phone_nums,
        "word_count":           word_count,
        "character_count":      char_count,
        "capital_letter_ratio": round(cap_ratio, 3),
        "avg_sentence_length":  round(avg_sent, 1),
    }

    ham_hits = [w for w in HAM_KEYWORDS if w in lower]
    analysis["ham_signals_found"] = ham_hits

    risk_factors = []
    if keyword_hits:
        total_kw = sum(len(v) for v in keyword_hits.values())
        risk_factors.append(f"{total_kw} spam keyword(s) across {len(keyword_hits)} "
                            f"category/categories: {', '.join(keyword_hits.keys())}")
    if urls:
        risk_factors.append(f"{len(urls)} URL(s) detected: {', '.join(urls[:3])}")
    if caps_words:
        risk_factors.append(f"{len(caps_words)} ALL-CAPS word(s): {', '.join(caps_words[:5])}")
    if exclaims >= 2:
        risk_factors.append(f"{exclaims} exclamation mark(s) — creates false urgency")
    if money_syms:
        risk_factors.append(f"Money/percentage symbols: {', '.join(money_syms[:4])}")
    if phone_nums:
        risk_factors.append(f"Phone number(s) detected: {', '.join(phone_nums[:2])}")
    if cap_ratio > 0.25:
        risk_factors.append(f"High capital letter ratio ({round(cap_ratio*100,1)}%)")
    if not ham_hits:
        risk_factors.append("No normal conversational words found")
    analysis["risk_factors"] = risk_factors

    safe_factors = []
    if ham_hits:
        safe_factors.append(f"Normal conversational words: {', '.join(ham_hits[:5])}")
    if not urls:
        safe_factors.append("No suspicious URLs or links")
    if not caps_words:
        safe_factors.append("No ALL-CAPS words detected")
    if exclaims == 0:
        safe_factors.append("No exclamation marks — calm tone")
    if not money_syms:
        safe_factors.append("No money or percentage symbols")
    if not keyword_hits:
        safe_factors.append("No spam trigger words found")
    if cap_ratio < 0.1:
        safe_factors.append(f"Normal capital ratio ({round(cap_ratio*100,1)}%)")
    analysis["safe_factors"] = safe_factors

    if label == "spam":
        strength = ("extremely high","high","moderate","low")[
            0 if spam_prob>=0.95 else 1 if spam_prob>=0.80 else 2 if spam_prob>=0.60 else 3]
        summary = (f"This message is classified as SPAM with {strength} confidence "
                   f"({round(spam_prob*100,1)}% spam probability). ")
        if keyword_hits:
            summary += f"Spam keywords found in: {', '.join(keyword_hits.keys())}. "
        if urls:
            summary += f"{len(urls)} link(s) detected — common spam tactic. "
        if caps_words:
            summary += f"ALL-CAPS words ({', '.join(caps_words[:3])}) used as pressure technique. "
        if exclaims >= 2:
            summary += f"{exclaims} exclamation marks create false urgency. "
        if money_syms:
            summary += f"Money symbols ({', '.join(money_syms[:3])}) suggest financial scam. "
        if not ham_hits:
            summary += "No normal conversational language found. "
        summary += "AI model pattern recognition also flagged this based on overall text structure."
    else:
        strength = ("extremely high","high","moderate","low")[
            0 if ham_prob>=0.95 else 1 if ham_prob>=0.80 else 2 if ham_prob>=0.60 else 3]
        summary = (f"This message is classified as HAM (legitimate) with {strength} confidence "
                   f"({round(ham_prob*100,1)}% ham probability). ")
        if ham_hits:
            summary += f"Normal words found: {', '.join(ham_hits[:4])}. "
        if not keyword_hits:
            summary += "No spam trigger words detected. "
        if not urls:
            summary += "No suspicious links present. "
        if exclaims == 0:
            summary += "Calm tone with no urgency markers. "
        summary += "Language pattern matches legitimate messages in training data."

    analysis["verdict_explanation"] = summary
    analysis["model_scores"] = {
        "spam_probability": round(spam_prob, 4),
        "ham_probability":  round(ham_prob,  4),
        "confidence":       round(max(spam_prob, ham_prob), 4),
        "final_label":      label
    }
    return analysis


# ── Groq AI fallback ──────────────────────────────────────────────────────────
def ask_groq(text):
    """
    Call Groq AI when Naive Bayes confidence is too low.
    Returns a dict with label, spam_prob, ham_prob, explanation.
    """
    if GROQ_API_KEY == "______":
        return None  # Key not set, skip Groq

    prompt = f"""You are a spam detection expert. Analyze the following message carefully and determine if it is SPAM or HAM (legitimate).

Message:
\"\"\"
{text}
\"\"\"

Respond ONLY with a valid JSON object in this exact format (no extra text):
{{
  "label": "spam" or "ham",
  "spam_probability": 0.0 to 1.0,
  "ham_probability": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "short_reason": "one sentence reason",
  "detailed_explanation": "2-4 sentence detailed explanation of why this is spam or ham",
  "red_flags": ["list", "of", "specific", "red", "flags", "found"],
  "safe_signals": ["list", "of", "safe", "signals", "found"]
}}"""

    payload = json.dumps({
        "model":       GROQ_MODEL,
        "temperature": 0.1,
        "max_tokens":  600,
        "messages": [
            {"role": "system", "content": "You are an expert spam detection AI. Always respond with valid JSON only."},
            {"role": "user",   "content": prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL,
        data    = payload,
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        },
        method  = "POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result   = json.loads(resp.read().decode("utf-8"))
            content  = result["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            content  = re.sub(r"```json|```", "", content).strip()
            groq_data = json.loads(content)
            return groq_data
    except Exception as e:
        print(f"Groq API error: {e}")
        return None


# ── Main Classifier ───────────────────────────────────────────────────────────
class SpamClassifier:

    def __init__(self, model, vectorizer):
        self.model      = model
        self.vectorizer = vectorizer

    @classmethod
    def load(cls):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"spam_model.pkl not found. Run: python train.py")
        if not os.path.exists(VECTOR_PATH):
            raise FileNotFoundError(f"vectorizer.pkl not found. Run: python train.py")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTOR_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully.")
        return cls(model, vectorizer)

    def predict(self, text):
        cleaned = clean_text(text)

        if not cleaned:
            return {
                "label":        "ham",
                "confidence":   1.0,
                "spam_prob":    0.0,
                "ham_prob":     1.0,
                "original":     text,
                "cleaned":      cleaned,
                "source":       "local_model",
                "groq_used":    False,
                "reasons":      ["Message was empty after cleaning"],
                "analysis":     {"verdict_explanation": "Empty message, defaulted to ham."}
            }

        # ── Step 1: Local Naive Bayes ─────────────────────────────────────────
        vec     = self.vectorizer.transform([cleaned])
        proba   = self.model.predict_proba(vec)[0]
        classes = list(self.model.classes_)

        spam_prob  = float(proba[classes.index('spam')]) if 'spam' in classes else 0.0
        ham_prob   = float(proba[classes.index('ham')])  if 'ham'  in classes else 0.0
        confidence = max(spam_prob, ham_prob)
        label      = 'spam' if spam_prob > ham_prob else 'ham'

        # ── Step 2: If confidence is low → ask Groq ───────────────────────────
        groq_used   = False
        groq_result = None

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Low confidence ({round(confidence*100,1)}%) — calling Groq AI...")
            groq_result = ask_groq(text)

            if groq_result:
                groq_used  = True
                label      = groq_result.get("label", label)
                spam_prob  = float(groq_result.get("spam_probability", spam_prob))
                ham_prob   = float(groq_result.get("ham_probability",  ham_prob))
                confidence = float(groq_result.get("confidence", max(spam_prob, ham_prob)))
                print(f"Groq result: {label} ({round(confidence*100,1)}%)")

        # ── Step 3: Build analysis ────────────────────────────────────────────
        analysis = deep_analysis(text, label, spam_prob, ham_prob)

        # Merge Groq explanation if available
        if groq_result:
            analysis["groq_explanation"]  = groq_result.get("detailed_explanation", "")
            analysis["groq_short_reason"] = groq_result.get("short_reason", "")
            analysis["groq_red_flags"]    = groq_result.get("red_flags", [])
            analysis["groq_safe_signals"] = groq_result.get("safe_signals", [])
            # Override verdict explanation with Groq's detailed one
            if groq_result.get("detailed_explanation"):
                analysis["verdict_explanation"] = (
                    f"[Groq AI Analysis] {groq_result['detailed_explanation']}"
                )

        reasons = (analysis["risk_factors"][:4] if label == 'spam'
                   else analysis["safe_factors"][:4])
        if not reasons:
            reasons = [f"AI model classified this as {label}"]

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "spam_prob":  round(spam_prob, 4),
            "ham_prob":   round(ham_prob, 4),
            "original":   text,
            "cleaned":    cleaned,
            "source":     "groq_ai" if groq_used else "local_model",
            "groq_used":  groq_used,
            "reasons":    reasons,
            "analysis":   analysis
        }

    def predict_batch(self, texts):
        if not isinstance(texts, list) or len(texts) == 0:
            return []
        return [self.predict(t) for t in texts]