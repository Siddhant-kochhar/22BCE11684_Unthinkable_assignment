# E-commerce Product Recommender (with LLM Explanations)

## **🎥 Project Demo Video**
**[📹 WATCH PROJECT DEMO](https://drive.google.com/file/d/1vthAC1xe4L9SJK1_gnOmpq_b_3VJjGsZ/view?usp=sharing)** 

---

This project combines a content-based recommender with an LLM that explains "why this product" to the user. It includes a FastAPI backend, a minimal frontend, and optional Google Form onboarding.

## ✅ Alignment with Brief

- Input: product catalog (`clean_data.csv`) + user behavior (click tracking)
- Output: recommended products + LLM-generated explanations
- Backend API: FastAPI endpoints for auth, tracking, and recommendations
- Database: CSV for products, JSON for user accounts/clicks (file-based for simplicity)
- LLM: Google Gemini via `google-generativeai`
- Optional frontend: `index.html` and `main.html` UIs with Bootstrap and JS

## 🗂️ Repository Layout

```
.
├── fastapi_app.py            # Main FastAPI server (templates, routes, ML, LLM)
├── index.html                # Landing page (trending + auth)
├── main.html                 # Main app page (recommendations)
├── static/                   # JS, images, video
│   └── click-tracking.js
├── backend/gemini_helper.py  # LLM helper (optional)
├── clean_data.csv            # Training dataset (4,090 rows)
├── trending_products.csv     # Subset for homepage
├── users_db.json             # File-based user store (sanitized)
├── run.sh                    # Start server using venv + uvicorn
├── setup.sh                  # Create venv and install deps
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quickstart

```bash
# 1) Create venv and install deps
./setup.sh

# 2) Add your Gemini API key
cp .env.example .env
echo 'GEMINI_API_KEY=your_key_here' >> .env

# 3) Run the server
./run.sh

# Open http://localhost:8000
```

## 🔌 Key Endpoints

- `GET /` – Homepage (Trending products)
- `GET /main` – Main app (search + recommendations UI)
- `POST /signin` – Username/password auth
- `POST /track-click` – Track product clicks per user session
- `GET /my-recommendations` – Personalized recommendations with explanations
- `GET /docs` – Auto-generated API docs (FastAPI)

## 🧠 Recommendation Logic

Content-based filtering using TF-IDF on product text fields with cosine similarity. We prevent NaN issues, dedupe, and filter out products the user already clicked. The system maintains a per-user click history in memory (and can be persisted).

High-level contract:
- Input: last N clicked products by user, product catalog
- Output: list of similar products with similarity scores
- Edge cases: empty behavior → fallback to trending; NaN similarity → filtered out

## 🤖 LLM Explanations

We call Gemini to generate a concise explanation:

Example prompt (simplified):
```
Explain why "{recommended_name}" is recommended for a user who viewed: {recent_clicks}.
Include brand/category affinity and 1-2 concrete attributes.
Tone: helpful, 1–2 sentences.
```

FastAPI integrates this via `generate_recommendation_explanation(...)` in `fastapi_app.py` with graceful fallback if LLM is unavailable.

## 🧪 Demo Flow

1. Start server: `./run.sh`
2. Open `/` and sign in (create a user via Sign Up)
3. Click on a few products (image/title)
4. Open the personal recommendations section
5. See recommended products with AI explanations

## 📊 Data

- `clean_data.csv`: main dataset used for training TF-IDF
- `trending_products.csv`: smaller curated list for homepage

Note: The repo ships with file-based storage to keep the demo self-contained. Swap to a DB for production.

## 🛠️ Development Notes

- Python 3.10+
- Dependencies in `requirements.txt`
- Environment: `.env` with `GEMINI_API_KEY`
- Run with `./run.sh` (expects `venv` created by `./setup.sh`)

## 🧹 Security & Privacy

- This repo has no default users; `users_db.json` is empty by default
- Do not commit real API keys; `.env` is gitignored

## 📹 Demo Video

Record with any screen recorder. Suggested flow:
1. Start the app and show `/`
2. Sign up → sign in
3. Click 3–4 products
4. Show personalized recommendations + AI explanations
5. Open `/docs` to highlight API

## Evaluation Mapping

- Accuracy: similarity scoring + filtering
- Explanation quality: short, personalized Gemini outputs
- Code design: single FastAPI app with modular helpers and clear routes
- Documentation: this README + inline comments