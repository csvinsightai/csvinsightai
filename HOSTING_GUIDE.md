# 🚀 DataML Studio — Streamlit Hosting Guide

## Complete Step-by-Step Deployment on Streamlit Cloud

---

## ✅ PREREQUISITES

Before you start, make sure you have:
- A **GitHub account** (free) → https://github.com
- A **Streamlit Cloud account** (free) → https://streamlit.io/cloud
- Python 3.9+ installed on your local machine

---

## 📁 STEP 1 — Prepare Your Project Files

Your project folder must contain **exactly these files**:

```
ml_platform/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
└── .streamlit/
    └── config.toml         ← UI theme settings
```

---

## 🐍 STEP 2 — Test Locally First

Open your **terminal / command prompt** and run:

```bash
# 1. Navigate to your project folder
cd ml_platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Your browser will open → **http://localhost:8501**

If it works locally, you're ready to deploy!

---

## 📤 STEP 3 — Push to GitHub

```bash
# 1. Initialize git (if not already done)
git init

# 2. Create a .gitignore
echo "__pycache__/
*.pyc
.env
venv/" > .gitignore

# 3. Add all files
git add .

# 4. Commit
git commit -m "Initial commit: DataML Studio"

# 5. Create a new repo on GitHub.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

> **Tip:** Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub details.

---

## ☁️ STEP 4 — Deploy on Streamlit Cloud

1. Go to → **https://share.streamlit.io**
2. Click **"New app"**
3. Fill in the form:
   - **Repository:** `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**

⏳ Deployment takes **1–3 minutes**. Once done, you'll get a public URL like:
```
https://your-app-name.streamlit.app
```

---

## 🎯 STEP 5 — Using the App

Once deployed, follow this workflow:

| Step | Section | What to Do |
|------|---------|-----------|
| 1 | 📁 Upload Data | Upload your CSV or click "Load Demo Dataset" |
| 2 | 📊 Data Overview | View shape, column types, and statistics |
| 3 | 🧹 Data Cleaning | Handle missing values & duplicates |
| 4 | 📈 EDA Visualization | Explore histograms, boxplots, heatmaps |
| 5 | 🧠 Model Training | Select target column + models → Train |
| 6 | 🏆 Results Dashboard | Compare models, view confusion matrix & ROC |

---

## 🔧 TROUBLESHOOTING

### ❌ "ModuleNotFoundError"
→ Make sure all packages are in `requirements.txt` with correct names

### ❌ App crashes on large files
→ Add to `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
```

### ❌ "Resource limits exceeded" on Streamlit Cloud
→ Free tier has 1GB RAM. For large datasets, consider:
- Sampling your data before upload
- Upgrading to Streamlit Cloud Pro

### ❌ Font / CSS not loading
→ Check your internet connection; Google Fonts requires internet access

---

## 🌐 ALTERNATIVE HOSTING OPTIONS

| Platform | Cost | Steps |
|----------|------|-------|
| **Streamlit Cloud** | Free | Easiest, shown above |
| **Hugging Face Spaces** | Free | Use Docker or Streamlit SDK |
| **Railway.app** | Free tier | Add `Procfile` with `web: streamlit run app.py --server.port $PORT` |
| **Render.com** | Free tier | Similar to Railway |
| **AWS / GCP / Azure** | Paid | Full control, use Docker |

---

## 📝 REQUIREMENTS.TXT (copy this exactly)

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
```

---

## 💡 PRO TIPS

- **Auto-updates:** Push to GitHub → Streamlit Cloud auto-redeploys
- **Sharing:** Your `.streamlit.app` URL is public and shareable
- **Secrets:** Use `st.secrets` for API keys (never hardcode)
- **Performance:** Use `@st.cache_data` decorator for expensive operations

---

*Built with ❤️ using Streamlit, Scikit-learn & Plotly*
