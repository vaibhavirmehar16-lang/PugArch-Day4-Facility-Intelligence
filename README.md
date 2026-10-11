# PugArch Technology — Day 4 Facility Intelligence & ML Platform

> **Production-style upgrade of the Day 3 analytics project:** a full-stack Facility Hygiene Risk platform with a FastAPI backend, React frontend, supervised machine learning, evaluation, explainable operational prioritization, and a REST API for live predictions.

# Demo Link-https://pugarch-day4-facility-intelligence-1.onrender.com/
## 1. Executive Summary

Day 3 answered **“What is happening?”** using data cleaning, EDA, statistical analysis and visualization.

Day 4 adds the predictive layer: **“What is likely to happen next, and what should the operator do?”**

The platform accepts facility inspection signals, sends them through a trained scikit-learn model, returns a hygiene-risk prediction and presents the result through a responsive React UI. Dashboard metrics and predictions are fetched from FastAPI APIs; they are not hardcoded into the frontend.

### Business problem

Facility teams need to prioritize cleaning and inspection resources. A location with poor cleanliness, elevated odor, high waste, heavy footfall, more complaints and a long time since the last cleaning may require faster intervention.

This project turns those operational signals into:

- Hygiene-risk predictions
- Location priority rankings
- Model performance comparisons
- High-risk probability
- Action-oriented recommendations
- REST endpoints suitable for another client application

> **Important modeling note:** this internship dataset is synthetic. `hygiene_risk` is an educational target generated after data cleaning from operational signals. In a production system, it should be replaced with historical labeled outcomes such as inspection failures, sanitation incidents or supervisor-confirmed risk labels.

---

## 2. Architecture

```text
                    ┌──────────────────────────┐
                    │       React Frontend      │
                    │ TypeScript + Recharts     │
                    │ Responsive Client UI      │
                    └────────────┬─────────────┘
                                 │ HTTP / JSON
                                 ▼
                    ┌──────────────────────────┐
                    │       FastAPI Backend     │
                    │ REST API + Validation     │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       ┌──────────────┐   ┌───────────────┐  ┌───────────────┐
       │ Clean Dataset│   │ ML Model       │  │ Analytics     │
       │ Pandas/NumPy │   │ scikit-learn  │  │ Risk Ranking  │
       └──────┬───────┘   └───────┬───────┘  └───────────────┘
              │                   │
              ▼                   ▼
       Data Cleaning        Prediction + Metrics
              │
              ▼
       Raw Inspection CSV
```

### End-to-end ML workflow

```text
Raw dataset
   ↓
Data quality inspection
   ↓
Cleaning + imputation + outlier treatment
   ↓
EDA / statistical analysis
   ↓
Feature engineering + target creation
   ↓
Train / test split
   ↓
Preprocessing pipeline
   ↓
Logistic Regression ─┐
                     ├── Compare → Select by F1
Random Forest ───────┘
   ↓
Saved model (.joblib)
   ↓
FastAPI /api/predict
   ↓
React client
   ↓
Risk + probability + recommendation
```

---

## 3. Why Machine Learning Is Actually Used

ML is not included only to make the project look advanced.

The Day 4 assignment specifically asks for a model that predicts facility hygiene risk and compares at least two algorithms. Therefore, the project now has a real ML training, evaluation and inference workflow.

### ML libraries

| Library | Purpose | Where used |
|---|---|---|
| **scikit-learn** | Core ML framework | preprocessing, train/test split, Logistic Regression, Random Forest, K-Means, metrics |
| **joblib** | Model serialization | saves the selected trained model for FastAPI inference |
| **NumPy** | Numerical processing | data cleaning, synthetic target generation and calculations |
| **Pandas** | Data preparation | loading, transforming and serving facility data |
| **SciPy** | Scientific computing support | statistical/data-science dependency |
| **Matplotlib / Seaborn** | Evaluation/documentation visuals | confusion matrix and Day 3 static analytics |

No unnecessary deep-learning library is included. **TensorFlow/PyTorch are not needed** for this binary tabular prediction problem.

---

## 4. Day 4 Dataset

The dataset contains the Day 3 operational fields plus the Day 4 feature:

| Feature | Role |
|---|---|
| `cleanliness_score` | Numerical hygiene indicator |
| `odor_score` | Numerical odor severity |
| `waste_level` | Categorical waste condition |
| `water_availability` | Categorical infrastructure signal |
| `footfall` | Numerical visitor pressure |
| `complaints` | Numerical service-quality signal |
| `hours_since_cleaning` | Numerical cleaning recency |
| `hygiene_risk` | ML target: `Low` / `High` |

The raw generator creates 155 rows: 150 base observations plus 5 intentional duplicates. The cleaning pipeline reduces this to 150 records.

---

## 5. Data Cleaning

`data-cleaning/clean_data.py` performs:

- Type normalization
- Duplicate removal
- Negative complaint correction
- Location-wise median imputation for cleanliness and odor
- Mode imputation for water availability
- IQR-based footfall outlier capping
- Date standardization
- `hours_since_cleaning` validation
- Synthetic educational `hygiene_risk` target generation

Output:

```text
dataset/cleaned_facility_data.csv
reports/data_quality_report.json
```

### Current data-quality result

- Raw rows: **155**
- Clean rows: **150**
- Duplicates removed: **5**
- Negative complaints corrected: **3**
- Missing water values imputed: **12**
- Footfall outliers capped: **4**
- Remaining missing values: **0**

---

## 6. Machine Learning Models

### Model 1 — Logistic Regression

Used as an interpretable baseline for binary classification.

Advantages:

- Fast
- Easy to explain
- Strong baseline for tabular data
- Produces class probabilities

### Model 2 — Random Forest

Used as a nonlinear ensemble model.

Advantages:

- Captures nonlinear relationships
- Handles mixed operational patterns well
- Provides a stronger benchmark against the linear baseline

### Optional unsupervised layer — K-Means

`models/cluster_facilities.py` groups facilities into three operational segments using cleanliness, odor, footfall, complaints and cleaning delay. This is useful for discovering facility profiles without labels.

---

## 7. Evaluation

The model is evaluated on a stratified 20% holdout test set.

Current generated-data benchmark:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7000 | 0.6471 | **0.7857** | **0.7097** |
| Random Forest | **0.7333** | **0.7500** | 0.6429 | 0.6923 |

The project selects the model using **holdout F1 score**, not by choosing the most complicated algorithm. The current selected model is **Logistic Regression** because its F1 score is higher.

The evaluation folder also contains:

- `model_metrics.json`
- `confusion_matrix.png`
- `segment_profiles.csv`
- `facility_segments.csv`

Probability-quality diagnostics also include MAE, MSE and RMSE between predicted high-risk probability and the binary holdout label. These are reported as probability-error diagnostics rather than pretending that classification is a regression problem.

---

## 8. FastAPI Backend

Main backend:

```text
backend/app/main.py
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service health check |
| `GET /api/meta` | Locations, waste levels and date range |
| `GET /api/summary` | Live KPI summary with optional filters |
| `GET /api/facilities` | Facility records |
| `GET /api/analytics/location-risk` | Location priority ranking |
| `GET /api/model/metrics` | Model benchmark and evaluation results |
| `POST /api/predict` | Live ML hygiene-risk prediction |

FastAPI automatically exposes interactive API documentation at:

```text
/docs
```

---

## 9. React Frontend

The frontend is located in:

```text
frontend/
```

Technology:

- React
- TypeScript
- Vite
- Recharts
- Lucide React
- Responsive CSS

The UI provides:

- Executive KPI cards
- Location and waste filters
- Interactive risk charts
- Model benchmark panel
- Operational insight cards
- Live ML prediction form
- Risk probability
- Action recommendation
- Responsive mobile layout

There are **no hardcoded KPI values or prediction results** in the frontend. Data is retrieved through the FastAPI API.

---

## 10. Project Structure

```text
day-04-facility-intelligence/
│
├── backend/
│   └── app/
│       └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── dataset/
│   ├── generate_dataset.py
│   ├── raw_facility_data.csv
│   └── cleaned_facility_data.csv
│
├── preprocessing/
│   ├── feature_engineering.py
│   └── train_test_split.py
│
├── models/
│   ├── train_models.py
│   ├── cluster_facilities.py
│   └── best_hygiene_risk_model.joblib
│
├── evaluation/
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── segment_profiles.csv
│   └── facility_segments.csv
│
├── predictions/
│   └── test_predictions.csv
│
├── data-cleaning/
│   └── clean_data.py
│
├── analysis/
│   └── facility_analysis.py
│
├── visualizations/
│   └── Day 3 visualization assets
│
├── Dockerfile
├── render.yaml
├── requirements.txt
└── README.md
```

---

## 11. Local Setup

### Backend / ML environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Generate the raw dataset:

```bash
python dataset/generate_dataset.py
```

Clean the dataset:

```bash
python data-cleaning/clean_data.py
```

Train and compare models:

```bash
python models/train_models.py
```

Run optional K-Means segmentation:

```bash
python models/cluster_facilities.py
```

Start FastAPI:

```bash
uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Frontend development

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to the FastAPI server.

---

## 12. Production-style Docker Workflow

The repository uses a multi-stage Docker build:

1. Node builds the React application.
2. Python runs FastAPI.
3. FastAPI serves the compiled React frontend and API from the same service.

Build:

```bash
docker build -t facility-intelligence-day4 .
```

Run:

```bash
docker run -p 8000:8000 facility-intelligence-day4
```

Open:

```text
http://localhost:8000
```

---

## 13. Render Deployment

The project is configured for Docker-based Render deployment using:

```text
render.yaml
Dockerfile
```

Connect the GitHub repository to Render as a **Web Service**. Render builds the Docker image and starts FastAPI on the platform-provided port.

Health endpoint:

```text
/api/health
```

Live UI and API are served from the same public service URL.

---

## 14. What This Project Demonstrates

This Day 4 implementation demonstrates the complete path from data to product:

```text
Data Engineering
      ↓
Data Quality
      ↓
EDA / Statistics
      ↓
Feature Engineering
      ↓
Supervised ML
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
REST API
      ↓
Responsive Frontend
      ↓
Docker
      ↓
Cloud Deployment
```

This is materially different from adding an ML import to a dashboard: **the trained model is persisted, served through an API and used by the client UI for a new prediction.**

---

## 15. Production Improvements

For a real facility-operations deployment, the next improvements would be:

- Replace synthetic risk labels with historical inspection outcomes.
- Add authentication and role-based access.
- Add a real database such as PostgreSQL instead of relying on CSV files.
- Add model versioning and experiment tracking.
- Add scheduled retraining when new labeled inspections arrive.
- Add model drift and data-quality monitoring.
- Add audit logs for prediction requests.
- Add explainability with SHAP for individual predictions.
- Add CI/CD tests and security scanning.

