from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
import joblib
import os
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Movie Box Office Prediction API",
    description="Predict Box Office Collection from movie input features.",
    version="1.1.0",
)

ALLOWED_ORIGINS = [
    "https://ais-dev-ypfyd2j2e2f3ntvj47zsb4-592544366683.asia-east1.run.app",
    "https://ais-pre-ypfyd2j2e2f3ntvj47zsb4-592544366683.asia-east1.run.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Locate model.pkl (works whether you run from project root or backend/)
# Override with:  set MODEL_PATH=C:\full\path\model.pkl
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.getenv("MODEL_PATH", ""),
    os.path.join(BASE_DIR, "..", "pickle files", "model.pkl"),        # app/pickle files
    os.path.join(BASE_DIR, "..", "..", "pickle files", "model.pkl"),  # project/pickle files
    os.path.join(BASE_DIR, "pickle files", "model.pkl"),
    os.path.join(os.getcwd(), "app", "pickle files", "model.pkl"),
]
MODEL_PATH = next(
    (os.path.abspath(p) for p in _candidates if p and os.path.exists(p)),
    os.path.abspath(_candidates[1]),
)

bundle = None
model = None
LOAD_ERROR = None

GENRE_KEYWORDS = [
    "Action", "Drama", "Comedy", "Thriller", "Romantic", "Romance",
    "Crime", "Horror", "Musical", "Biographical", "Mystery", "Family",
    "Historical", "Social", "Spy", "Period", "Mythological", "Fantasy",
    "War", "Sports",
]


def group_genre(value):
    if pd.isna(value):
        return "Unknown"
    tokens = re.findall(r"[A-Za-z]+", str(value))
    for token in tokens:
        for keyword in GENRE_KEYWORDS:
            if token.lower() == keyword.lower():
                return "Romance" if keyword == "Romantic" else keyword
    return "Other"


def clean_raw(frame, cat_cols, num_cols):
    frame = frame.copy()
    for c in cat_cols:
        s = frame[c].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        s = s.mask(s == "")
        frame[c] = s.astype("object").where(s.notna(), np.nan)
    for c in num_cols:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def norm_key(series):
    return (
        series.astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True).str.lower()
    )


def load_model():
    global bundle, model, LOAD_ERROR

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.pkl was not found. Expected location: {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    if not isinstance(loaded, dict):
        raise ValueError("model.pkl must contain a model bundle dictionary.")

    required_keys = [
        "model", "model_name", "target_column", "raw_input_columns",
        "cat_imputer", "num_imputer", "budget_cap",
        "director_freq_map", "production_house_freq_map",
        "male_lead_freq_map", "female_lead_freq_map",
        "genre_map", "global_mean", "final_model_features",
    ]
    missing = [k for k in required_keys if k not in loaded]
    if missing:
        raise ValueError("model.pkl is missing required keys: " + ", ".join(missing))

    bundle = loaded
    model = loaded["model"]
    LOAD_ERROR = None

    try:
        import sklearn
        saved_v = loaded.get("sklearn_version")
        if saved_v and saved_v != sklearn.__version__:
            print(f"WARNING: model trained with scikit-learn {saved_v}, "
                  f"API is running {sklearn.__version__}")
    except Exception:
        pass

    print("==========================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("Model:", bundle["model_name"])
    print("Target:", bundle["target_column"])
    print("Model file:", MODEL_PATH)
    print("==========================================")


try:
    load_model()
except Exception as e:
    LOAD_ERROR = str(e)
    print("ERROR LOADING MODEL:", e)


class MovieInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Year: int = Field(..., ge=1900, le=2100, examples=[2025])
    Budget: float = Field(..., ge=0, description="Budget in Crores", examples=[150])
    Director: str = Field(..., examples=["Rohit Shetty"])
    Female_Lead: str = Field(..., examples=["Deepika Padukone"])
    Male_Lead: str = Field(..., examples=["Ajay Devgn"])
    Production_House: str = Field(..., examples=["Reliance Entertainment"])
    Genre: str = Field(..., examples=["Action"])
    Release_type: str = Field(..., alias="Release type", examples=["Theatrical"])


def prepare_movie(raw_input: MovieInput) -> pd.DataFrame:
    """Same steps as prepare_features() in the notebook."""
    if bundle is None or model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is not loaded. {LOAD_ERROR or ''} Check: {MODEL_PATH}",
        )

    raw = pd.DataFrame([{
        "Year": raw_input.Year,
        "Budget": raw_input.Budget,
        "Director": raw_input.Director,
        "Female_Lead": raw_input.Female_Lead,
        "Male_Lead": raw_input.Male_Lead,
        "Production_House": raw_input.Production_House,
        "Genre": raw_input.Genre,
        "Release type": raw_input.Release_type,
    }])

    # Column order taken from the FITTED imputers -> can never mismatch
    cat_cols = list(bundle["cat_imputer"].feature_names_in_)
    num_cols = list(bundle["num_imputer"].feature_names_in_)

    row = clean_raw(raw[bundle["raw_input_columns"]], cat_cols, num_cols)
    row[cat_cols] = bundle["cat_imputer"].transform(row[cat_cols])
    row[num_cols] = bundle["num_imputer"].transform(row[num_cols])

    row["Budget"] = row["Budget"].clip(upper=bundle["budget_cap"])
    row["Genre_Grouped"] = row["Genre"].apply(group_genre)
    row["Budget_Log"] = np.log1p(row["Budget"].clip(lower=0))

    row["Director_Freq"] = norm_key(row["Director"]).map(bundle["director_freq_map"]).fillna(1)
    row["ProdHouse_Freq"] = norm_key(row["Production_House"]).map(bundle["production_house_freq_map"]).fillna(1)
    row["MaleLead_Freq"] = norm_key(row["Male_Lead"]).map(bundle["male_lead_freq_map"]).fillna(1)
    row["FemaleLead_Freq"] = norm_key(row["Female_Lead"]).map(bundle["female_lead_freq_map"]).fillna(1)
    row["Genre_Hist_Avg"] = row["Genre_Grouped"].map(bundle["genre_map"]).fillna(bundle["global_mean"])

    final_features = list(bundle["final_model_features"])
    missing_features = [f for f in final_features if f not in row.columns]
    if missing_features:
        raise ValueError("Required model features were not created: " + ", ".join(missing_features))

    return row[final_features].copy()


@app.get("/")
def home():
    return {
        "message": "Movie Box Office Prediction API",
        "status": "running",
        "prediction_endpoint": "/predict",
        "health_endpoint": "/health",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_name": bundle.get("model_name") if bundle else None,
        "target": bundle.get("target_column") if bundle else None,
        "model_path": MODEL_PATH,
        "error": LOAD_ERROR,
    }


@app.post("/predict")
def predict(movie: MovieInput):
    try:
        X = prepare_movie(movie)
        predicted_log = model.predict(X)
        predicted_value = max(0.0, float(np.expm1(predicted_log[0])))

        return {
            "status": "success",
            "model": bundle["model_name"],
            "target": "Box Office Collection",
            "prediction": {
                "box_office_collection": round(predicted_value, 2),
                "unit": "Crores",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)