name(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model    = joblib.load(f"{MODEL_DIR}/agri_model.pkl")
scaler   = joblib.load(f"{MODEL_DIR}/agri_scaler.pkl")
FEATURES = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# Load SHAP explainer once at startup
try:
    import shap
    EXPLAINER      = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    # For regression, expected_value is a single float
    EXPECTED_VALUE = float(EXPLAINER.expected_value)
    logger.info(f"SHAP loaded. Base yield={EXPECTED_VALUE:.4f} t/ha")
except ImportError:
    SHAP_AVAILABLE = False
    EXPECTED_VALUE = None
    logger.warning("shap not installed — pip install shap")

logger.info(f"Agri model loaded. Features={len(FEATURES)}")

CROPS = ("Maize", "Wheat", "Beans", "Sorghum", "Millet", "Tea", "Coffee")
SEEDS = ("Hybrid", "Traditional", "Improved")
COUNTIES = [
    "Nairobi","Mombasa","Kisumu","Nakuru","Uasin Gishu","Meru","Kilifi",
    "Kakamega","Machakos","Garissa","Turkana","Mandera","Wajir","Trans Nzoia",
    "Nandi","Kericho","Bomet","Kisii","Nyamira","Migori","Homa Bay","Siaya",
    "Busia","Bungoma","Vihiga","Laikipia","Nyeri","Kirinyaga","Murang'a",
    "Kiambu","Nyandarua","Embu","Tharaka Nithi","Isiolo","Marsabit","Samburu",
    "Baringo","West Pokot","Elgeyo Marakwet","Kwale","Taita Taveta","Kajiado",
    "Narok","Makueni","Kitui","Tana River","Lamu",
]

# ── APP ───────────────────────────────────────────────────
app = FastAPI(
    title="Agricultural Yield Prediction API",
    description=(
        "Random Forest yield model with SHAP explainability.\n\n"
        "**Endpoints:**\n"
        "- `POST /predict` — yield prediction + insights\n"
        "- `POST /predict/explain` — full SHAP values per feature\n"
        "- `POST /predict/batch` — score up to 200 farms\n"
        "- `GET /health` — model status\n"
        "- `GET /model/info` — model metadata"
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── INPUT SCHEMA ──────────────────────────────────────────
class FarmInput(BaseModel):
    county           : str   = Field(...,                   example="Trans Nzoia")
    crop             : str   = Field("Maize",               example="Maize")
    rainfall_mm      : float = Field(..., ge=0,   le=2000,  example=950.0)
    avg_temp_c       : float = Field(..., ge=10,  le=40,    example=21.5)
    fertiliser_kg_ha : float = Field(0.0, ge=0,  le=500,   example=80.0)
    soil_quality     : int   = Field(3,   ge=1,  le=5,     example=4)
    drought_year     : int   = Field(0,   ge=0,  le=1,     example=0)
    seed_variety     : str   = Field("Hybrid",              example="Hybrid")
    farm_size_ha     : float = Field(1.0, ge=0.1,le=100,   example=3.0)
    irrigation       : int   = Field(0,   ge=0,  le=1,     example=0)
    year             : int   = Field(2024, ge=2000,le=2030, example=2024)

    @validator("crop")
    def v_crop(cls, v):
        if v not in CROPS:
            raise ValueError(f"crop must be one of {CROPS}")
        return v

    @validator("seed_variety")
    def v_seed(cls, v):
        if v not in SEEDS:
            raise ValueError(f"seed_variety must be one of {SEEDS}")
        return v

    @validator("county")
    def v_county(cls, v):
        if v not in COUNTIES:
            raise ValueError(f"'{v}' is not a valid Kenya county")
        return v

    class Config:
        schema_extra = {"example": {
            "county": "Trans Nzoia", "crop": "Maize",
            "rainfall_mm": 950, "avg_temp_c": 21.5,
            "fertiliser_kg_ha": 80, "soil_quality": 4,
            "drought_year": 0, "seed_variety": "Hybrid",
            "farm_size_ha": 3.0, "irrigation": 0, "year": 2024,
        }}

# ── FEATURE BUILDER ───────────────────────────────────────
def build_row(f: FarmInput) -> pd.DataFrame:
    row = {feat: 0 for feat in FEATURES}
    row.update({
        "rainfall_mm"       : f.rainfall_mm,
        "avg_temp_c"        : f.avg_temp_c,
        "fertiliser_kg_ha"  : f.fertiliser_kg_ha,
        "soil_quality"      : f.soil_quality,
        "drought_year"      : f.drought_year,
        "farm_size_ha"      : f.farm_size_ha,
        "irrigation"        : f.irrigation,
        "year"              : f.year,
        "rainfall_temp_idx" : f.rainfall_mm / max(f.avg_temp_c, 1),
        "fertiliser_per_ha" : f.fertiliser_kg_ha / max(f.farm_size_ha, 0.1),
    })
    for col in [f"county_{f.county}", f"crop_{f.crop}",
                f"seed_{f.seed_variety}"]:
        if col in row:
            row[col] = 1
    return pd.DataFrame([row])[FEATURES]

def get_insights(f: FarmInput, yield_tha: float) -> List[str]:
    insights = []
    if f.drought_year == 1:
        insights.append("Drought year — expected 44.8% yield reduction vs normal season")
    if f.seed_variety == "Hybrid":
        insights.append("Hybrid seed — typically 35% higher yield than traditional varieties")
    if f.seed_variety == "Traditional":
        insights.append("Traditional seed — consider switching to hybrid for +35% yield uplift")
    if f.rainfall_mm < 400:
        insights.append("Low rainfall (<400mm) — irrigation strongly recommended")
    if f.rainfall_mm > 1500:
        insights.append("Very high rainfall — monitor for waterlogging and fungal disease")
    if f.fertiliser_kg_ha < 30:
        insights.append("Low fertiliser — increasing to 60-80 kg/ha improves yield significantly")
    if f.fertiliser_kg_ha > 200:
        insights.append("High fertiliser application — diminishing returns above 200 kg/ha")
    if f.soil_quality <= 2:
        insights.append("Poor soil quality — consider soil amendment or lime application")
    if yield_tha > 3.5:
        insights.append(f"Above-average yield expected ({yield_tha:.2f} t/ha vs 1.84 t/ha national)")
    if yield_tha < 1.0:
        insights.append(f"Below-average yield expected — review inputs and growing conditions")
    return insights or ["No specific agronomic flags for this profile"]

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "api"      : "Agricultural Yield Prediction API",
        "version"  : "1.0.0",
        "counties" : 47,
        "crops"    : list(CROPS),
        "shap"     : SHAP_AVAILABLE,
        "docs"     : "/docs",
        "endpoints": {
            "predict"        : "POST /predict",
            "predict_explain": "POST /predict/explain",
            "predict_batch"  : "POST /predict/batch",
            "health"         : "GET /health",
            "model_info"     : "GET /model/info",
        }
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"          : "healthy",
        "model"           : "RandomForestRegressor",
        "r2"              : 0.88,
        "mae_tha"         : 0.19,
        "feature_count"   : len(FEATURES),
        "shap_available"  : SHAP_AVAILABLE,
        "shap_base_yield" : round(EXPECTED_VALUE, 4) if EXPECTED_VALUE else None,
    }

@app.get("/model/info", tags=["Info"])
def model_info():
    return {
        "algorithm"          : "RandomForestRegressor",
        "dataset"            : "KNBS + KMD Kenya agricultural data",
        "counties"           : 47,
        "r2"                 : 0.88,
        "mae_tha"            : 0.19,
        "national_avg_tha"   : 1.84,
        "supported_crops"    : list(CROPS),
        "supported_seeds"    : list(SEEDS),
        "shap_explainer"     : "TreeExplainer" if SHAP_AVAILABLE else "not available",
        "key_findings"       : [
            "Rainfall strongest predictor (r=0.61)",
            "Hybrid seed +35% yield vs traditional",
            "Drought reduces yield by 44.8%",
            "Trans Nzoia and Uasin Gishu top yielding counties",
        ],
    }

@app.post("/predict", tags=["Prediction"])
def predict(farm: FarmInput):
    """Predict crop yield — returns t/ha, total tonnes, confidence range, insights."""
    start = time.time()
    try:
        X      = build_row(farm)
        X_s    = scaler.transform(X)
        yield_ = max(0.0, float(model.predict(X_s)[0]))
        total  = round(yield_ * farm.farm_size_ha, 2)
        return {
            "predicted_yield_tha"  : round(yield_, 3),
            "predicted_yield_kgha" : round(yield_ * 1000, 0),
            "total_yield_tonnes"   : total,
            "total_yield_kg"       : round(total * 1000, 0),
            "farm_size_ha"         : farm.farm_size_ha,
            "county"               : farm.county,
            "crop"                 : farm.crop,
            "seed_variety"         : farm.seed_variety,
            "confidence_range_tha" : [round(yield_ - 0.19, 3),
                                      round(yield_ + 0.19, 3)],
            "model_mae_tha"        : 0.19,
            "national_avg_tha"     : 1.84,
            "vs_national_pct"      : round((yield_ - 1.84) / 1.84 * 100, 1),
            "insights"             : get_insights(farm, yield_),
            "processing_ms"        : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(farm: FarmInput):
    """
    Predict yield AND return full SHAP explanation.

    Returns everything from /predict PLUS:
    - shap_values: every feature and its contribution to the yield prediction
    - shap_top_drivers: top 10 features ranked by importance for THIS farm
    - shap_explanation: plain English summary of what is driving the yield
    - shap_base_value: average yield across all training farms (t/ha)

    How to read SHAP values for regression:
    - Positive value → feature INCREASED predicted yield
    - Negative value → feature DECREASED predicted yield
    - Units are in t/ha — a SHAP value of +0.5 means that feature
      added 0.5 t/ha to the prediction above the base yield

    Example:
    base_value = 1.84 t/ha (national average)
    rainfall_mm shap = +0.72 → good rainfall added 0.72 t/ha above average
    drought_year shap = -0.83 → drought removed 0.83 t/ha from prediction
    """
    if not SHAP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SHAP not installed. Run: pip install shap"
        )
    start = time.time()
    try:
        X      = build_row(farm)
        X_s    = scaler.transform(X)
        X_df   = pd.DataFrame(X_s, columns=FEATURES)

        # Prediction
        yield_ = max(0.0, float(model.predict(X_s)[0]))
        total  = round(yield_ * farm.farm_size_ha, 2)

        # SHAP values — regression model returns single array (not list)
        shap_vals = EXPLAINER.shap_values(X_df)
        # For RandomForest regression: shap_vals is shape (n_samples, n_features)
        sv_row = shap_vals[0]

        # Full SHAP dict — every feature and its contribution in t/ha
        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in zip(FEATURES, sv_row)
        }

        # Top 10 drivers for this farm
        sorted_shap = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_drivers = [
            {
                "feature"    : feat,
                "shap_value" : val,
                "shap_tha"   : round(val, 4),
                "direction"  : "increases yield" if val > 0 else "reduces yield",
                "magnitude"  : (
                    "high"   if abs(val) > 0.3 else
                    "medium" if abs(val) > 0.1 else "low"
                ),
            }
            for feat, val in sorted_shap[:10]
        ]

        # Plain English explanation
        top3  = sorted_shap[:3]
        parts = []
        for feat, val in top3:
            direction = "added" if val > 0 else "removed"
            parts.append(
                f"{feat} {direction} {abs(val):.2f} t/ha "
                f"{'above' if val > 0 else 'below'} the base yield"
            )
        explanation = ". ".join(parts) + "."

        return {
            # Standard prediction
            "predicted_yield_tha"  : round(yield_, 3),
            "predicted_yield_kgha" : round(yield_ * 1000, 0),
            "total_yield_tonnes"   : total,
            "total_yield_kg"       : round(total * 1000, 0),
            "farm_size_ha"         : farm.farm_size_ha,
            "county"               : farm.county,
            "crop"                 : farm.crop,
            "seed_variety"         : farm.seed_variety,
            "confidence_range_tha" : [round(yield_ - 0.19, 3),
                                      round(yield_ + 0.19, 3)],
            "model_mae_tha"        : 0.19,
            "national_avg_tha"     : 1.84,
            "vs_national_pct"      : round((yield_ - 1.84) / 1.84 * 100, 1),
            "insights"             : get_insights(farm, yield_),
            # SHAP explanation
            "shap_base_value"      : round(EXPECTED_VALUE, 4),
            "shap_values"          : shap_dict,
            "shap_top_drivers"     : top_drivers,
            "shap_explanation"     : explanation,
            "shap_note"            : (
                f"Base yield across all training farms: "
                f"{EXPECTED_VALUE:.2f} t/ha. "
                f"SHAP values show how each factor moved this farm's "
                f"predicted yield from that base to {yield_:.3f} t/ha. "
                f"Values are in tonnes per hectare."
            ),
            "processing_ms"        : round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"SHAP explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(farms: List[FarmInput]):
    """Score 1–200 farms. Returns all predictions + summary stats."""
    if len(farms) > 200:
        raise HTTPException(status_code=422,
                            detail="Max 200 farms per batch")
    start   = time.time()
    results = []
    for farm in farms:
        X_s    = scaler.transform(build_row(farm))
        yield_ = max(0.0, round(float(model.predict(X_s)[0]), 3))
        results.append({
            "county"       : farm.county,
            "crop"         : farm.crop,
            "farm_size_ha" : farm.farm_size_ha,
            "yield_tha"    : yield_,
            "total_tonnes" : round(yield_ * farm.farm_size_ha, 2),
            "vs_national"  : round((yield_ - 1.84) / 1.84 * 100, 1),
        })
    avg_yield  = round(sum(r["yield_tha"]    for r in results) / len(results), 3)
    total_prod = round(sum(r["total_tonnes"] for r in results), 2)
    return {
        "total_farms"        : len(results),
        "avg_yield_tha"      : avg_yield,
        "total_production_t" : total_prod,
        "national_avg_tha"   : 1.84,
        "vs_national_pct"    : round((avg_yield - 1.84) / 1.84 * 100, 1),
        "predictions"        : results,
        "processing_ms"      : round((time.time() - start) * 1000, 2),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)}
    )
