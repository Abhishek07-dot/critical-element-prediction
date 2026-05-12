# =========================================================
# CRITICAL ELEMENT PREDICTION MODEL
# XGBoost + DEM + Residual Kriging + Visualization
# =========================================================

import ee
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from pykrige.ok import OrdinaryKriging

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KDTree
from pyproj import Transformer

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# INITIALIZE GEE
# =========================================================

ee.Initialize(project='prediction-488105')

# =========================================================
# DEM STACK
# =========================================================

dem = ee.Image("USGS/SRTMGL1_003")

terrain = ee.Terrain.products(dem)

slope = terrain.select("slope")
aspect = terrain.select("aspect")

curvature = dem.convolve(
    ee.Kernel.laplacian8()
).rename("curvature")

twi = slope.expression(
    "log((100 + 1) / (tan(slope * 0.01745) + 0.001))",
    {"slope": slope}
).rename("twi")

stack = dem.rename("elevation") \
    .addBands(slope.rename("slope")) \
    .addBands(aspect.rename("aspect")) \
    .addBands(curvature) \
    .addBands(twi)

# =========================================================
# DEM EXTRACTION FUNCTION
# =========================================================

def get_dem(lat, lon):

    point = ee.Geometry.Point(lon, lat)

    sample = stack.sample(
        region=point,
        scale=30,
        numPixels=1
    ).first()

    props = sample.getInfo()["properties"]

    return (
        props["elevation"],
        props["slope"],
        props["aspect"],
        props["curvature"],
        props["twi"]
    )

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\VS_Code\Lithium_with_DEM.csv"
).dropna().reset_index(drop=True)

print("Dataset loaded:", df.shape)

# =========================================================
# COORDINATE TRANSFORMATION
# =========================================================

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32617",
    always_xy=True
)

df["easting"], df["northing"] = transformer.transform(
    df["long"].values,
    df["lat"].values
)

# =========================================================
# LOCAL NEIGHBOR FEATURE
# =========================================================

coords = df[["easting","northing"]].values

tree = KDTree(coords)

local_mean = []

for i, point in enumerate(coords):

    idx = tree.query_radius([point], r=10000)[0]

    idx = idx[idx != i]

    if len(idx) == 0:
        local_mean.append(df["Y_mean_c"].mean())
    else:
        local_mean.append(
            df.iloc[idx]["Y_mean_c"].mean()
        )

df["local_lithium_mean"] = local_mean

# =========================================================
# FEATURE SET
# =========================================================

features = [
    "easting",
    "northing",
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "twi",
    "local_lithium_mean"
]

X = df[features]

y = df["Y_mean_c"]

# =========================================================
# TRAIN XGBOOST MODEL
# =========================================================

model = XGBRegressor(
    n_estimators=900,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    reg_alpha=1,
    random_state=42
)

model.fit(X, y)

print("Model trained")

# =========================================================
# MODEL METRICS
# =========================================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

ml_rmse = -cv_scores.mean()

pred_train = model.predict(X)

mae = mean_absolute_error(y, pred_train)

rmse = np.sqrt(
    mean_squared_error(y, pred_train)
)

r2 = r2_score(y, pred_train)

print("\nModel Performance Metrics")

print("MAE :", round(mae,3))

print("RMSE:", round(rmse,3))

print("R²  :", round(r2,3))

print("Estimated ML RMSE:",
      round(ml_rmse,3))

# =========================================================
# RESIDUAL KRIGING MODEL
# =========================================================

trend_pred = model.predict(X)

residuals_ml = y - trend_pred

OK = OrdinaryKriging(
    df["easting"].values,
    df["northing"].values,
    residuals_ml.values,
    variogram_model="exponential",
    verbose=False,
    enable_plotting=False
)

print("Residual kriging model created")

# =========================================================
# GENERATE FULL DATASET PREDICTIONS
# =========================================================

print("Generating predictions...")

residual_pred, var = OK.execute(
    "points",
    df["easting"].values,
    df["northing"].values
)

pred = trend_pred + residual_pred

residuals = y - pred

lower_bounds = pred - ml_rmse

upper_bounds = pred + ml_rmse

print("Predictions generated.")

# =========================================================
# VISUALIZATION SETTINGS
# =========================================================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 120
})

# =========================================================
# EXTENDED VISUALIZATION SECTION
# =========================================================

print("Generating visualization plots...")

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 120
})

# ---------------------------------------------------------
# 1 HISTOGRAM
# ---------------------------------------------------------

plt.figure(figsize=(9,5))

sns.histplot(
    df["Y_mean_c"],
    bins=40,
    kde=True,
    color="#2E86AB"
)

plt.title("Distribution of Lithium Concentration")

plt.xlabel("Concentration (ppm)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2 BOX PLOT
# ---------------------------------------------------------

plt.figure(figsize=(6,5))

sns.boxplot(
    y=df["Y_mean_c"],
    color="#E69F00"
)

plt.title("Box Plot of Concentration")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3 SAMPLE LOCATION MAP
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

plt.scatter(
    df["long"],
    df["lat"],
    s=10,
    c="#1B9E77",
    alpha=0.6,
    edgecolor="black",
    linewidth=0.2
)

plt.title("Spatial Distribution of Sample Locations")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4 ELEVATION vs CONCENTRATION
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df["elevation"],
    df["Y_mean_c"],
    c=df["Y_mean_c"],
    cmap="plasma",
    alpha=0.7
)

plt.colorbar(label="Concentration")

plt.xlabel("Elevation")
plt.ylabel("Concentration")

plt.title("Elevation vs Concentration")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5 CORRELATION HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(11,9))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6 OBSERVED vs PREDICTED
# ---------------------------------------------------------

plt.figure(figsize=(7,6))

plt.scatter(
    y,
    pred,
    c=pred,
    cmap="viridis",
    alpha=0.6
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--",
    linewidth=2
)

plt.xlabel("Observed")
plt.ylabel("Predicted")

plt.title("Observed vs Predicted")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7 RESIDUAL PLOT
# ---------------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(
    pred,
    residuals,
    c=residuals,
    cmap="coolwarm",
    alpha=0.7
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--"
)

plt.xlabel("Predicted")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8 FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = model.feature_importances_

pd.Series(
    importance,
    index=features
).sort_values().plot(
    kind="barh",
    figsize=(9,6),
    color="#E69F00"
)

plt.title("Feature Importance Ranking")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9 SPATIAL PREDICTION MAP
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

plt.scatter(
    df["long"],
    df["lat"],
    c=pred,
    cmap="turbo",
    s=15,
    alpha=0.8
)

plt.colorbar(
    label="Predicted Concentration"
)

plt.title("Spatial Prediction Map")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10 RESIDUAL MAP
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

plt.scatter(
    df["long"],
    df["lat"],
    c=residuals,
    cmap="coolwarm",
    s=15
)

plt.colorbar(
    label="Residual Error"
)

plt.title("Spatial Residual Map")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11 HEXBIN HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

plt.hexbin(
    df["long"],
    df["lat"],
    C=df["Y_mean_c"],
    gridsize=60,
    cmap="inferno"
)

plt.colorbar(
    label="Concentration"
)

plt.title("Spatial Concentration Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 12 VARIOGRAM
# ---------------------------------------------------------

OK.display_variogram_model()

# ---------------------------------------------------------
# 13 CONFIDENCE BAND PLOT
# ---------------------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(pred[:200], color="orange", label="Prediction")

plt.fill_between(
    range(200),
    lower_bounds[:200],
    upper_bounds[:200],
    color="blue",
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.xlabel("Sample Index")
plt.ylabel("Lithium Concentration (ppm)")
plt.title("Prediction with Confidence Interval")

plt.legend()

plt.show()

# ---------------------------------------------------------
# 14 KDE DENSITY MAP
# ---------------------------------------------------------

plt.figure(figsize=(9,5))

sns.kdeplot(
    x=df["elevation"],
    y=df["Y_mean_c"],
    cmap="viridis",
    fill=True
)

plt.title("Elevation–Concentration Density")

plt.tight_layout()
plt.show()

print("All plots generated successfully.")


# =========================================================
# USER PREDICTION LOOP
# =========================================================

def estimate_local_mean(e, n, radius=10000):

    idx = tree.query_radius(
        [[e, n]],
        r=radius
    )[0]

    if len(idx) == 0:
        return df["Y_mean_c"].mean()

    return df.iloc[idx]["Y_mean_c"].mean()

print("\nEnter coordinates to predict concentration.")

print("Type 'exit' to stop.\n")

while True:

    lat_input = input("Enter latitude: ")

    if lat_input.lower() == "exit":
        break

    lon_input = input("Enter longitude: ")

    if lon_input.lower() == "exit":
        break

    try:

        lat = float(lat_input)

        lon = float(lon_input)

    except ValueError:

        print("Invalid coordinates")

        continue

    elev, slope, aspect, curvature, twi_val = get_dem(
        lat,
        lon
    )

    e, n = transformer.transform(
        lon,
        lat
    )

    local_est = estimate_local_mean(e, n)

    user_X = pd.DataFrame({

        "easting":[e],
        "northing":[n],
        "elevation":[elev],
        "slope":[slope],
        "aspect":[aspect],
        "curvature":[curvature],
        "twi":[twi_val],
        "local_lithium_mean":[local_est]

    })

    trend = model.predict(user_X)

    residual, var = OK.execute(
        "points",
        np.array([e]),
        np.array([n]),
        n_closest_points=40,
        backend="loop"
    )

    prediction = trend + residual

    residual_q = np.percentile(
        residuals,
        [2.5, 97.5]
    )

    lower = prediction[0] + residual_q[0]

    upper = prediction[0] + residual_q[1]

    print("\nPrediction:",
          round(prediction[0],2),
          "ppm")

    print("95% confidence interval:",
          round(lower,2),
          "to",
          round(upper,2))

    print("--------------------------------")

print("\nPrediction session ended")