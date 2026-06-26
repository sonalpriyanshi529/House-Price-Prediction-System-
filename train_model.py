
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE, "Dataset", "train.csv")
model_dir = os.path.join(BASE, "Model")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(data_path)

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols)
])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

models={
    "Linear Regression":LinearRegression(),
    "Random Forest":RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1),
    "XGBoost":XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

best_name=None
best_pipeline=None
best_r2=-1e9
best_metrics=None

for name,model in models.items():
    pipe=Pipeline([
        ("preprocessor",preprocessor),
        ("model",model)
    ])
    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)
    mae=mean_absolute_error(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    rmse=mse**0.5
    r2=r2_score(y_test,pred)
    print(f"{name}: R2={r2:.4f}, RMSE={rmse:.2f}")
    if r2>best_r2:
        best_r2=r2
        best_name=name
        best_pipeline=pipe
        best_metrics={
            "Best Model":name,
            "MAE":mae,
            "MSE":mse,
            "RMSE":rmse,
            "R2 Score":r2
        }

joblib.dump(best_pipeline, os.path.join(model_dir,"best_model.pkl"))
joblib.dump(best_metrics, os.path.join(model_dir,"model_metrics.pkl"))

print("\nBest model:",best_name)
print(best_metrics)
print("Saved to",model_dir)
