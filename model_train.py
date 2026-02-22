import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df= pd.read_csv("insurance.csv")
df.head()
df.describe()
df.shape

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.4, 24.9, 29.9, float('inf')], labels=['Underweight','Normal weight','Overweight','Obesity'])
print(df['bmi_category'])
q1 = df['charges'].quantile(0.25)
q3 = df['charges'].quantile(0.75)
IQR = q3 - q1

outliers = df[(df['charges'] < (q1 - 1.5 * IQR)) |
              (df['charges'] > (q3 + 1.5 * IQR))]

print(outliers)

scaler = StandardScaler()
df[['age', 'bmi', 'charges']] = scaler.fit_transform(df[['age', 'bmi', 'charges']])
df.head()

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region", "bmi_category"]

numeric_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X = df.drop("charges", axis=1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_lr = LinearRegression()
reg_rf = RandomForestRegressor(n_estimators=300, random_state=42)
reg_gb = GradientBoostingRegressor(random_state=42)

voting_reg = VotingRegressor(
    estimators=
     [("lr", reg_lr),
     ("rf", reg_rf),
     ("gb", reg_gb)]
)

stacking_reg = StackingRegressor(
    estimators=[("lr", reg_lr), ("rf", reg_rf), ("gb", reg_gb)],
    final_estimator=Ridge(alpha=1.0),
    passthrough=False
)

model_to_train = {
    "Linear Regression": reg_lr,
    "Random Forest": reg_rf,
    "Gradient Boosting": reg_gb,
    "Voting Ensemble": voting_reg,
    "Stacking Ensemble": stacking_reg
}

result=[]
for name, model in model_to_train.items():
   pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
   pipe.fit(X_train, y_train)
   y_pred = pipe.predict(X_test)

   r2 = r2_score(y_test, y_pred)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   mae = mean_absolute_error(y_test, y_pred)

   result.append({
       "Model": name,
       "R2 Score": r2,
       "RMSE": rmse,
       "MAE": mae
   })

result_df = pd.DataFrame(result).sort_values(by="R2 Score", ascending=False)
print(result_df)

rf_model = RandomForestRegressor(n_estimators=300,random_state=42)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

cv_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring="neg_mean_squared_error")

print(cv_scores)
print(cv_scores.mean())
print(cv_scores.std())

rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 4],
    "model__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=rf_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    return_train_score=True
)

grid.fit(X_train, y_train)

cv_results = pd.DataFrame(grid.cv_results_)
results_table = cv_results[
    ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
].sort_values("rank_test_score")

print(results_table.head(10))
print(grid.best_params_)
print(grid.best_score_)

{
 'model__n_estimators': 400,
 'model__max_depth': 10,
 'model__min_samples_split': 2,
 'model__min_samples_leaf': 1,
 'model__max_features': 'sqrt'
}

final_model = grid.best_estimator_

y_test_pred = final_model.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)

print(r2)
print(rmse)
print(mae)

import pickle

filename = "random_forest_model.pkl"

with open(filename, "wb") as file:
    pickle.dump(grid.best_estimator_, file)

with open(filename, "rb") as file:
    loaded_model = pickle.load(file)

loaded_model.predict(X_test)  


import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

mlflow.set_experiment("Insurance Charges Prediction using Random Forest")

best_params = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42
}

final_rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(**best_params))
])

with mlflow.start_run(run_name="Final_Tuned_RandomForest"):

  
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", "RandomForestRegressor")

  
    final_rf_pipeline.fit(X_train, y_train)

 
    y_train_pred = final_rf_pipeline.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

   
    y_test_pred = final_rf_pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

   
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)

   
    mlflow.sklearn.log_model(
        sk_model=final_rf_pipeline,
        artifact_path="random_forest_model"
    )

print("MLflow run successfully")