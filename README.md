# Insurance Charges Prediction using Machine Learning

A machine learning project that predicts medical insurance charges based on customer attributes such as age, sex, BMI, number of children, smoking status, and region. The best-performing model is deployed as an interactive web app using **Gradio** and hosted on **Hugging Face Spaces**.

## Live Demo

[Insurance Charges Prediction on Hugging Face Spaces](https://huggingface.co/spaces/shajedul/Insurance-Charges-Prediction)

## Dataset

The dataset (`insurance.csv`) contains **1,338 records** with the following features:

| Feature    | Description                                      |
|------------|--------------------------------------------------|
| `age`      | Age of the primary beneficiary                   |
| `sex`      | Gender (male / female)                           |
| `bmi`      | Body Mass Index                                  |
| `children` | Number of dependents covered by insurance        |
| `smoker`   | Smoking status (yes / no)                        |
| `region`   | Residential area (southwest, southeast, northwest, northeast) |
| `charges`  | Individual medical costs billed by insurance (target) |

## Project Structure

```
├── app.py              # Gradio web application
├── model_train.py      # Model training, evaluation & hyperparameter tuning
├── insurance.csv       # Dataset
├── requirments.txt     # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

## ML Pipeline

### 1. Data Preprocessing
- Label encoding for categorical features (`sex`, `smoker`, `region`)
- BMI categorization into `Underweight`, `Normal weight`, `Overweight`, `Obesity`
- Outlier detection using IQR method
- Standard scaling for numerical features
- `ColumnTransformer` with `SimpleImputer` + `StandardScaler` for numeric features and `OneHotEncoder` for categorical features

### 2. Models Trained & Compared

| Model                | R² Score |
|----------------------|----------|
| Stacking Ensemble    | 0.8784   |
| Gradient Boosting    | 0.8784   |
| Voting Ensemble      | 0.8694   |
| Random Forest        | 0.8620   |
| Linear Regression    | 0.7792   |

### 3. Hyperparameter Tuning (GridSearchCV)
The **Random Forest Regressor** was tuned using 5-fold cross-validation with the following search space:

- `n_estimators`: [200, 400]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 10]
- `min_samples_leaf`: [1, 4]
- `max_features`: ["sqrt", "log2"]

**Best Parameters:**
```
n_estimators: 400
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
```

### 4. Experiment Tracking
- **MLflow** is used to log parameters, metrics, and the trained model artifact.

## Installation & Usage

### Prerequisites
- Python 3.10+

### Setup
```bash
# Clone the repository
git clone https://github.com/Rafi12234/Insurance_charges_prediction-using-Machine-Learning.git
cd Insurance_charges_prediction-using-Machine-Learning

# Install dependencies
pip install -r requirments.txt

# Train the model (generates random_forest_model.pkl)
python model_train.py

# Run the Gradio app
python app.py
```

The app will launch locally and provide a shareable public URL.

## Technologies Used

- **Python** — Core language
- **scikit-learn** — ML models, preprocessing, and evaluation
- **Gradio** — Interactive web UI
- **MLflow** — Experiment tracking
- **pandas / NumPy** — Data manipulation
- **Hugging Face Spaces** — Cloud deployment

## License

This project is open source and available under the [MIT License](LICENSE).