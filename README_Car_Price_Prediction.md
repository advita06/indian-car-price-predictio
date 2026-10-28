# Indian Pre-Owned Car Price Prediction

An end-to-end machine learning project that predicts the resale price of pre-owned cars in India and provides an interactive **Streamlit web application** for individual and batch predictions.

The project is organized into two components:
- **Component 1A — Machine Learning Pipeline:** data exploration, preprocessing, model training, evaluation, tuning, and prediction generation.
- **Component 1B — Streamlit App:** interactive car valuation, batch CSV predictions, and data insights.

## Objective

Predict the price of a used car using characteristics such as maker, model, location, distance driven, age, engine details, fuel type, transmission, ownership type, and other available attributes.

## Dataset

The project uses three files:
- **Training dataset:** contains car features and the target `Price`.
- **Test dataset:** contains car features without `Price`.
- **Sample submission:** specifies the required output format.

The project summary reports approximately **42,800 training cars** and **5,615 test cars**.

## Component 1A: Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)

The analysis covered:
- Missing values, duplicate rows, data types, and price distribution.
- Outliers using the IQR rule and boxplots.
- Distributions of distance driven, car age, and engine power.
- Category counts and correlations with price.
- Relationships between price and fuel type, owner type, transmission, age, and distance driven.

### 2. Data Cleaning

- Removed **117 rows** with prices below ₹50,000, which were treated as impossible prices for this project.
- Dropped `ID` because it is an identifier rather than a predictive feature.
- Dropped `body_type` because most of its values were missing.
- Dropped `manufacture_year` because it duplicated information represented by `Age of car`.

### 3. Preprocessing

The dataset was split into **80% training and 20% validation** before fitting the preprocessing pipeline.

A reusable `ColumnTransformer` was used:
- **Numeric features:** median imputation and standard scaling.
- **Categorical features:** most-frequent imputation and one-hot encoding.

Categorical features included:
- `Maker`
- `model`
- `Location`
- `Owner Type`
- `transmission`
- `fuel_type`

The same fitted preprocessor was reused for test data to keep training and prediction transformations consistent.

### 4. Models Compared

The following models were evaluated:
- Mean-price baseline
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

### 5. Tuning and Evaluation

- Used **5-fold cross-validation**.
- Applied `RandomizedSearchCV` to Random Forest and XGBoost, using cross-validated MAE for model selection.
- Compared models on the same validation set using:
  - **MAE** — Mean Absolute Error
  - **RMSE** — Root Mean Squared Error
  - **R²** — coefficient of determination

The project summary reports that tree-based models clearly outperformed Linear Regression and the baseline.

### 6. Final Model and Predictions

The final model was a **tuned XGBoost model trained on log-transformed prices** to account for the heavily skewed price distribution. Predictions were converted back to rupees.

Test predictions were generated using the same fitted preprocessor and saved to `my_submission.csv` in the required `ID, Price` format.

## Component 1B: Streamlit Web Application

The app provides three main capabilities.

### Individual Car Valuation

Users can enter car details through sidebar controls, including:
- Maker and model
- Location
- Fuel type and transmission
- Owner type
- Distance driven and car age
- Engine details
- Seats and doors
- Vroom audit rating

The app loads the saved model and preprocessor with `joblib` and `st.cache_resource` to support quick predictions.

It displays an estimated car price with a **fair-price range of ±5%** around the estimate.

### Batch Processing

- Upload a test CSV file.
- Generate predictions for multiple cars.
- Download the results as a CSV file.

### Insights Dashboard

Native Streamlit charts explore how car price varies with:
- Car age
- Distance driven
- Fuel type
- Maker

## Workflow

```text
Car Dataset
    ↓
EDA and Data Cleaning
    ↓
Train / Validation Split
    ↓
Reusable Preprocessing Pipeline
    ↓
Model Training and Hyperparameter Tuning
    ↓
Validation with MAE, RMSE and R²
    ↓
Tuned XGBoost Model
    ↓
Test Predictions → my_submission.csv
    ↓
Streamlit App: Individual Valuation + Batch Processing + Insights
```

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- LightGBM
- joblib
- Streamlit
- Matplotlib / Seaborn for EDA visualizations, if used in the notebook

## Running the Streamlit App

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Then start the app:

```bash
streamlit run app.py
```

Ensure the saved model, fitted preprocessor, and any required data files are available at the paths expected by `app.py`.

## Project Outputs

- `my_submission.csv` — test predictions in the required `ID, Price` format.
- Saved model and preprocessor files used by the Streamlit application.
- Interactive Streamlit interface for single-car valuation, batch predictions, and insights.

## Limitations and Future Improvements

The supplied project summary does not provide final numerical validation metrics or the exact hyperparameters of the tuned XGBoost model. These can be added to this README once confirmed from the notebook.

Possible future improvements include evaluating the app with new real-world listings, monitoring prediction errors over time, and expanding the dataset to cover more car models and market conditions.

---

**Note:** The estimated price and ±5% fair-price band are model-based estimates, not guarantees of a car's actual market value.
