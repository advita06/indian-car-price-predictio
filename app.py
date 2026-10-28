"""
Indian Pre-Owned Car Price Predictor (Streamlit app)

Run with:  streamlit run app.py
Needs in the same folder:
  - car_price_model.joblib      (XGBoost trained on log price)
  - preprocessor.joblib         (fitted ColumnTransformer)
  - 12573233-Indian_Car_Market_Capstone_2025_Dataset/Cap_Training_Data_2025.csv
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st

TRAIN_PATH = "12573233-Indian_Car_Market_Capstone_2025_Dataset/Cap_Training_Data_2025.csv"

# Columns the preprocessor expects (note the trailing space in 'Distance ')
FEATURES = ['Maker', 'model', 'Location', 'Distance ', 'Owner Type', 'Age of car',
            'engine_displacement', 'engine_power', 'Vroom Audit Rating',
            'transmission', 'door_count', 'seat_count', 'fuel_type']

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")


# ---------- Loading (cached) ----------
@st.cache_resource
def load_artifacts():
    """Load the trained model and the fitted preprocessor once."""
    model = joblib.load("car_price_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor


@st.cache_data
def load_training_data():
    """Load training data for dropdown options and the insights charts."""
    df = pd.read_csv(TRAIN_PATH)
    df.columns = df.columns.str.strip()          # clean names for display use
    df = df[df["Price"] >= 50000]                # same outlier removal as notebook
    return df


model, preprocessor = load_artifacts()
train = load_training_data()


# ---------- Prediction helper ----------
def predict_price(df: pd.DataFrame) -> np.ndarray:
    """Raw feature DataFrame -> price in rupees (model predicts log price)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Distance": "Distance "})   # match training column name
    X = preprocessor.transform(df[FEATURES])
    if hasattr(X, "toarray"):                            # model was trained on dense data
        X = X.toarray()
    return np.expm1(model.predict(X))


st.title("🚗 Indian Pre-Owned Car Price Predictor")
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Processing", "Insights"])

# ---------- Tab 1: single prediction ----------
with tab1:
    st.sidebar.header("Car details")
    maker = st.sidebar.selectbox("Maker", sorted(train["Maker"].dropna().unique()))
    models_for_maker = sorted(train.loc[train["Maker"] == maker, "model"].dropna().unique())
    car_model = st.sidebar.selectbox("Model", models_for_maker)
    location = st.sidebar.selectbox("Location", sorted(train["Location"].dropna().unique()))
    fuel = st.sidebar.selectbox("Fuel type", sorted(train["fuel_type"].dropna().unique()))
    transmission = st.sidebar.selectbox("Transmission", sorted(train["transmission"].dropna().unique()))
    owner = st.sidebar.selectbox("Owner type", sorted(train["Owner Type"].dropna().unique()))
    distance = st.sidebar.slider("Distance driven (km)", 0, 300000, 50000, step=1000)
    age = st.sidebar.slider("Age of car (years)", 0, 20, 5)

    # Engine / body details default to the typical values for the chosen model
    sub = train[(train["Maker"] == maker) & (train["model"] == car_model)]
    def typical(col, fallback):
        v = sub[col].median() if col in sub else np.nan
        return float(v) if pd.notna(v) else float(fallback)

    engine_disp = st.sidebar.number_input("Engine displacement (cc)", 500, 8000,
                                          int(typical("engine_displacement", 1500)), step=50)
    engine_power = st.sidebar.number_input("Engine power (kW)", 20, 600,
                                           int(typical("engine_power", 90)), step=5)
    seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)
    doors = st.sidebar.selectbox("Doors", [2, 3, 4, 5], index=2)
    rating = st.sidebar.slider("Vroom audit rating", 0, 10, 5)

    st.subheader(f"{maker.title()} {car_model.title()}, {age} yrs, {distance:,} km")

    if st.button("Estimate price", type="primary"):
        row = pd.DataFrame([{
            "Maker": maker, "model": car_model, "Location": location,
            "Distance ": distance, "Owner Type": owner, "Age of car": age,
            "engine_displacement": engine_disp, "engine_power": engine_power,
            "Vroom Audit Rating": rating, "transmission": transmission,
            "door_count": doors, "seat_count": seats, "fuel_type": fuel,
        }])
        price = predict_price(row)[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated price", f"₹{price:,.0f}")
        c2.metric("Fair range (low, -5%)", f"₹{price * 0.95:,.0f}")
        c3.metric("Fair range (high, +5%)", f"₹{price * 1.05:,.0f}")
        st.caption("The model's typical error is about 10%, so treat this as a guide.")

# ---------- Tab 2: batch processing ----------
with tab2:
    st.write("Upload a test CSV (same columns as the Cap_Test_Data file) to predict in bulk.")
    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        try:
            batch = pd.read_csv(file)
            out = batch.copy()
            out["Price"] = predict_price(batch)
            cols = ["ID", "Price"] if "ID" in out.columns else ["Price"]
            st.success(f"Predicted {len(out):,} cars.")
            st.dataframe(out.head(20))
            st.download_button("Download predictions (ID, Price)",
                               out[cols].to_csv(index=False),
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not predict. Check the column names. Details: {e}")

# ---------- Tab 3: insights ----------
with tab3:
    st.subheader("Average price by age of car")
    st.line_chart(train.groupby("Age of car")["Price"].mean())

    st.subheader("Average price by distance driven")
    bins = pd.cut(train["Distance"], bins=[0, 10000, 25000, 50000, 75000, 100000,
                                           150000, 200000, 500000])
    st.bar_chart(train.groupby(bins, observed=True)["Price"].mean().rename(lambda x: str(x)))

    st.subheader("Average price by fuel type")
    st.bar_chart(train.groupby("fuel_type")["Price"].mean())

    st.subheader("Top 10 makers by average price")
    st.bar_chart(train.groupby("Maker")["Price"].mean().sort_values(ascending=False).head(10))
