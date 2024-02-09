import streamlit as st
import pickle
import pandas as pd

# Setting Webpage Configurations
st.set_page_config(page_icon="⚙", page_title="Car Price Prediction", layout="wide")

st.title(":green[AutoWise] - Your Car Price Predictor 🚀")


@st.cache_resource
def load_model():
    model = pickle.load(
        open(
            r"F:\GUVI_DATA_SCIENCE\Project\CarDekho-Used-Car-Price-Prediction\Artifacts\model.pkl",
            "rb",
        )
    )
    return model


model = load_model()

transformer = pickle.load(
    open(
        r"F:\GUVI_DATA_SCIENCE\Project\CarDekho-Used-Car-Price-Prediction\Artifacts\transformer.pkl",
        "rb",
    )
)


df = pd.read_csv(
    r"F:\GUVI_DATA_SCIENCE\Project\CarDekho-Used-Car-Price-Prediction\Datasets\Final\processed_final_data.csv"
)


col1, col2, col3 = st.columns(3)

col4, col5, col6 = st.columns(3)

col7, col8, col9 = st.columns(3)

col10, col11, col12 = st.columns(3)

col13, col14, col15 = st.columns(3)

col16, col17, col18, col19 = st.columns(4)

with col1:
    manufacturer = st.selectbox(
        "Select the manufacturer",
        options=df["manufacturer"].value_counts().index.sort_values(),
    )

with col2:
    body_type = st.selectbox(
        "Select the body type",
        options=df["body_type"].value_counts().index.sort_values(),
    )

with col3:
    year = st.selectbox(
        "Select the model year",
        options=df["model_year"].value_counts().index.sort_values(),
    )

with col4:
    transmission_type = st.selectbox(
        "Select the transmission type",
        options=df["transmission_type"].value_counts().index.sort_values(),
    )

with col5:
    fuel_type = st.selectbox(
        "Select the fuel type",
        options=df["fuel_type"].value_counts().index.sort_values(),
    )

with col6:
    insurance_type = st.selectbox(
        "Select the insurance type",
        options=df["insurance_type"].value_counts().index.sort_values(),
    )

with col7:
    drive_type = st.selectbox(
        "Select the drive type",
        options=df["drive_type"].value_counts().index.sort_values(),
    )

with col8:
    steering_type = st.selectbox(
        "Select the steering type",
        options=df["steering_type"].value_counts().index.sort_values(),
    )

with col9:
    super_charger = st.selectbox(
        "Super charger", options=df["super_charger"].value_counts().index
    )

with col10:
    turbo_charger = st.selectbox(
        "Turbo charger", options=df["turbo_charger"].value_counts().index
    )

with col11:
    total_owners = st.number_input("Total owners")

with col12:
    total_kms = st.number_input("Total kms")

with col13:
    mileage = st.number_input("mileage(kmpl)")

with col14:
    engine = st.number_input("engine(CC)")

with col15:
    torque = st.number_input("torque(nm)")

with col16:
    seats = st.number_input("Number of Seats")

with col17:
    number_of_cylinders = st.number_input("Total Number of cylinders")

with col18:
    valves_per_cylinder = st.number_input("Total valves / cylinder")

with col19:
    cargo_volume = st.number_input("Cargo volume")

user_df = pd.DataFrame(
    data=[
        [
            manufacturer,
            year,
            body_type,
            transmission_type,
            fuel_type,
            total_kms,
            total_owners,
            insurance_type,
            mileage,
            engine,
            torque,
            seats,
            number_of_cylinders,
            valves_per_cylinder,
            super_charger,
            turbo_charger,
            drive_type,
            steering_type,
            cargo_volume,
        ]
    ],
    columns=[
        "manufacturer",
        "model_year",
        "body_type",
        "transmission_type",
        "fuel_type",
        "total_kms",
        "total_owners",
        "insurance_type",
        "mileage(kmpl)",
        "engine(CC)",
        "torque(nm)",
        "seats",
        "number_of_cylinders",
        "valves_per_cylinder",
        "super_charger",
        "turbo_charger",
        "drive_type",
        "steering_type",
        "cargo_volume",
    ],
)

st.dataframe(user_df, height=70, hide_index=True)

submit = st.button("Predict Car Price")

if submit:
    user_df_transformed = transformer.transform(user_df)

    result = model.predict(user_df_transformed)
    st.subheader(f":green[Predicted Car Price] : ₹ {round(result[0], 2)}")