import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.set_page_config(
    page_title="Spaceship Titanic Predictor",
    page_icon = "##",
    layout = "centered"
)
@st.cache_resource
def load_model():
    model = joblib.load('spaceship_model.pkl')
    return model
model = load_model()
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Spending
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend']    = df[spend_cols].sum(axis=1)
    df['HasSpent']      = (df['TotalSpend'] > 0).astype(int)
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
    for col in spend_cols:
        df[f'Log_{col}']   = np.log1p(df[col])
        df[f'Ratio_{col}'] = df[col] / (df['TotalSpend'] + 1)

    # CryoSleep override
    if df['CryoSleep'].values[0] == 1:
        for col in spend_cols:
            df[col] = 0
        df['TotalSpend']    = 0
        df['HasSpent']      = 0
        df['LogTotalSpend'] = 0

    # Group features
    df['GroupSize']      = 1
    df['IsAlone']        = 1
    df['CabinGroupSize'] = 1
    df['GroupSpendMean'] = df['TotalSpend']
    df['GroupCryoMean']  = df['CryoSleep']

    # Age group
    age = df['Age'].values[0]
    if age <= 12:   df['AgeGroup'] = 0   # Child
    elif age <= 18: df['AgeGroup'] = 4   # Teen
    elif age <= 35: df['AgeGroup'] = 1   # Adult
    elif age <= 60: df['AgeGroup'] = 2   # Middle
    else:           df['AgeGroup'] = 3   # Senior

    # Cabin
    df['CabinNum']  = df['CabinNum'].fillna(500).astype(int)
    df['CabinBand'] = min(df['CabinNum'].values[0] // 200, 9)

    # Encode categoricals
    home_map    = {'Earth': 0, 'Europa': 1, 'Mars': 2}
    dest_map    = {'55 Cancri e': 0, 'PSO J318.5-22': 1, 'TRAPPIST-1e': 2}
    deck_map    = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}
    side_map    = {'P': 0, 'S': 1}
    planet_dest = f"{df['HomePlanet'].values[0]}_{df['Destination'].values[0]}"
    planet_dest_map = {
        'Earth_55 Cancri e': 0, 'Earth_PSO J318.5-22': 1, 'Earth_TRAPPIST-1e': 2,
        'Europa_55 Cancri e': 3, 'Europa_PSO J318.5-22': 4, 'Europa_TRAPPIST-1e': 5,
        'Mars_55 Cancri e': 6, 'Mars_PSO J318.5-22': 7, 'Mars_TRAPPIST-1e': 8
    }

    df['HomePlanet']  = home_map.get(df['HomePlanet'].values[0], 0)
    df['Destination'] = dest_map.get(df['Destination'].values[0], 2)
    df['Deck']        = deck_map.get(df['Deck'].values[0], 5)
    df['Side']        = side_map.get(df['Side'].values[0], 0)
    df['Planet_Dest'] = planet_dest_map.get(planet_dest, 0)

    # Final features — same order as training
    features = [
    'HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'Group', 'Member', 'GroupSize', 'IsAlone',
    'Deck', 'CabinNum', 'Side',
    'TotalSpend', 'HasSpent', 'LogTotalSpend',
    'Log_RoomService', 'Ratio_RoomService',
    'Log_FoodCourt', 'Ratio_FoodCourt',
    'Log_ShoppingMall', 'Ratio_ShoppingMall',
    'Log_Spa', 'Ratio_Spa',
    'Log_VRDeck', 'Ratio_VRDeck',
    'AgeGroup', 'CabinBand', 'CabinGroupSize',
    'GroupSpendMean', 'GroupCryoMean', 'Planet_Dest'
]

    return df[features]

# ============================================================
# UI
# ============================================================
st.title("🚀 Spaceship Titanic — Passenger Predictor")
st.markdown("Fill in passenger details to predict if they were **transported** to another dimension!")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Info")
    group      = st.number_input("Group Number", min_value=1, max_value=9999, value=1)
    member     = st.number_input("Member Number", min_value=1, max_value=9, value=1)
    age        = st.slider("Age", 0, 100, 25)
    home       = st.selectbox("Home Planet", ['Earth', 'Europa', 'Mars'])
    dest       = st.selectbox("Destination", ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22'])
    cryo       = st.toggle("CryoSleep 🧊", value=False)
    vip        = st.toggle("VIP 💎", value=False)

with col2:
    st.subheader("🛸 Cabin Info")
    deck       = st.selectbox("Deck", ['A','B','C','D','E','F','G','T'])
    cabin_num  = st.number_input("Cabin Number", min_value=0, max_value=2000, value=500)
    side       = st.selectbox("Side", ['P', 'S'])

    st.subheader("💸 Spending (credits)")
    room       = st.number_input("Room Service",   min_value=0.0, value=0.0, step=10.0)
    food       = st.number_input("Food Court",     min_value=0.0, value=0.0, step=10.0)
    shop       = st.number_input("Shopping Mall",  min_value=0.0, value=0.0, step=10.0)
    spa        = st.number_input("Spa",            min_value=0.0, value=0.0, step=10.0)
    vr         = st.number_input("VR Deck",        min_value=0.0, value=0.0, step=10.0)

st.divider()

# ============================================================
# PREDICT
# ============================================================
if st.button("🔮 Predict", use_container_width=True, type="primary"):
    input_data = {
        'Group': group, 'Member': member,
        'Age': age, 'HomePlanet': home, 'Destination': dest,
        'CryoSleep': int(cryo), 'VIP': int(vip),
        'Deck': deck, 'CabinNum': cabin_num, 'Side': side,
        'RoomService': room, 'FoodCourt': food,
        'ShoppingMall': shop, 'Spa': spa, 'VRDeck': vr
    }

    try:
        X = preprocess_input(input_data)
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        st.divider()
        if pred:
            st.success("## ✅ TRANSPORTED!")
            st.markdown(f"This passenger was **transported** to another dimension.")
        else:
            st.error("## ❌ NOT TRANSPORTED")
            st.markdown(f"This passenger was **not transported**.")

        col_a, col_b = st.columns(2)
        col_a.metric("🚀 Transported Probability",   f"{proba[1]*100:.1f}%")
        col_b.metric("🌍 Not Transported Probability", f"{proba[0]*100:.1f}%")

        st.progress(float(proba[1]))

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure spaceship_model.pkl is in the same folder as app.py")

