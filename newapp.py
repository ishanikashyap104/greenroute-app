# app.py
# SDG 13: Eco-Transport Route Planner + Learning Hub
# Built with Streamlit (Python)

import streamlit as st
import pandas as pd
import math
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import matplotlib.pyplot as plt

 # -----------------------------
# Page Config & Branding
# -----------------------------
st.set_page_config(
    page_title="GreenRoute 🌍",
    page_icon="https://raw.githubusercontent.com/ishanikashyap104/greenroute-app/main/icon.png",
    layout="wide"
)

st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #e6f7f1; border-radius: 12px; margin-bottom: 20px;">
        <h1 style="color: #2e8b57;">🌱 GreenRoute: Climate Action App 🌍</h1>
        <p style="font-size:18px;">An interactive project to support <b>SDG 13: Climate Action</b><br>
        Plan eco-friendly routes • Learn about climate change • Compare emissions</p>
    </div>
""", unsafe_allow_html=True)


# -----------------------------
# Emission factors (kg CO2 per km)
# -----------------------------
EMISSION_FACTORS = {
    "Car": 0.120,
    "Motorbike": 0.072,
    "Bus": 0.089,
    "Metro/Train": 0.041,
    "Cycle": 0.0,
    "Walk": 0.0,
    "Plane": 0.200
}

# -----------------------------
# Helper functions
# -----------------------------
def clean_address(s: str) -> str:
    return s.strip()

def distance_km_between(a_latlon, b_latlon) -> float:
    return round(geodesic(a_latlon, b_latlon).km, 2)

def compute_emissions(distance_km: float, mode: str) -> float:
    factor = EMISSION_FACTORS.get(mode, 0.0)
    return distance_km * factor

def best_mode_for(distance_km: float, modes_available: list[str]) -> str:
    best = None
    best_em = math.inf
    for m in modes_available:
        em = compute_emissions(distance_km, m)
        if em < best_em:
            best_em = em
            best = m
    return best

def emissions_table(distance_km: float, modes: list[str]) -> pd.DataFrame:
    rows = []
    for m in modes:
        rows.append({
            "Mode": m,
            "CO₂ per km (kg)": EMISSION_FACTORS[m],
            f"CO₂ for {distance_km} km (kg)": round(compute_emissions(distance_km, m), 3)
        })
    df = pd.DataFrame(rows).sort_values(by=f"CO₂ for {distance_km} km (kg)")
    return df

def show_bar_chart_emissions(distance_km: float, modes: list[str], highlight: str | None = None):
    labels = modes
    values = [compute_emissions(distance_km, m) for m in modes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=['#69b3a2' for _ in labels])

    ax.set_ylabel(f"CO₂ for {distance_km} km (kg)")
    ax.set_title("CO₂ Emissions by Transport Mode")

    if highlight and highlight in labels:
        idx = labels.index(highlight)
        bars[idx].set_color('#ff7f0e')  # highlight recommended mode
        ax.annotate("Recommended",
                    xy=(idx, values[idx]),
                    xytext=(idx, values[idx] * 1.05 + 0.01),
                    ha='center', arrowprops=dict(arrowstyle="->"))

    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

# -----------------------------
# Selectable transport modes based on domestic/international + distance
# -----------------------------
def selectable_modes(start_location, end_location, distance_km: float):
    geolocator = Nominatim(user_agent="sdg13_climate_app")
    s = geolocator.geocode(start_location, addressdetails=True)
    e = geolocator.geocode(end_location, addressdetails=True)
    
    if not s or not e:
        return []

    s_country = s.raw.get('address', {}).get('country')
    e_country = e.raw.get('address', {}).get('country')

    if s_country != e_country:
        # International → only Plane
        return ["Plane"]
    else:
        # Domestic
        if distance_km <= 5:
            return ["Walk", "Cycle", "Metro/Train", "Bus", "Car", "Motorbike"]
        elif distance_km <= 50:
            return ["Metro/Train", "Bus", "Motorbike", "Car"]
        else:
            # Long domestic trips → Plane optional, plus Bus/Car/Train
            return ["Metro/Train", "Bus", "Car", "Plane"]

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🌍 SDG 13 Climate Action")
page = st.sidebar.radio(
    "Navigate",
    ["🌱 Route Planner", "📘 Learn About SDG 13", "📊 Emission Comparison", "🌡️ Global Impact"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python • Streamlit")

# -----------------------------
# PAGE 1: Route Planner
# -----------------------------
if page == "🌱 Route Planner":
    st.title("🌱 Eco-Transport Route Planner")
    st.write("Enter your trip, compare transport options, and see how much CO₂ you can save vs. a typical car.")

    colA, colB = st.columns(2)
    with colA:
        start = st.text_input("Start location", placeholder="e.g., Connaught Place, New Delhi")
    with colB:
        end = st.text_input("Destination", placeholder="e.g., Nagaland, India")

    if start and end:
        try:
            geolocator = Nominatim(user_agent="sdg13_climate_app")
            s = geolocator.geocode(clean_address(start))
            e = geolocator.geocode(clean_address(end))
            if s and e:
                distance_km = distance_km_between((s.latitude, s.longitude), (e.latitude, e.longitude))
                st.success(f"Distance found: **{distance_km} km**")

                modes_available = selectable_modes(start, end, distance_km)
                st.markdown("**Available modes** (auto-selected based on trip type and distance):")
                modes_selected = st.multiselect(
                    "Transport modes",
                    options=list(EMISSION_FACTORS.keys()),
                    default=modes_available
                )

                if st.button("Calculate & Recommend"):
                    if not modes_selected:
                        st.warning("Please select at least one transport mode.")
                    else:
                        best = best_mode_for(distance_km, modes_selected)
                        car_em = compute_emissions(distance_km, "Car") if "Car" in EMISSION_FACTORS else None
                        best_em = compute_emissions(distance_km, best) if best else None
                        saved = (car_em - best_em) if (car_em and best_em) else None

                        st.subheader("Recommendation")
                        if best:
                            st.write(f"**Best mode for this trip:** `{best}`")
                        if saved is not None:
                            st.write(f"**CO₂ saved vs. Car:** `{saved:.3f} kg`")

                        df = emissions_table(distance_km, modes_selected)
                        st.dataframe(df, use_container_width=True)
                        show_bar_chart_emissions(distance_km, modes_selected, highlight=best)
            else:
                st.error("Could not geocode one or both locations. Try being more specific.")
        except Exception as ex:
            st.error(f"Something went wrong: {ex}")

# -----------------------------
# PAGE 2: Learn About SDG 13
# -----------------------------
elif page == "📘 Learn About SDG 13":
    st.title("📘 Learn About SDG 13 — Climate Action")
    st.write("""
**SDG 13** calls for urgent action to combat climate change and its impacts.
Greenhouse gases, especially CO₂ from burning fossil fuels (transport, electricity, industry), trap heat in the atmosphere and warm the planet.
This leads to more extreme weather, sea-level rise, and risks to food and water security.
""")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Why it matters")
        st.markdown("""
- **Rising temperatures** increase heatwaves, droughts, and extreme rainfall.
- **Melting ice & rising seas** threaten coastal communities and ecosystems.
- **Health impacts** include heat stress, spread of vector-borne diseases, and air pollution.
- **Economic risks** from climate-related disasters are growing.
        """)
        st.markdown("### What can individuals & schools do?")
        st.markdown("""
- Choose **low-carbon transport**: walk, cycle, bus, or train.
- Save electricity and adopt **energy-efficient appliances**.
- Reduce, reuse, recycle — cut waste and **avoid single-use plastics**.
- Plant trees and protect green spaces.
- Advocate for climate-smart policies in your community.
        """)
        st.markdown("### How this app helps")
        st.markdown("""
- It **compares transport options** for your trip and shows **CO₂ savings**.
- It **educates users** about climate change and SDG 13 with visuals.
- It encourages **behavior change** at a local level, which scales up when many people participate.
        """)
    # ✅ NEW QUIZ SECTION
    st.subheader("📝 Quick Quiz: Test your knowledge")
    with st.expander("Try answering these!"):
        q1 = st.radio("Which gas is the biggest driver of climate change?", 
                      ["Oxygen", "Carbon Dioxide (CO₂)", "Nitrogen"], key="q1")
        if q1 == "Carbon Dioxide (CO₂)":
            st.success("✅ Correct!")
        else:
            st.error("❌ Nope, it’s CO₂.")

        q2 = st.radio("Which agreement aims to limit global warming to 1.5°C?", 
                      ["Kyoto Protocol", "Paris Agreement", "Montreal Protocol"], key="q2")
        if q2 == "Paris Agreement":
            st.success("✅ Correct!")
        else:
            st.error("❌ The right answer is Paris Agreement.")

# -----------------------------
# PAGE 3: Emission Comparison
# -----------------------------
elif page == "📊 Emission Comparison":
    st.title("📊 CO₂ Emissions by Transport Mode (per km)")

    st.write("Compare the *average* CO₂ per km for common transport modes.")
    modes = list(EMISSION_FACTORS.keys())
    per_km = [EMISSION_FACTORS[m] for m in modes]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(modes, per_km)
    ax.set_ylabel("CO₂ per km (kg)")
    ax.set_title("Average CO₂ Intensity by Transport Mode")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Try a custom trip distance")
    distance_km = st.slider("Distance (km)", min_value=1, max_value=3000, value=10, step=1)
    df = emissions_table(distance_km, modes)
    st.dataframe(df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(df["Mode"], df[f"CO₂ for {distance_km} km (kg)"])
    ax2.set_ylabel(f"CO₂ for {distance_km} km (kg)")
    ax2.set_title(f"Total CO₂ for {distance_km} km by Mode")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig2)

    st.info("Note: Actual emissions vary by vehicle efficiency, occupancy, electricity mix, and route conditions. These are indicative averages for learning.")

# -----------------------------
# PAGE 4: Global Impact (Mini dataset for illustration)
# -----------------------------
elif page == "🌡️ Global Impact":
    st.title("🌡️ Global Climate Trends")

    st.write("""
Below are **illustrative mini-datasets** to help visualize three key trends:
**(1)** Rising atmospheric CO₂ concentration **(2)** Rising global temperature anomaly and **(3)** Top 5 Emitters of the World.
 For research, use official datasets
(e.g., NASA GISTEMP, NOAA, or Our World in Data).
""")

    # Mini (illustrative) CO2 ppm dataset (not official numbers; roughly indicative trend)
    years = list(range(1960, 2025, 5))
    co2_ppm = [317, 325, 334, 344, 354, 367, 380, 395, 409, 421, 419, 424, 419]  # rough trend-like values

    df_co2 = pd.DataFrame({"Year": years, "CO₂ concentration (ppm)": co2_ppm})
    st.subheader("Atmospheric CO₂ concentration over time (ppm)")
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(df_co2["Year"], df_co2["CO₂ concentration (ppm)"], marker="o")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("ppm")
    ax1.set_title("Rising Atmospheric CO₂ (illustrative)")
    st.pyplot(fig1)
    st.caption("Illustrative mini-dataset; real data shows a steady increase from ~316 ppm in 1960 to 420+ ppm recently.")

    # Mini (illustrative) temperature anomaly dataset (°C)
    years_t = list(range(1960, 2025, 5))
    temp_anom = [-0.05, -0.03, -0.01, 0.02, 0.08, 0.19, 0.31, 0.43, 0.54, 0.75, 0.90, 1.00, 0.98]  # rough trend-like values

    df_temp = pd.DataFrame({"Year": years_t, "Global temperature anomaly (°C)": temp_anom})
    st.subheader("Global temperature anomaly (°C) relative to mid-20th century")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(df_temp["Year"], df_temp["Global temperature anomaly (°C)"], marker="o")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("°C")
    ax2.set_title("Warming Global Temperatures (illustrative)")
    st.pyplot(fig2)
    st.caption("Illustrative mini-dataset; real datasets show ~1.1–1.3°C warming since pre-industrial levels.")
# ✅ NEW TOP 5 EMITTERS CHART
    st.subheader("🌍 Top 5 CO₂ Emitters (Illustrative)")
    emitters = {"China": 10.5, "USA": 5.3, "India": 2.7, "EU": 2.5, "Russia": 1.5}  # in Gt CO₂
    df_emit = pd.DataFrame(list(emitters.items()), columns=["Country", "Annual Emissions (Gt CO₂)"])
    fig4, ax4 = plt.subplots()
    ax4.bar(df_emit["Country"], df_emit["Annual Emissions (Gt CO₂)"], color="purple")
    ax4.set_ylabel("Gt CO₂ per year")
    ax4.set_title("Top 5 Global Emitters (approx)")
    st.pyplot(fig4)

    st.markdown("### What this means")
    st.markdown("""
- Higher CO₂ → more trapped heat → **warmer planet**.
- More heat → **stronger, wetter, or drier extremes**, depending on the region.
- **Sea levels rise** due to thermal expansion and melting land ice.
- **Ecosystems and agriculture** are stressed; adaptation and mitigation are both needed.
    """)

    st.markdown("### What helps")
    st.markdown("""
- Rapidly **reduce fossil fuel use**; electrify transport & industry.
- Expand **renewable energy**; improve energy efficiency.
- Protect and restore **forests and wetlands** (carbon sinks).
- Encourage **low-carbon transport** — exactly what this app nudges users to do.
    """)


