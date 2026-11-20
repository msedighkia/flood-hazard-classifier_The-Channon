# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:40:27 2026

@author: Windows
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Flood Hazard Classification Tool",
    page_icon="🌧️",
    layout="centered",
)

# ============================================================
# 1. Load the trained classification model (Cached for speed)
# ============================================================
MODEL_FILE = "Best_Flood_Hazard_Classification_ModelComp.joblib"


@st.cache_resource
def load_model():
  return joblib.load(MODEL_FILE)


try:
  saved_data = load_model()
  classification_model = saved_data["model"]
  encoder = saved_data["encoder"]
  model_name = saved_data.get("model_name", "Extra Trees Classifier")
  input_names = saved_data.get("input_names", [])
  class_numbers = saved_data.get("class_numbers", list(range(1, 8)))
except Exception as e:
  st.error(
      f"Could not load `{MODEL_FILE}`. Make sure it is uploaded to your GitHub"
      f" repository alongside `app.py`. Error: {e}"
  )
  st.stop()

# ============================================================
# 2. Define allowable input ranges & class descriptions
# ============================================================
MIN_TOTAL_RAINFALL = 115.0
MAX_TOTAL_RAINFALL = 4600.0

MIN_DURATION = 6.0
MAX_DURATION = 408.0

MIN_ANTECEDENT_RAINFALL = 0.0
MAX_ANTECEDENT_RAINFALL = 445.0

CLASS_DESCRIPTIONS = {
    "Class 1": (
        "Minor impacts: Low-lying roads and bridges, including "
        "Tuntable Creek Road Bridge near The Channon, are likely "
        "to be waterlogged. However, very little or no water "
        "accumulation or surface runoff is expected around "
        "The Channon."
    ),
    "Class 2": (
        "Minor impacts with minor local safety risks: Roads and bridges, "
        "including Tuntable Creek Road Bridge, are likely to be "
        "waterlogged."
        "Minor water accumulation may occur around "
        "The Channon Tavern, but significant impacts on buildings "
        "and surrounding properties are not expected."
    ),
    "Class 3": (
        "Insignificant localised impacts: Flooding of roads and bridges "
        "is likely to create safety risks for vehicles and "
        "pedestrians. Limited water accumulation "
        "are expected around The Channon Tavern, making walking through "
        "garden areas difficult."
    ),
    "Class 4": (
        "Significant localised impacts: All impacts associated with previous "
        "classes (1 to 3) are expected. Only areas close to the river, including the"
        " "
        "Tennis Club, are likely to be inundated, with flood depths of "
        "less than 1 m. Flooding may also affect road intersections "
        "further from the river, including the Dunoon Street–Tuntable "
        "Road intersection."
    ),
    "Class 5": (
        "Significant widespread impacts: All previous impacts in classes 1 to 4 are expected,"
        " with flood "
        "depths increasing to approximately 1–2 m in areas close to the "
        "river, such as the Tennis Club. Significant flooding is also "
        "expected at bridges around The Channon, making them unsafe "
        "and potentially impassable."
    ),
    "Class 6": (
        "Severe widespread impacts: Impacts associated with Class 5 are "
        "expected to intensify. Flood depths in areas close to the river, "
        "including the Tennis Club, are likely to exceed 2 m. Flooding "
        "is also expected to extend further inland, reaching locations "
        "such as The Channon Tavern, where depths may be less than 1 m."
    ),
    "Class 7": (
        "Extreme widespread impacts: All impacts associated with Class 6 are "
        "expected to become more severe. Flooding around The Channon "
        "Tavern is likely to increase substantially, potentially reaching "
        "depths of approximately 1–2 m, with significant inundation "
        "of surrounding properties and infrastructure."
    ),
}


# ============================================================
# 3. Calculation & Prediction Logic
# ============================================================
def calculate_rainfall_characteristics(total_rainfall, duration):
  average_intensity = total_rainfall / duration
  maximum_intensity = (
      0.0306 * average_intensity**2 + 1.2287 * average_intensity + 8.5542
  )
  return average_intensity, maximum_intensity


def predict_flood_hazard(total_rainfall, duration, antecedent_rainfall):
  average_intensity, maximum_intensity = calculate_rainfall_characteristics(
      total_rainfall, duration
  )

  # Input order: Total rainfall, Maximum intensity, Average intensity, Antecedent rainfall, Event duration
  X = np.array(
      [
          total_rainfall,
          maximum_intensity,
          average_intensity,
          antecedent_rainfall,
          duration,
      ],
      dtype=float,
  ).reshape(1, -1)

  predicted_encoded = classification_model.predict(X)
  predicted_class_number = encoder.inverse_transform(
      predicted_encoded.astype(int)
  )[0]
  predicted_class = f"Class {int(predicted_class_number)}"

  probabilities = None
  if hasattr(classification_model, "predict_proba"):
    probabilities = classification_model.predict_proba(X)[0]

  return (
      average_intensity,
      maximum_intensity,
      predicted_class,
      int(predicted_class_number),
      probabilities,
  )


# ============================================================
# 4. Web User Interface
# ============================================================
st.title("🌧️ Rainfall-Based Flood Hazard Classification")
st.markdown(
    "Enter the rainfall event parameters below to calculate rainfall"
    " intensities and predict the flood hazard classification."
)

# Optional: Display a site/reference image if you upload one (e.g., "map.png")
# if os.path.exists("map.png"):
#     st.image("map.png", use_container_width=True)

with st.form("flood_prediction_form"):
  col1, col2, col3 = st.columns(3)

  with col1:
    total_rainfall = st.number_input(
        "Total Rainfall (mm)",
        min_value=MIN_TOTAL_RAINFALL,
        max_value=MAX_TOTAL_RAINFALL,
        value=250.0,
        step=5.0,
        help="Allowable range: 115 – 4600 mm",
    )

  with col2:
    duration = st.number_input(
        "Event Duration (hours)",
        min_value=MIN_DURATION,
        max_value=MAX_DURATION,
        value=24.0,
        step=1.0,
        help="Allowable range: 6 – 408 hours",
    )

  with col3:
    antecedent_rainfall = st.number_input(
        "Prior 2-Week Rainfall (mm)",
        min_value=MIN_ANTECEDENT_RAINFALL,
        max_value=MAX_ANTECEDENT_RAINFALL,
        value=50.0,
        step=5.0,
        help="Antecedent rainfall over prior two weeks (0 – 445 mm)",
    )

  submitted = st.form_submit_button(
      "Classify Flood Hazard", type="primary", use_container_width=True
  )

if submitted:
  avg_int, max_int, pred_class, pred_num, probs = predict_flood_hazard(
      total_rainfall, duration, antecedent_rainfall
  )

  st.divider()
  st.subheader("Calculated Rainfall Intensities")
  m1, m2 = st.columns(2)
  m1.metric("Average Rainfall Intensity", f"{avg_int:.2f} mm/h")
  m2.metric("Estimated Maximum Intensity", f"{max_int:.2f} mm/h")

  st.subheader("Predicted Flood Hazard Classification")

  # Colour-code the alert box based on severity
  description = CLASS_DESCRIPTIONS.get(
      pred_class, "No description available."
  )
  if pred_num <= 2:
    st.info(f"### {pred_class}\n{description}")
  elif pred_num <= 4:
    st.warning(f"### {pred_class}\n{description}")
  else:
    st.error(f"### {pred_class}\n{description}")

  # Show class probabilities if supported by the model
  if probs is not None:
    with st.expander("View Model Class Probabilities"):
      decoded_classes = [
          f"Class {int(c)}"
          for c in encoder.inverse_transform(np.arange(len(probs)))
      ]
      prob_df = pd.DataFrame(
          {"Flood Hazard Class": decoded_classes, "Probability (%)": probs * 100}
      ).set_index("Flood Hazard Class")

      st.bar_chart(prob_df)
      st.dataframe(prob_df.style.format("{:.1f}%"), use_container_width=True)
