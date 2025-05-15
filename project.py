# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:01:35 2025

@author: colon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import expon

# Load the ILI dataset 
df = pd.read_csv("ilidata.csv")

# Streamlit title
st.title("Influenza-Like Illness (ILI) Over Time")

# Create a "weeks" column that counts from 0 to max number of rows (n-1)
df = df.sort_values(by=["state", "epiweek"])
df["weeks"] = df.groupby("state").cumcount()

# Select state using dropdown
states = df["state"].unique()
selected_state = st.selectbox("Select a state", states)

# Filter dataframe for selected state
state_data = df[df["state"] == selected_state]
ili_values = state_data["ili"].dropna()

# Plot ILI percentage (column 'ili') over time
st.line_chart(
    data=state_data.set_index("weeks")[["ili"]],
    x_label="Weeks Since Start",
    y_label="ILI (%)"
)

# Estimate lambda using LLN:
y_bar = ili_values.mean()
lambda_hat = 1 / y_bar

# Create histogram with overlayed exponential PDF
fig, ax = plt.subplots()
count, bins, ignored = ax.hist(
    ili_values, bins=30, density=True, alpha=0.6, color='blue', label='ILI Histogram')

# Overlay exponential PDF
x_vals = np.linspace(0, ili_values.max(), 500)

# scale = 1/λ = mean
pdf_vals = expon.pdf(x_vals, scale=1/lambda_hat)  
ax.plot(x_vals, pdf_vals, 'r-', lw=2, label='Exponential Fit (λ̂ = {:.2f})'.format(lambda_hat))

# Labels
ax.set_xlim(left=0) 
ax.set_title(f"Influenza-Like Illness (ILI) Percent Distribution for {selected_state}")
ax.set_xlabel("ILI (%)")
ax.set_ylabel("Density")
ax.legend()

st.header("Plot Descriptions")

# Description for the time series plot
st.subheader("1. ILI Percentage Over Time")

st.markdown("""
The line plot above shows how the percentage of outpatient visits related to Influenza-Like Illness (ILI) changes over time for the selected state.
Each point represents a weekly observation, starting from week 0 (the beginning of the dataset).
The y-axis shows what percentage of visits were due to ILI that week.

This longtitudinal visualization helps identify patterns in flu-like illness, such as seasonal spikes, long-term trends, or abnormal activity.
Consistent with public health patterns, many states display clear seasonal cycles, with peaks in colder months.
This visual representation aligns with the Law of Large Numbers (LLN) which tells us that the more data we collect, the closer our average gets to the true value.
""")

# Description for the histogram + exponential distribution
st.subheader("2. Histogram of ILI (%) and Exponential Fit")

st.markdown("""
The histogram shows how frequently different ILI percentages occurred. The blue bars represent how often we saw low, medium, or high flu activity across all weeks.

Overlaid on the histogram is a red curve representing an exponential distribution fit.
 To build this curve, we estimated the rate parameter λ using the Law of Large Numbers, where we took the average ILI percentage and used λ̂ = 1/ȳ (where ȳ is the ILI mean).
This density curve lets us compare how well the exponential model approximates the observed ILI data.

This comparison allows us to understand how real-world epidemiological data may follow (or deviate from) statistical theory and its distributions.
If the red curve matches the shape of the histogram, it suggests that the exponential model is a good fit for this data.
""")

st.pyplot(fig)

