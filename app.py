import matplotlib
matplotlib.use('Agg')  # <-- Add this before importing pyplot
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# Initialize Flask app
app = Flask(__name__)

# Load and clean dataset once
df = pd.read_csv("co-emissions-per-capita new.csv")
df['Annual CO₂ emissions (per capita)'] = pd.to_numeric(df['Annual CO₂ emissions (per capita)'], errors='coerce')
df.dropna(inplace=True)
countries = sorted(df['Entity'].unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure static directory exists
    os.makedirs("static", exist_ok=True)

    # Default country selection
    selected_country = "India"
    if request.method == 'POST':
        selected_country = request.form['country']

    # Filter dataset by selected country
    country_df = df[df['Entity'] == selected_country]
    X = country_df[['Year']]
    y = country_df['Annual CO₂ emissions (per capita)']

    # Train regression model
    model = LinearRegression().fit(X, y)

    # Predict future emissions from 2025 to 2035
    future_years = pd.DataFrame({'Year': list(range(2025, 2036))})
    future_preds = model.predict(future_years)
    predictions = list(zip(future_years['Year'], future_preds))

    # Plot and save the emission trend graph
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label="Historical", color="blue")
    plt.plot(future_years, future_preds, label="Prediction (2025–2035)", color="red", linestyle='--')
    plt.title(f"{selected_country} - CO₂ Emission per Capita")
    plt.xlabel("Year")
    plt.ylabel("Emissions per Capita")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Render template with all data
    return render_template("index.html",
                           countries=countries,
                           selected=selected_country,
                           plot_path=plot_path,
                           predictions=predictions)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
