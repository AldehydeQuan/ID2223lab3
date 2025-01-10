# app.py
import gradio as gr
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import PreTrainedModel, PretrainedConfig

# Define model
class PowerModel(torch.nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# Load pre-trained model
model = PowerModel()
model.load_state_dict(torch.load("electricity_forecast_model.pt"))
model.eval()

# Define prediction function
def predict(year, month, day, avg_temp, precipitation, humidity):
    # Validate inputs
    for param, name in zip([year, month, day, avg_temp, precipitation, humidity],
                           ["year", "month", "day", "avg_temp", "precipitation", "humidity"]):
        if not isinstance(param, (int, float)) or np.isnan(param):
            return f"Invalid input for {name}: {param}"

    try:
        # Prepare input data
        input_data = torch.tensor([[[day, month, avg_temp, precipitation, humidity]]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_data).item()
        return f"Predicted Power Consumption: {prediction:.2f} kW"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Year"),
        gr.Number(label="Month (1-12)"),
        gr.Number(label="Day (1-31)"),
        gr.Number(label="Average Temperature (\u00b0C)"),
        gr.Number(label="Precipitation (mm)"),
        gr.Number(label="Humidity (%)")
    ],
    outputs="text",
    title="Electricity Forecasting",
    description="Input date and weather-related features to predict daily electricity consumption."
)

# Launch app
if __name__ == "__main__":
    interface.launch()
