import gradio as gr
import torch
import pandas as pd
import numpy as np

# define model
class PowerModel(torch.nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(50, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# load model
model = PowerModel()
model.load_state_dict(torch.load("electricity_forecast_model.pt"))
model.eval()

# define prediction function
def predict(hour, day, month):
    input_data = torch.tensor([[[hour, day, month, 0]]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_data).item()
    return f"Predicted Power Consumption: {prediction:.2f} kW"

# create gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Hour (0-23)"),
        gr.Number(label="Day (1-31)"),
        gr.Number(label="Month (1-12)")
    ],
    outputs="text",
    title="Electricity Forecast",
    description="Input the hour, day, and month to predict electricity consumption."
)

# launch
if __name__ == "__main__":
    interface.launch()