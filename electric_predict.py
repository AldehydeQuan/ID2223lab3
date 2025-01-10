import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import json

# Load electricity consumption data
df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False)

# Convert 'Date' and 'Time' columns to datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Set Datetime as index
df.set_index('Datetime', inplace=True)

# Convert relevant columns to numeric
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Fill missing values
df = df.ffill()

# Resample daily and calculate mean
daily_data = df['Global_active_power'].resample('D').mean().reset_index()
daily_data.rename(columns={'Datetime': 'Date'}, inplace=True)

# Load weather data
with open('weather_data_paris.json', 'r') as f:
    weather_data = json.load(f)

# Extract relevant features from weather data
weather_records = []
for record in weather_data:
    weather_records.append({
        'Date': pd.to_datetime(record['date']),
        'avg_temp': float(record['avgtempC']),
        'precipitation': float(record['hourly'][0]['precipMM']),
        'humidity': float(record['hourly'][0]['humidity'])
    })
weather_df = pd.DataFrame(weather_records)

# Merge electricity and weather data
merged_data = pd.merge(daily_data, weather_df, on='Date', how='left')

# Drop rows with missing values
merged_data.dropna(inplace=True)

# Add time-related features
merged_data['day'] = merged_data['Date'].dt.day
merged_data['month'] = merged_data['Date'].dt.month

# Split into train and test data
train_data = merged_data[:int(0.8 * len(merged_data))]
test_data = merged_data[int(0.8 * len(merged_data)):]  

# Define dataset class
class PowerDataset(TorchDataset):
    def __init__(self, data, look_back=30):
        self.data = data
        self.look_back = look_back

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.look_back][['day', 'month', 'avg_temp', 'precipitation', 'humidity']].values
        y = self.data.iloc[idx + self.look_back]['Global_active_power']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

train_dataset = PowerDataset(train_data)
test_dataset = PowerDataset(test_data)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM model
class PowerModel(nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

model = PowerModel()

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)  # 模型输出
        loss = criterion(output, y.squeeze(-1))  # 确保目标形状匹配
        if torch.isnan(loss):
            raise ValueError("Loss is NaN during training.")
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "electricity_forecast_model.pt")
