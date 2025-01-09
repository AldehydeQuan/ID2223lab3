from datasets import Dataset
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig

# load dataset and pre-procession
df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False)

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)
df.replace('?', np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df['Global_active_power'] = df['Global_active_power'].astype(float)

# aggregate in hours
hourly_data = df['Global_active_power'].resample('H').mean().reset_index()

# add time tag
hourly_data['hour'] = hourly_data['Datetime'].dt.hour
hourly_data['day'] = hourly_data['Datetime'].dt.day
hourly_data['month'] = hourly_data['Datetime'].dt.month

# devide data into train data and test data
train_data = hourly_data[:int(0.8 * len(hourly_data))]
test_data = hourly_data[int(0.8 * len(hourly_data)):]

# define dataset
class PowerDataset(TorchDataset):
    def __init__(self, data, look_back=24):
        self.data = data
        self.look_back = look_back

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.look_back][['hour', 'day', 'month', 'Global_active_power']].values
        y = self.data.iloc[idx + self.look_back]['Global_active_power']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = PowerDataset(train_data)
test_dataset = PowerDataset(test_data)

# loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# LSTM model
class PowerModel(nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = PowerModel()

# train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

# save model
torch.save(model.state_dict(), "electricity_forecast_model.pt")

from huggingface_hub import HfApi, HfFolder, Repository

# set directory
model_dir = "./electricity_forecast_model"  
repo_name = "AldehydeQuan/lab3"  

import os
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model.state_dict(), f"{model_dir}/electricity_forecast_model.pt")

config = {
    "model_type": "lstm",
    "input_size": 4,
    "hidden_size": 50,
    "num_layers": 2,
    "framework": "pytorch",
}
with open(f"{model_dir}/config.json", "w") as f:
    import json
    json.dump(config, f)

try:
    with open("huggingface_api_token.txt", "r") as token_file:
        api_token = token_file.read().strip()
        HfFolder.save_token(api_token)
        print("Successfully read and saved API Token from huggingface_api_token.txt.")
except FileNotFoundError:
    print("Error: huggingface_api_token.txt file not found. Please create the file and add your API Token.")
except Exception as e:
    print(f"An error occurred while reading the token: {e}")

repo = Repository(local_dir=model_dir, clone_from=repo_name)
repo.push_to_hub(commit_message="Upload LSTM model for electricity forecasting")