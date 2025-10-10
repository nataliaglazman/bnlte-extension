import pandas as pd
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



class NeuralODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)
    

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('data.csv')  # Replace with your dataset path
    features = data.drop(columns=['target']).values  # Replace 'target' with your target column
    target = data['target'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Define model
    input_dim = X_train.shape[1]
    hidden_dim = 50
    func = NeuralODEFunc(input_dim, hidden_dim)

    # Training settings
    optimizer = torch.optim.Adam(func.parameters(), lr=0.001)
    num_epochs = 1000
    t = torch.linspace(0., 1., steps=2)  # Time points for ODE solver

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_y = odeint(func, X_train_tensor, t)[-1]
        loss = nn.MSELoss()(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    with torch.no_grad():
        pred_y_test = odeint(func, X_test_tensor, t)[-1]
        test_loss = mean_squared_error(y_test_tensor.numpy(), pred_y_test.numpy())
        print(f'Test MSE: {test_loss:.4f}')