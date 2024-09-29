import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load input and output data
file_path = r'Rogowski_data\Rogowski_data\20C\20C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

# Constants
tr_50 = 100 / 1000 / 1000  # Transducer ratio
f = 50  # Frequency in Hz
w_50 = 2 * np.pi * f  # Angular frequency
T_0 = 20  # Reference temperature in °C
T = 20  # Current temperature in °C
M_0 = tr_50 / w_50  # Nominal mutual inductance
time = 2000  # Number of time points
dt = 1 / f  # Time step based on frequency

# Load data
I_p = df.iloc[0, :time].values  # Input current
real_output = df.iloc[1, :time].values  # Real output voltage

# Normalize the input and output data
I_p_min, I_p_max = I_p.min(), I_p.max()
real_output_min, real_output_max = real_output.min(), real_output.max()

I_p_normalized = (I_p - I_p_min) / (I_p_max - I_p_min)  # Min-Max Scaling
real_output_normalized = (real_output - real_output_min) / (real_output_max - real_output_min)  # Min-Max Scaling

# Define the PINN model with alpha as a learnable parameter
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)   # Increased the number of neurons in the first layer
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)     # Output layer
        self.x = nn.Parameter(torch.tensor(-0.01, device=device))  # x as a learnable parameter
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))  # Alpha as a learnable parameter

    def forward(self, I_p):
        x = torch.tanh(self.fc1(I_p))  # Activation function
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = PINN().to(device)  # Move the model to the CUDA device

# Prepare data for PyTorch
I_p_tensor = torch.tensor(I_p_normalized, dtype=torch.float32, device=device).view(-1, 1)
real_output_tensor = torch.tensor(real_output_normalized, dtype=torch.float32, device=device).view(-1, 1)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning rate scheduler

# Training loop with 100,000 epochs
num_epochs = 100000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    
    # Forward pass: predict output using the model
    predicted_output = model(I_p_tensor)
    
    # Calculate mutual inductance based on temperature and current value of x
    M = torch.tensor(M_0 * (1 + model.alpha * (T - T_0)), dtype=torch.float32, device=device)  # Use learnable alpha
    
    # Calculate the time derivative of the input current
    dI_p_dt = torch.diff(I_p_tensor.squeeze(), dim=0) / dt
    dI_p_dt = torch.cat((dI_p_dt, torch.tensor([0.0], dtype=torch.float32, device=device)))  # Append 0 for dimension match
    
    # Ensure that model.x has the right shape for multiplication
    x_value = model.x.view(1)  # Reshape x to ensure it's a single-element tensor
    
    # Output voltage using the physical equation
    u_s = -M * dI_p_dt.view(-1, 1) * x_value  # Reshape dI_p_dt for compatibility

    # Calculate residual loss (enforces the physical relationship)
    residual_loss = nn.MSELoss()(u_s, real_output_tensor)

    # Calculate data loss (MSE between predicted and real output)
    data_loss = nn.MSELoss()(predicted_output, real_output_tensor)

    # Total loss (combine residual and data losses)
    total_loss = data_loss + residual_loss
    
    # Backward pass: compute gradients
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    scheduler.step()  # Update learning rate
    
    if epoch % 10000 == 0:  # Log every 10,000 epochs for visibility
        print(f'Epoch [{epoch}/{num_epochs}], Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Residual Loss: {residual_loss.item():.4f}')

# After training, let's visualize the results
model.eval()
with torch.no_grad():
    predicted_output_normalized = model(I_p_tensor).cpu().numpy()  # Move output back to CPU for plotting

# Inverse normalization of predicted output
predicted_output = predicted_output_normalized * (real_output_max - real_output_min) + real_output_min

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 1, time), real_output, label='Real Output Voltage', color='green')
plt.plot(np.linspace(0, 1, time), predicted_output, label='Predicted Output Voltage', color='orange')
plt.title('Comparison of Real and Predicted Output Voltages')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()
plt.show()

# Print the estimated value of x and alpha
print(f'Estimated value of x: {model.x.item():.4f}')
print(f'Estimated value of alpha: {model.alpha.item():.4f}')
