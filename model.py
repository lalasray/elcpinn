import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = r'Rogowski_data\Rogowski_data\55C\55C_50Hz_50A.csv'
df = pd.read_csv(file_path, sep=';', header=None)

tr_50 = 100 / 1000 / 1000 
f = 50  
w_50 = 2 * np.pi * f  
T_0 = 55  
T = 20  
M_0 = tr_50 / w_50  
time = 10000  
dt = 1 / f  

I_p = df.iloc[0, :time].values  
real_output = df.iloc[1, :time].values  


I_p_min, I_p_max = I_p.min(), I_p.max()
real_output_min, real_output_max = real_output.min(), real_output.max()

I_p_normalized = (I_p - I_p_min) / (I_p_max - I_p_min)  
real_output_normalized = (real_output - real_output_min) / (real_output_max - real_output_min)  # Min-Max Scaling

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)   
        self.fc2 = nn.Linear(100, 400)
        self.fc3 = nn.Linear(400, 1)     
        self.x = nn.Parameter(torch.tensor(-0.01, device=device))  
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))  
    
    def forward(self, I_p):
        x = torch.tanh(self.fc1(I_p))  
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = PINN().to(device)  
I_p_tensor = torch.tensor(I_p_normalized, dtype=torch.float32, device=device).view(-1, 1)
real_output_tensor = torch.tensor(real_output_normalized, dtype=torch.float32, device=device).view(-1, 1)

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  

num_epochs = 100000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    
    predicted_output = model(I_p_tensor)
    M = torch.tensor(M_0 * (1 + model.alpha * (T - T_0)), dtype=torch.float32, device=device)  # Use learnable alpha
    
    dI_p_dt = torch.diff(I_p_tensor.squeeze(), dim=0) / dt
    dI_p_dt = torch.cat((dI_p_dt, torch.tensor([0.0], dtype=torch.float32, device=device)))  # Append 0 for dimension match
    
    x_value = model.x.view(1)  
    u_s = -M * dI_p_dt.view(-1, 1) * x_value  
    residual_loss = nn.MSELoss()(u_s, real_output_tensor)
    data_loss = nn.MSELoss()(predicted_output, real_output_tensor)
    total_loss = data_loss + residual_loss
    total_loss.backward()
    optimizer.step()
    scheduler.step()  
    if epoch % 10000 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Total Loss: {total_loss.item():.4f}, Data Loss: {data_loss.item():.4f}, Residual Loss: {residual_loss.item():.4f}')

model.eval()
with torch.no_grad():
    predicted_output_normalized = model(I_p_tensor).cpu().numpy()  
predicted_output = predicted_output_normalized * (real_output_max - real_output_min) + real_output_min

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)  
plt.plot(np.linspace(0, 1, time), real_output, label='Real Output Voltage', color='green')
plt.title('Real Output Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, 1, time), predicted_output, label='Predicted Output Voltage', color='orange')
plt.title('Predicted Output Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print(f'Estimated value of x: {model.x.item():.4f}')
print(f'Estimated value of alpha: {model.alpha.item():.4f}')
