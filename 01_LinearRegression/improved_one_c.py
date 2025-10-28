import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ==================== 1. 数据加载 ====================
def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()
    return timestamp, output_voltage, output_current, thermistor_temperatures


# ==================== 2. 传感器位置 ====================
def calculate_positions():
    bottom_distance = 0.003
    spacing = 0.005
    return torch.tensor([bottom_distance + i * spacing for i in range(8)], dtype=torch.float32)


# ==================== 3. 去趋势函数 ====================
def detrend_temperature_data(timestamps, temperatures):
    detrended = np.zeros_like(temperatures)
    for i in range(temperatures.shape[1]):
        coeffs = np.polyfit(timestamps, temperatures[:, i], 1)
        b_i, a_i = coeffs
        detrended[:, i] = temperatures[:, i] - (a_i + b_i * timestamps)
    return detrended


# ==================== 4. PyTorch 模型 ====================
class HeatDiffusionModel(torch.nn.Module):
    def __init__(self, omega, positions, L=0.041):
        super().__init__()
        self.omega = torch.tensor(omega, dtype=torch.float32)
        self.positions = positions
        self.L = L

        # 可学习参数
        self.logD = torch.nn.Parameter(torch.tensor(np.log(3e-5)))  # logD to ensure positivity
        self.C_minus = torch.nn.Parameter(torch.tensor(1e-5))

    def forward(self, t):
        """
        t: 时间 (Tensor, shape [N])
        输出每个位置的预测温度波动 (shape [len(positions), N])
        """
        D = torch.exp(self.logD)  # 保证 D > 0
        k = torch.sqrt(self.omega / (2 * D))
        beta = torch.sqrt(self.omega / (2 * D)) * (1j - 1)
        C_plus = self.C_minus * torch.exp(2 * self.L * k * (1 - 1j))

        temps = []
        for x in self.positions:
            term = C_plus * torch.exp(beta * x) + self.C_minus * torch.exp(-beta * x)
            fluctuating = torch.real(term * torch.exp(-1j * self.omega * t))
            temps.append(fluctuating)
        return torch.stack(temps)  # shape: [num_positions, N]


# ==================== 5. 训练函数 ====================
def train_model(model, timestamps, measured_data, lr=1e-2, epochs=4000, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = torch.ones((8, 1))
    #weights = torch.tensor([1.0, 0.7, 0.7, 0.5, 0.5, 0.3, 0.2, 0.1])[:, None]
    #loss_fn = torch.mean(weights[:, None] * (pred - measured_data)**2)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(timestamps)
        loss = torch.mean(weights * (pred - measured_data)**2)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and epoch % 200 == 0:
            D_val = torch.exp(model.logD).item()
            print(f"Epoch {epoch:5d}: Loss={loss.item():.6e}, D={D_val:.4e}, C-={model.C_minus.item():.4f}")

    return loss_history


# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    period = 15 # just an example
    omega = 2 * np.pi / period

    timestamp, _, _, thermistor_temperatures = load_dataset(f"/Users/frank/Desktop/D2L_Projects/data/brass data/brass {period}s.csv")
    detrended = detrend_temperature_data(timestamp, thermistor_temperatures)

    t = torch.tensor(timestamp, dtype=torch.float32)
    positions = calculate_positions()
    T_measured = torch.tensor(detrended.T, dtype=torch.float32)  # shape [8, N]

    model = HeatDiffusionModel(omega, positions)
    loss_history = train_model(model, t, T_measured, lr=1e-2, epochs=3000)

    # ===== 训练完成后结果 =====
    D_fit = torch.exp(model.logD).item()
    C_minus_fit = model.C_minus.item()

    # 计算 C_plus
    with torch.no_grad():
        D_torch = torch.exp(model.logD)
        k_fit = torch.sqrt(model.omega / (2 * D_torch))
        C_plus_fit = model.C_minus * torch.exp(2 * model.L * k_fit * (1 - 1j))

    print("\n===== Training Complete =====")
    print(f"period          = {period}s")
    print(f"Fitted D        = {D_fit:.8e} m²/s")
    print(f"Fitted C_minus  = {C_minus_fit:.6f}")
    print(f"Fitted C_plus   = {C_plus_fit.real.item():.6f} + {C_plus_fit.imag.item():.6f}j")
    

    # ===== 绘制 loss 曲线 =====
    plt.figure(figsize=(7, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Training Loss Curve (Global Fit via Gradient Descent)")
    plt.grid(True)
    plt.show()

    # ===== 绘制拟合曲线 =====
    with torch.no_grad():
        pred = model(t).numpy()

    plt.figure(figsize=(10, 6))
    for i in range(len(positions)):
        plt.plot(t, T_measured[i], 'b.', alpha=0.4)
        plt.plot(t, pred[i], 'r-', alpha=0.8, label=f'x={positions[i].item():.3f} m' if i == 0 else None)
    plt.xlabel("Time (s)")
    plt.ylabel("Detrended Temperature (°C)")
    plt.title(f"Global Fit Result (Period={period}s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
