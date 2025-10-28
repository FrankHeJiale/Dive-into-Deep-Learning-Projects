import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from scipy.optimize import curve_fit

def sin_plus_line(t, A, omega, phi, B, C):
    return A * np.sin(omega * t + phi) + B * t + C

def initial_guess(t, y, omega_guess):
    A_guess = (np.max(y) - np.min(y)) / 2
    phi_guess = 0
    B_guess = 0
    C_guess = np.mean(y)
    return [A_guess, omega_guess, phi_guess, B_guess, C_guess]

# Disabled: Individual sensor fit plots
def plot_fit(period):
    pass  # Skip individual fit plots

def error_percentages(i):
    y = thermistor_temperatures[:, i]
    omega_0 = omega_guesses[i]
    popt, pcov = curve_fit(
        sin_plus_line,
        timestamp,
        y,
        p0=initial_guess(timestamp, y, omega_0)
    )
    A, omega, phi, B, C = popt
    perr = np.sqrt(np.diag(pcov))
    return (perr/popt)

def sin_fit_val(i):
    y = thermistor_temperatures[:, i]
    omega_0 = omega_guesses[i]
    popt, pcov = curve_fit(
        sin_plus_line,
        timestamp,
        y,
        p0=initial_guess(timestamp, y, omega_0)
    )
    A, omega, phi, B, C = popt
    if A < 0:
        A = -A
        phi = phi + np.pi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return np.array([A, omega, phi, B, C])

def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()
    with open(path, 'r') as f:
        content = f.read()
    comments = re.search(r"Comments: (.*)$", content, re.MULTILINE)[1] if re.search(r"Comments: (.*)$", content, re.MULTILINE) else ""
    return timestamp, output_voltage, output_current, thermistor_temperatures, comments

def calculate_positions():
    L = 0.041  
    bottom_distance = 0.003  
    spacing = 0.005  
    positions = []
    for i in range(8):
        pos = bottom_distance + i * spacing
        positions.append(pos)
    return np.array(positions)

def error_from_two(a, b, quantity):
    a_errors_percentages = error_percentages(a)
    b_errors_percentages = error_percentages(b)
    result = np.sqrt((a_errors_percentages[quantity])**2 + (b_errors_percentages[quantity])** 2)
    return result

def DA_DPhi(a, b):
    A_a, omega_a, phi_a, _, _ = sin_fit_val(a)
    A_b, omega_b, phi_b, _, _ = sin_fit_val(b)
    omega = np.mean([omega_a, omega_b])
    delta_d = 0.005 * np.abs(a - b)
    A_ratio = np.abs(A_b) / np.abs(A_a)
    delta_phi = np.abs(phi_b - phi_a)
    DA = omega * (delta_d)**2 / (2 * (np.log(A_ratio))** 2)
    Dphi = omega * (delta_d)**2 / (2 * (delta_phi)** 2)
    error_in_A_ratio = error_from_two(a, b, 0)      
    error_in_delta_phi = error_from_two(a, b, 2)    
    r_A = error_in_A_ratio / 100
    r_phi = error_in_delta_phi / 100
    DA_rel_error = 2 * r_A / np.abs(np.log(A_ratio))
    Dphi_rel_error = 2 * r_phi / np.abs(delta_phi)
    DA_error = DA * DA_rel_error
    Dphi_error = Dphi * Dphi_rel_error
    DA_error_percent = DA_rel_error * 100
    Dphi_error_percent = Dphi_rel_error * 100
    return {
        "DA": DA,
        "DA_error": DA_error,
        "DA_error_%": DA_error_percent,
        "Dphi": Dphi,
        "Dphi_error": Dphi_error,
        "Dphi_error_%": Dphi_error_percent
    }

# Modified: Plot D values + weighted average lines (no error bars on data)
def plot_D(new_model_results, period):
    i_values = []
    DA_values = []
    DA_errors = []  # Collect DA errors for weighting
    Dphi_values = []
    Dphi_errors = []  # Collect Dphi errors for weighting
    
    # Get DA, Dphi values and their errors
    for i in range(7):
        result = DA_DPhi(0, i + 1)
        i_values.append(i + 1)
        DA_values.append(result["DA"])
        DA_errors.append(result["DA_error"])
        Dphi_values.append(result["Dphi"])
        Dphi_errors.append(result["Dphi_error"])
    
    # Get new model D values and their errors
    new_D_values = []
    new_D_errors = []
    new_i_values = []
    for res in new_model_results:
        if 'error' not in res:
            new_i_values.append(res['column'])
            new_D_values.append(res['D'][0])
            new_D_errors.append(res['D'][1])  # Error for weighting
    
    # Calculate weighted averages (handle division by zero with epsilon)
    eps = 1e-12
    # Weighted average for DA
    weights_da = 1 / (np.array(DA_errors) + eps)**2
    weighted_avg_da = np.sum(np.array(DA_values) * weights_da) / np.sum(weights_da) if np.sum(weights_da) != 0 else np.nan
    
    # Weighted average for Dphi
    weights_dphi = 1 / (np.array(Dphi_errors) + eps)**2
    weighted_avg_dphi = np.sum(np.array(Dphi_values) * weights_dphi) / np.sum(weights_dphi) if np.sum(weights_dphi) != 0 else np.nan
    
    # Weighted average for new model D
    if new_D_values:  # Avoid empty list issues
        weights_new = 1 / (np.array(new_D_errors) + eps)**2
        weighted_avg_new = np.sum(np.array(new_D_values) * weights_new) / np.sum(weights_new) if np.sum(weights_new) != 0 else np.nan
    else:
        weighted_avg_new = np.nan

    # Plot data without error bars
    plt.figure(figsize=(10, 6))
    plt.plot(i_values, DA_values, 'o-', color='blue', label='DA (Old Model)', markersize=6)
    plt.plot(i_values, Dphi_values, 's--', color='green', label='Dphi (Old Model)', markersize=6)
    plt.plot(new_i_values, new_D_values, '^-.', color='purple', label='D (New Model)', markersize=6)

    # Add weighted average lines
    plt.axhline(weighted_avg_da, color='blue', linestyle=':', linewidth=2, 
                label=f'DA Weighted Avg: {weighted_avg_da:.8f}')
    plt.axhline(weighted_avg_dphi, color='green', linestyle=':', linewidth=2, 
                label=f'Dphi Weighted Avg: {weighted_avg_dphi:.8f}')
    if not np.isnan(weighted_avg_new):
        plt.axhline(weighted_avg_new, color='purple', linestyle=':', linewidth=2, 
                    label=f'New Model Weighted Avg: {weighted_avg_new:.8f}')

    plt.title(f"Diffusivity Estimates with Weighted Averages (Period: {period} s) vs Sensor Index")
    plt.xlabel("Sensor Index")
    plt.ylabel("Diffusivity (m²/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# Disabled: Redundant weighted average plot (now integrated into plot_D)
def plot_D_weighted_avg(new_model_results, period):
    pass  

def temp_model(t, D, C, T0, phi0, omega, x):
    alpha = np.sqrt(omega / (2 * D))
    exp_term = np.exp(-alpha * x)
    sine_term = np.sin(alpha * x - omega * t + phi0)
    if C < 0:
        C = -C
        phi0 += np.pi
        phi0 = (phi0 + np.pi) % (2 * np.pi) - np.pi
    return T0 + C * exp_term * sine_term

def new_initial_guess(t, y, x, omega):
    T0_guess = np.mean(y)
    amplitude_range = (np.max(y) - np.min(y)) / 2
    D_guess = 3e-5
    alpha_guess = np.sqrt(omega / (2 * D_guess))
    exp_term_guess = np.exp(-alpha_guess * x)
    C_guess = amplitude_range / exp_term_guess
    phi0_guess = 0
    return [D_guess, C_guess, T0_guess, phi0_guess]

# Modified: Generate new model results without individual plots
def plot_fit_new(timestamp, thermistor_temperatures, positions, omega_guesses, period):
    n_columns = thermistor_temperatures.shape[1]
    assert len(positions) == n_columns, "Number of positions must match temperature columns"
    assert len(omega_guesses) == n_columns, "omega_guesses length must match number of sensors"
    fit_results = []
    for i in range(n_columns):
        y = thermistor_temperatures[:, i]
        current_x = positions[i]
        current_omega = omega_guesses[i]
        def model_wrapper(t, D, C, T0, phi0):
            return temp_model(t, D, C, T0, phi0, current_omega, current_x)
        try:
            popt, pcov = curve_fit(
                model_wrapper,
                timestamp,
                y,
                p0=new_initial_guess(timestamp, y, current_x, current_omega),
                bounds=([1e-6, 1e-3, np.min(y)-1, -np.pi],
                        [5e-5, 100, np.max(y)+1, np.pi]),
                maxfev=50000
            )
            D_fit, C_fit, T0_fit, phi0_fit = popt
            if C_fit < 0:
                C_fit = -C_fit
                phi0_fit += np.pi
                phi0_fit = (phi0_fit + np.pi) % (2 * np.pi) - np.pi
            perr = np.sqrt(np.diag(pcov))
            D_err, C_err, T0_err, phi0_err = perr
            fit_results.append({
                'column': i,
                'position': current_x,
                'omega': current_omega,
                'D': (D_fit, D_err),
                'C': (C_fit, C_err),
                'T0': (T0_fit, T0_err),
                'phi0': (phi0_fit, phi0_err)
            })
        except RuntimeError as e:
            fit_results.append({
                'column': i,
                'position': current_x,
                'omega': current_omega,
                'error': str(e)
            })
    return fit_results


if __name__ == "__main__":
    periods = [2.5, 5, 10, 15, 20, 30, 40, 45, 50, 60, 80]
    
    for period in periods:
        print(f"Processing period: {period}s")
        omega_guesses = [2 * np.pi / period] * 8
        
        # Load data
        timestamp, output_voltage, output_current, thermistor_temperatures, _ = (
            load_dataset(f"../data/brass data/brass {period}s.csv")
        )
        
        # Get new model results
        positions = calculate_positions()
        new_model_results = plot_fit_new(timestamp, thermistor_temperatures, positions, omega_guesses, period)
        
        # Generate D plot with weighted averages
        plot_D(new_model_results, period)