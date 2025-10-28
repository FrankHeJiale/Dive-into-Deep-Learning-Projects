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

def plot_fit(period):
    n_columns = thermistor_temperatures.shape[1]
    assert len(omega_guesses) == n_columns, "Length of omega_guesses must match number of columns"
    positions = calculate_positions()
    fit_results = []  
    for i in range(n_columns):
        y = thermistor_temperatures[:, i]
        omega_0 = omega_guesses[i]
        try:
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
            perr = np.sqrt(np.diag(pcov))
            A_err, omega_err, phi_err, B_err, C_err = perr
            period_calc = 2 * np.pi / omega if omega != 0 else np.nan
            period_err = (2 * np.pi / omega**2) * omega_err if omega != 0 else np.nan
            fit_results.append({
                'column': i,
                'A': (A, A_err),
                'omega': (omega, omega_err),
                'phi': (phi, phi_err),
                'B': (B, B_err),
                'C': (C, C_err),
                'period': (period_calc, period_err)
            })
            print(f"\nColumn {i} fit results:")
            print(f"  A     = {A:.4f} ± {A_err:.4f}")
            print(f"  omega = {omega:.6f} ± {omega_err:.6f} rad/s")
            print(f"  phi   = {phi:.4f} ± {phi_err:.4f} rad")
            print(f"  B     = {B:.6f} ± {B_err:.6f} (slope)")
            print(f"  C     = {C:.4f} ± {C_err:.4f} (offset)")
            print(f"  Period = {period_calc:.4f} ± {period_err:.4f} s")
            t_fit = np.linspace(min(timestamp), max(timestamp), 1000)
            y_fit = sin_plus_line(t_fit, *popt)
            transient_line = B * t_fit + C
            pos = positions[i]
            if i == 0:
                da_str = "N/A (reference)"
                dphi_str = "N/A (reference)"
            else:
                da_dphi = DA_DPhi(0, i)
                da_str = f"{da_dphi['DA']:.8f} ± {da_dphi['DA_error']:.8f}"
                dphi_str = f"{da_dphi['Dphi']:.8f} ± {da_dphi['Dphi_error']:.8f}"
            plt.figure(figsize=(10, 6))
            plt.plot(timestamp, y, 'o', label='Measured Data', markersize=4)
            plt.plot(t_fit, y_fit, '-', label='Fitted Curve', linewidth=2)
            plt.plot(t_fit, transient_line, '--', label='Transient Line', linewidth=2, color='orange')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature')
            plt.title(f'Fit for Distance {pos:.3f} m (Period: {period} s)\nDA = {da_str}, DPhi = {dphi_str}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except RuntimeError as e:
            print(f"Fit failed for column {i}: {e}")
            fit_results.append({'column': i, 'error': str(e)})
    for res in fit_results:
        if 'error' in res:
            print(f"\nColumn {res['column']}: Fit failed ({res['error']})")
        else:
            print(f"\nColumn {res['column']} fit results:")
            print(f"  A     = {res['A'][0]:.4f} ± {res['A'][1]:.4f}")
            print(f"  omega = {res['omega'][0]:.6f} ± {res['omega'][1]:.6f} rad/s")
            print(f"  phi   = {res['phi'][0]:.4f} ± {res['phi'][1]:.4f} rad")
            print(f"  B     = {res['B'][0]:.6f} ± {res['B'][1]:.6f} (slope)")
            print(f"  C     = {res['C'][0]:.4f} ± {res['C'][1]:.4f} (offset)")
            print(f"  Period = {res['period'][0]:.4f} ± {res['period'][1]:.4f} s")

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
    A_err, omega_err, phi_err, B_err, C_err = perr
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
    comments = re.search(r"Comments: (.*)$", content, re.MULTILINE)[1]
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

def plot_D(new_model_results, period):
    i_values = []
    DA_values = []
    DA_err_values = []
    Dphi_values = []
    Dphi_err_values = []
    for i in range(7):
        result = DA_DPhi(0, i + 1)
        i_values.append(i + 1)
        DA_values.append(result["DA"])
        DA_err_values.append(result["DA_error"])
        Dphi_values.append(result["Dphi"])
        Dphi_err_values.append(result["Dphi_error"])
    
    new_D_values = []
    new_D_err_values = []
    new_i_values = []
    for res in new_model_results:
        if 'error' not in res:
            new_i_values.append(res['column'])
            new_D_values.append(res['D'][0])
            new_D_err_values.append(res['D'][1])

    plt.figure(figsize=(10, 6))
    plt.errorbar(i_values, DA_values, yerr=DA_err_values, fmt='o-', capsize=5, label='DA (Old Model) ± Error', markersize=6)
    plt.errorbar(i_values, Dphi_values, yerr=Dphi_err_values, fmt='s--', capsize=5, label='Dphi (Old Model) ± Error', markersize=6)
    plt.errorbar(new_i_values, new_D_values, yerr=new_D_err_values, fmt='^-.', capsize=5, color='purple', label='D (New Model) ± Error', markersize=6)

    plt.title(f"Diffusivity Estimates from Old and New Models (Period: {period} s) vs Sensor Index")
    plt.xlabel("Sensor Index")
    plt.ylabel("Diffusivity (m²/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_D_weighted_avg(new_model_results, period):
    i_values = []
    DA_values = []
    DA_err_values = []
    Dphi_values = []
    Dphi_err_values = []
    for i in range(7):
        result = DA_DPhi(0, i + 1)
        i_values.append(i + 1)
        DA_values.append(result["DA"])
        DA_err_values.append(result["DA_error"])
        Dphi_values.append(result["Dphi"])
        Dphi_err_values.append(result["Dphi_error"])
    
    new_D_values = []
    new_D_err_values = []
    new_i_values = []
    for res in new_model_results:
        if 'error' not in res:
            new_i_values.append(res['column'])
            new_D_values.append(res['D'][0])
            new_D_err_values.append(res['D'][1])

    eps = 1e-12
    weights_da = 1 / (np.array(DA_err_values) + eps)**2
    weighted_avg_da = np.sum(np.array(DA_values) * weights_da) / np.sum(weights_da)

    weights_dphi = 1 / (np.array(Dphi_err_values) + eps)**2
    weighted_avg_dphi = np.sum(np.array(Dphi_values) * weights_dphi) / np.sum(weights_dphi)

    valid_new_D = np.array(new_D_values)
    valid_new_D_err = np.array(new_D_err_values)
    weights_new = 1 / (valid_new_D_err + eps)**2
    weighted_avg_new = np.sum(valid_new_D * weights_new) / np.sum(weights_new)

    plt.figure(figsize=(10, 6))
    plt.plot(i_values, DA_values, 'o-', color='blue', label='DA (Old Model)', markersize=6)
    plt.plot(i_values, Dphi_values, 's--', color='green', label='Dphi (Old Model)', markersize=6)
    plt.plot(new_i_values, new_D_values, '^-.', color='purple', label='D (New Model)', markersize=6)

    plt.axhline(weighted_avg_da, color='blue', linestyle=':', linewidth=2, label=f'DA Weighted Avg: {weighted_avg_da:.8f}')
    plt.axhline(weighted_avg_dphi, color='green', linestyle=':', linewidth=2, label=f'Dphi Weighted Avg: {weighted_avg_dphi:.8f}')
    plt.axhline(weighted_avg_new, color='purple', linestyle=':', linewidth=2, label=f'New Model Weighted Avg: {weighted_avg_new:.8f}')

    plt.title(f"Diffusivity Estimates with Weighted Averages (Period: {period} s)")
    plt.xlabel("Sensor Index")
    plt.ylabel("Diffusivity (m²/s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

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

def plot_fit_new(timestamp, thermistor_temperatures, positions, omega_guesses, period):
    n_columns = thermistor_temperatures.shape[1]
    assert len(positions) == n_columns, "Number of positions must match temperature columns"
    assert len(omega_guesses) == n_columns, "omega_guesses length must match number of sensors"
    fit_results = []
    for i in range(n_columns):
        y = thermistor_temperatures[:, i]
        current_x = positions[i]
        current_omega = omega_guesses[i]
        print(f"\nFitting sensor {i+1} at position {current_x:.3f} m (omega = {current_omega:.6f} rad/s)...")
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
            print(f"  D (diffusivity): {D_fit:.8f} ± {D_err:.8f} m²/s")
            print(f"  C (amplitude): {C_fit:.4f} ± {C_err:.4f}")
            print(f"  T0 (baseline): {T0_fit:.4f} ± {T0_err:.4f}")
            print(f"  phi0 (phase): {phi0_fit:.4f} ± {phi0_err:.4f} rad")
            t_fit = np.linspace(min(timestamp), max(timestamp), 1000)
            y_fit = model_wrapper(t_fit, D_fit, C_fit, T0_fit, phi0_fit)
            plt.figure(figsize=(10, 6))
            plt.plot(timestamp, y, 'o', label='Measured Data', markersize=4, alpha=0.6)
            plt.plot(t_fit, y_fit, '-', label='Fitted Curve', linewidth=2, color='r')
            plt.axhline(T0_fit, color='g', linestyle='--', label=f'Baseline T0 = {T0_fit:.2f}')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature')
            plt.title(f'Temperature Fit at x = {current_x:.3f} m (Period: {period} s)\nD = {D_fit:.8f} ± {D_err:.8f} m²/s')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        except RuntimeError as e:
            print(f"Fitting failed for sensor {i+1}: {e}")
            fit_results.append({
                'column': i,
                'position': current_x,
                'omega': current_omega,
                'error': str(e)
            })
    return fit_results


period = 50
omega_guesses = [2 * np.pi / period] * 8

timestamp, output_voltage, output_current, thermistor_temperatures, comments = (
    load_dataset("../data/brass data/brass 50s.csv")
)

plot_fit(period)

positions = calculate_positions()
new_model_results = plot_fit_new(timestamp, thermistor_temperatures, positions, omega_guesses, period)    

plot_D(new_model_results, period)
plot_D_weighted_avg(new_model_results, period)

# if __name__ == "__main__":
#     periods = [2.5, 5, 10, 15, 20, 30, 40, 45, 50, 60, 80]
    
#     for i in periods:
#         period = i
#         omega_guesses = 2 * np.pi / period 
        
#         timestamp, output_voltage, output_current, thermistor_temperatures = (
#             load_dataset(f"../data/brass data/brass {i}s.csv")
#         )
        
#         plot_fit(period)
        
#         positions = calculate_positions()
#         new_model_results = plot_fit_new(timestamp, thermistor_temperatures, positions, omega_guesses, period)  
        
#         plot_D(new_model_results, period)
#         plot_D_weighted_avg(new_model_results, period)
        
