import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


def load_dataset(path):
    data = pd.read_csv(path, header=3)
    timestamp = data.iloc[:, 0].to_numpy()  
    output_voltage = data.iloc[:, 1].to_numpy()
    output_current = data.iloc[:, 2].to_numpy()
    thermistor_temperatures = data.iloc[:, 3:].to_numpy()  
    
    return timestamp, output_voltage, output_current, thermistor_temperatures


def calculate_positions():
    L = 0.041  
    bottom_distance = 0.003  
    spacing = 0.005  
    
    positions = []
    for i in range(8):
        pos = bottom_distance + i * spacing
        positions.append(pos)
    
    return np.array(positions)

def temperature_model(t, D, C_minus, a, b, omega, x):
    k = np.sqrt(omega / (2 * D))
    L = 0.041  
    C_plus = C_minus * np.exp(2 * L * k * (1 - 1j))
    beta = np.sqrt(omega / (2 * D)) * (1j - 1)  
    term = C_plus * np.exp(beta * x) + C_minus * np.exp(-beta * x)
    fluctuating_temp = np.real(term * np.exp(-1j * omega * t))
    total_temperature = (a + b * t) + fluctuating_temp
    
    return total_temperature

def fit_function(t, D, C_minus, a, b):
    global current_x
    return temperature_model(t, D, C_minus, a, b, omega, current_x)

def fit_diffusivity(timestamp, temperatures, positions):
    results = []
    
    for i, x in enumerate(positions):
        print(f"\nFitting the {i+1}th thermometer ({x:.4f}m)")
        global current_x
        current_x = x
        
        T_measured = temperatures[:, i]
        
        initial_guess = [
            3e-5,                
            1.0,                 
            np.mean(T_measured), 
            0.0                  
        ]
        
        bounds = (
            [1e-8, -10.0, np.min(T_measured) - 2, -0.1],  
            [1e-3, 10.0, np.max(T_measured) + 2, 0.1]     
        )
        
        try:
            popt, pcov = curve_fit(
                fit_function, 
                timestamp, 
                T_measured,
                p0=initial_guess,
                bounds=bounds,
                maxfev=30000
            )
            
            D_fit, C_minus_fit, a_fit, b_fit = popt
            perr = np.sqrt(np.diag(pcov))
            D_err, C_minus_err, a_err, b_err = perr

            k_fit = np.sqrt(omega / (2 * D_fit))
            L = 0.041  
            C_plus_fit = C_minus_fit * np.exp(2 * L * k_fit * (1 - 1j))
            C_plus_err = C_minus_err * np.exp(2 * L * k_fit * (1 - 1j))
            
            results.append({
                'position': x,
                'D': D_fit,
                'D_error': D_err,
                'C_minus': C_minus_fit,
                'C_minus_error': C_minus_err,
                'C_plus' : C_plus_fit,
                'C_plus_error' : C_plus_err,
                'a': a_fit,
                'a_error': a_err,
                'b': b_fit,
                'b_error': b_err
            })
            
            print(f"Fitting complete - D = {D_fit:.8f} ± {D_err:.8f} m²/s, "
                  f"C- = {C_minus_fit:.6f} ± {C_minus_err:.6f}, "
                  f"C+ = {C_plus_fit:.6f} ± {C_plus_err:.6f}, "
                  f"a = {a_fit:.4f} ± {a_err:.4f}, "
                  f"b = {b_fit:.6f} ± {b_err:.6f}"
            )
            
            plot_fit(timestamp, T_measured, x, D_fit, C_minus_fit, a_fit, b_fit)
            
        except RuntimeError as e:
            print(f"Fitting failed: {e}")
            results.append(None)
    
    return results

def plot_fit(t, T_measured, x, D, C_minus, a, b):
    k = np.sqrt(omega / (2 * D))
    L = 0.041
    C_plus = C_minus * np.exp(2 * L * k * (1 - 1j))
    
    T_fitted = temperature_model(t, D, C_minus, a, b, omega, x)
    t_baseline = np.linspace(min(t), max(t), 100)
    baseline = a + b * t_baseline
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, T_measured, 'b.', label='Measured Data')
    plt.plot(t, T_fitted, 'r-', label='Fitted Curve')
    plt.plot(t_baseline, baseline, 'g--', 
             label=f'C_plus: {C_plus:.6f}\nC_minus: {C_minus:.6f}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Fit at Position x = {x:.4f} m (period = {period}s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def analyze_results(results):
    valid_results = [res for res in results if res is not None]
    
    if not valid_results:
        print("No valid fitting results")
        return None
    
    D_values = [res['D'] for res in valid_results]
    D_errors = [res['D_error'] for res in valid_results]
    a_values = [res['a'] for res in valid_results]
    b_values = [res['b'] for res in valid_results]
    C_minus_values = [res['C_minus'] for res in valid_results]
    C_plus_values = [res['C_plus'] for res in valid_results]
    positions = [res['position'] for res in valid_results]
    
    weights_D = 1 / np.square(D_errors)
    D_avg = np.sum(np.array(D_values) * weights_D) / np.sum(weights_D)
    D_avg_unweighted = np.mean(D_values)
    
    print("\n===== Fitting Results Statistics =====")
    print(f"D values at each position: {[f'{d:.8f}' for d in D_values]}")
    print(f"Weighted average of D: {D_avg:.8f} m²/s")
    print(f"Unweighted average of D: {D_avg_unweighted:.8f} m²/s")
    print(f"\nLinear baseline a (intercept) values: {[f'{a:.4f}' for a in a_values]}")
    print(f"Linear baseline b (slope) values: {[f'{b:.6f}' for b in b_values]}")
    print(f"\nC_minus values at each position: {[f'{c:.6f}' for c in C_minus_values]}")
    print(f"C_plus values at each position: {[f'{c:.6f}' for c in C_plus_values]}")
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(positions, D_values, yerr=D_errors, fmt='o', color='b', label='D values')
    plt.plot(positions, D_values, 'b-', alpha=0.7, label='D trend')
    plt.axhline(D_avg, color='r', linestyle='--', label=f'Weighted Avg D = {D_avg:.8f} m²/s')
    plt.axhline(D_avg_unweighted, color='blue', linestyle='-.', label=f'Unweighted Avg D = {D_avg_unweighted:.8f} m²/s')
    
    for x, d, c_plus, c_minus in zip(positions, D_values, C_plus_values, C_minus_values):
        text = f"C_plus: {c_plus:.6f}\nC_minus: {c_minus:.6f}"
        plt.annotate(
            text,
            (x, d),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            color='darkgreen'
        )
    
    plt.xlabel('Position (m)')
    plt.ylabel('Diffusivity D (m²/s)')
    plt.title(f'Fitted Diffusivity Values with C_plus and C_minus Parameters (period = {period})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'D_avg': D_avg,
        'D_avg_unweighted': D_avg_unweighted,
        'a_values': a_values,
        'b_values': b_values,
        'C_minus_values': C_minus_values,
        'C_plus_values': C_plus_values     
    }

        
if __name__ == "__main__":
    periods = [2.5, 5, 10, 15, 20, 30, 40, 45, 50, 60, 80]
    
    for i in periods:
        period = i
        omega = 2 * np.pi / period 
        
        timestamp, output_voltage, output_current, thermistor_temperatures = (
            load_dataset(f"../data/brass data/brass {i}s.csv")
        )
        
        positions = calculate_positions()
        
        results = fit_diffusivity(timestamp, thermistor_temperatures, positions)
        
        average_results = analyze_results(results)
        
        if average_results:
            print(f"\nFinal average D for period {i}s = {average_results['D_avg']:.8f} m²/s")
            print(f"Unweighted average D for period {i}s = {average_results['D_avg_unweighted']:.8f} m²/s")    
            
            
            
            
            
            
            
            
            
            