import numpy as np
from scipy.signal import fftconvolve


def calculate_fwhm_williams_hall(two_theta, wavelength, L, U):
    theta_rad = np.radians(two_theta / 2)
    cos_theta = np.cos(theta_rad)
    tan_theta = np.tan(theta_rad)
    fwhm = (5.72957795130823 * wavelength / (L * cos_theta)) + (U * tan_theta)
    return fwhm


def axial_model_simple(x0, asym_coef):
    return np.abs(asym_coef) / np.tan(np.radians(x0))  # Добавлен np.abs()


def instrumental_profile(x, sigma):
    #return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    g1 = 1 #(sigma * np.sqrt(2 * np.pi))
    g2 = 4.77
    return g1 * np.exp(-g2 * (x / sigma) ** 2)

    #g1 = 0.29903217326 / sigma
    #g2 = 2.77258872224
    #return g1 * np.exp(-g2 * (x / sigma ) ** 2)


def instrumental_profile_asym(x, sigma, asym_coef):
    # Вычисляем ширину для левой и правой частей
    sigma_left = sigma * (1.0 + asym_coef)
    sigma_right = sigma / (1.0 + asym_coef)

    height_factor = 1 #np.sqrt(sigma_left / sigma_right)

    # Разделяем область по центру mu=0
    left_part = x < 0
    right_part = x >= 0

    # Вычисляем значение функции Гаусса для каждой части
    g1 = 1 / (sigma_left * np.sqrt(2 * np.pi))
    g1_left = g1 * height_factor
    g1_right = g1 / height_factor
    g2 = 1 / 6

    result = np.zeros_like(x)
    result[left_part] = g1_left * np.exp(-g2 * (x[left_part] / sigma_left) ** 2)
    result[right_part] = g1_right * np.exp(-g2 * (x[right_part] / sigma_right) ** 2)

    return result


def lorentzian_profile(scale, x, x0, gamma):
    gamma_sqr = 4 / (gamma ** 2)
    intens = scale * (2 / np.pi) / gamma
    return intens / (1 + gamma_sqr * (x - x0) ** 2)

def convolve_lorentzian_instrumental(x, x0, gamma, scale, asym_coef, dx=0.01):
    wight_calc = 15 * gamma

    x_lorentz = np.arange(x0 - wight_calc, x0 + wight_calc, dx)
    lorentz = lorentzian_profile(scale, x_lorentz, x0, gamma)

    asym = axial_model_simple(x0, asym_coef)
    if asym < 0:
        asym = 0

    # Сетка для инструментальной функции (гарантируем корректный диапазон)
    #x_inst = np.arange(-5 * sigma, 5 * sigma + dx, dx)  # +dx для включения верхней границы
    x_inst = np.arange(-wight_calc, wight_calc, dx)
    #inst = instrumental_profile_asym(x_inst, 0.031, 0.06 * x0 * asym)
    #inst = instrumental_profile_asym(x_inst, 0.019, 5*asym) # 4.87
    inst = instrumental_profile_asym(x_inst, 0.0215, 4 * asym) # 4.71

    #print(x0, gamma, asym, np.cos(np.radians(x0 / 2)))

    # Нормировка и свёртка
    inst /= np.trapz(inst, x_inst)
    conv = fftconvolve(lorentz, inst, mode='same') * dx

    return np.interp(x, x_lorentz, conv, left=0, right=0)


def calculate_diffraction_pattern(two_theta_values, peaks, wave_l, wave_b, scale, two_theta_range, step, L=100, U=0.0, Lx=12, Rs=200):
    diffraction_pattern = np.zeros_like(two_theta_values)

    asym_coef = 28.64788975 * (Lx / Rs) ** 2
    if asym_coef < 0:
        asym_coef = 0

    for angle, intens in zip(peaks['angle'], peaks['int']):
        if angle is not None and two_theta_range[0] - 3 <= angle <= two_theta_range[1] + 3:
            broadening = calculate_fwhm_williams_hall(angle, wave_l, L, U)
            broadening *= wave_b

            #corrected_intensity /= np.sin(np.radians(x / 2)**0.5)

            lorentzian = convolve_lorentzian_instrumental(two_theta_values, angle, broadening, intens * scale, asym_coef, step)

            diffraction_pattern += lorentzian
            
    return diffraction_pattern


"""
# Пример использования
peaks = type('Peaks', (object,), {'x': [15], 'y': [1], 'hkls': [[{'hkl': [1, 1, 1]}]]})()
texture_params = []

# Типовые параметры для тестирования
Lx = 16  # Длина источника излучения, мм
R = 215  # Радиус дифрактометра, мм

diffraction_pattern = calculate_diffraction_pattern(peaks, texture_params, 5.431, 5.431, 5.431, 90, 90, 90, 1.54056, 1, 1, (10, 20), 0.02, Lx=Lx, R=R)

# Построение графика дифракционного паттерна
import matplotlib.pyplot as plt

two_theta_values, intensities = zip(*diffraction_pattern)
plt.plot(two_theta_values, intensities)
plt.xlabel('Two-Theta (degrees)')
plt.ylabel('Intensity')
plt.title('Diffraction Pattern with Axial Divergence')
plt.grid(True)
plt.show()
#"""