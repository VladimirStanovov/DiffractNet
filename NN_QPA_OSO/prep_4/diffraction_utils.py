import os
import random
import math
import numpy as np
from diffraction_module7 import calculate_diffraction_pattern
from typing import List

from objCryst_dll import CrystLib


"""
lam = [
    [0.0159, 1.534753, 3.6854],
    [0.5791, 1.540596, 0.437],
    [0.0762, 1.541058, 0.6],
    [0.2417, 1.54441, 0.52],
    [0.0871, 1.544721, 0.62]
]
"""

lam = [
    [1.0, 1.540598, 1.0],
    [0.514, 1.544426, 1.0]
]

lib = CrystLib()


def erase_lib_memory():
    lib.FreeMemory()

def calculate_peaks_with_angle(peaks, wavelength):
    # Извлекаем данные из списка пиков
    d_values = []
    int_values = []
    angles = []

    for peak in peaks:
        d = peak['d']
        intensity = peak['int']

        # Проверяем условие d <= wavelength / 2
        if d <= wavelength / 2:
            break

        # Рассчитываем угол 2theta для текущего пика
        two_theta = 2 * math.asin(wavelength / (2 * d))
        angle_degrees = math.degrees(two_theta)

        # Добавляем значения в соответствующие списки
        d_values.append(d)
        int_values.append(intensity)
        angles.append(angle_degrees)

    # Преобразуем списки в numpy-массивы
    d_array = np.array(d_values)
    int_array = np.array(int_values)
    angle_array = np.array(angles)

    # Создаем словарь с numpy-массивами
    peaks_with_angle = {
        'angle': angle_array,
        'd': d_array,
        'int': int_array
    }

    return peaks_with_angle

def generate_random_phase_parameters(phase_info):
    L = random.uniform(*phase_info['L_range'])
    U = random.uniform(*phase_info['U_range'])
    
    return L, U

def generate_lognormal_weights(n):
    # Генерация логнормально распределенных значений
    mu, sigma = 0, 0.9  # Параметры среднего и стандартного отклонения
    lognorm_values = np.random.lognormal(mu, sigma, n)
    # Нормализация значений так, чтобы их сумма была равна 1
    total_weight = sum(lognorm_values)
    return [w / total_weight for w in lognorm_values]

def generate_pareto_weights(n, alpha=2.0):
    # Генерация значений согласно распределению Парето
    pareto_values = (np.random.pareto(alpha, n) + 1)
    # Нормализация значений так, чтобы их сумма была равна 1
    total_weight = sum(pareto_values)
    return [w / total_weight for w in pareto_values]


def _lognormal_weights(k: int) -> List[float]:
    """Случайные веса из логнормального распределения."""
    mu, sigma = 0, 1.2  # sigma > 1 даёт более «тяжёлый» хвост
    vals = np.random.lognormal(mu, sigma, k)
    return vals.tolist()


def _pareto_weights(k: int, alpha: float = 1.0) -> List[float]:
    """Случайные веса из распределения Парето."""
    vals = np.random.pareto(alpha, k) + 1
    return vals.tolist()


def _dirichlet_weights(k: int) -> List[float]:
    """Генерация через Dirichlet с маленькими alpha (неравномерно)."""
    alpha = np.random.uniform(0.05, 0.8, k)
    vals = np.random.dirichlet(alpha, size=1)[0]
    return vals.tolist()


def generate_ph_fractions(n: int) -> List[float]:
    if n <= 0:
        return []

    # 1. Выбираем, сколько «активных» позиций будет
    k = min(random.randint(5, 15), n)

    # 2. Выбираем, каким способом заполнять активные веса
    r = random.random()
    if r < 0.05:
        raw_weights = [random.random() for _ in range(k)]
    elif r < 0.35:
        raw_weights = _lognormal_weights(k)
    elif r < 0.70:
        raw_weights = _pareto_weights(k)
    else:
        raw_weights = _dirichlet_weights(k)

    # 3. Случайно выбираем k индексов
    active_indices = random.sample(range(n), k)

    # 4. Размещаем веса в fractions
    fractions = [0.0] * n
    for idx, w in zip(active_indices, raw_weights):
        fractions[idx] = w

    # 5. Нормализация
    total = sum(fractions)
    if total == 0:
        return [1.0 / n] * n
    result = [v / total for v in fractions]
    return result

def is_param_refinable(param):
    if param['min'] is not None and param['max'] is not None:
        return True
    else:
        return False

def get_random_value(param, reference=None, initial_value=None):
    if reference is not None and initial_value is not None and param['value'] == initial_value:
        return reference

    p_min = param['min']
    p_max = param['max']

    if p_min is not None and p_max is not None:
        return random.uniform(p_min, p_max)
    else:
        return param['value']

def create_structure(phase_info, cr_idx):
    global lib
    a = get_random_value(phase_info['CellPar']['a'])
    b = get_random_value(phase_info['CellPar']['b'], a, phase_info['CellPar']['a']['value'])
    c = get_random_value(phase_info['CellPar']['c'], a, phase_info['CellPar']['a']['value'])
    alpha = get_random_value(phase_info['CellPar']['alpha'])
    beta = get_random_value(phase_info['CellPar']['beta'])
    gamma = get_random_value(phase_info['CellPar']['gamma'])

    lib.AddCrystal(a, b, c, alpha, beta, gamma, phase_info['SpaceGroup'])

    at_idx = 0
    for atom in phase_info['Atoms']:
        lib.AddScatteringPowerToCrystal(cr_idx, atom['Type'], atom['Type'], 1.0)
        lib.AddAtomToCrystal(cr_idx, at_idx, get_random_value(atom['X']), get_random_value(atom['Y']), get_random_value(atom['Z']), get_random_value(atom['Occ']))
        at_idx += 1

    return True

def modify_structure(phase_info, cr_idx):
    global lib
    a = get_random_value(phase_info['CellPar']['a'])
    b = get_random_value(phase_info['CellPar']['b'], a, phase_info['CellPar']['a']['value'])
    c = get_random_value(phase_info['CellPar']['c'], a, phase_info['CellPar']['a']['value'])
    alpha = get_random_value(phase_info['CellPar']['alpha'])
    beta = get_random_value(phase_info['CellPar']['beta'])
    gamma = get_random_value(phase_info['CellPar']['gamma'])

    lib.SetCrystalCellParameters(cr_idx, a, b, c, alpha, beta, gamma)

    at_idx = 0
    for atom in phase_info['Atoms']:
        if is_param_refinable(atom['X']) or is_param_refinable(atom['Y']) or is_param_refinable(atom['Z']) or is_param_refinable(atom['Occ']):
            lib.SetAtomParameters(cr_idx, at_idx, get_random_value(atom['X']), get_random_value(atom['Y']), get_random_value(atom['Z']), get_random_value(atom['Occ']))
        at_idx += 1

    return True

def initialize_structures(PHASES_INFO, two_theta_range, step):
    global lib
    cr_idx = 0
    for phase_info in PHASES_INFO:
        create_structure(phase_info, cr_idx)
        cr_idx += 1

    lib.InitializePowderData(lam[0][1], two_theta_range[0], step, len(np.arange(two_theta_range[0], two_theta_range[1] + step, step)))

    cr_idx = 0
    for phase_info in PHASES_INFO:
        for tex_par in phase_info['MDtexDir']:
            h, k, l = map(float, tex_par['name'].strip().split())
            lib.AddTextureToDiffData(cr_idx, random.uniform(tex_par['min'], tex_par['max']), (h), (k), (l))
        cr_idx += 1

    return

def generate_combined_diffraction_pattern(sample_info, PHASES_INFO, two_theta_range, step, weights):
    #print("weights = ", weights)
    #print("sum(weights) = ", sum(weights))

    Lx = sample_info['SAM']
    Rs = sample_info['Rs']

    x_shift = random.uniform(-sample_info['ZE'], sample_info['ZE'])
    #x_shift = 0.032
    
    # Создаем массивы для хранения значений 2θ и интенсивностей
    two_theta_values = np.arange(two_theta_range[0], two_theta_range[1] + step, step)
    combined_intensity_values = np.zeros_like(two_theta_values)

    cr_idx = 0
    # Проход по всем фазам
    for weight, phase_info in zip(weights, PHASES_INFO):

        modify_structure(phase_info, cr_idx)
        #print(phase_info['Name'])

        tex_idx = 0
        for tex_par in phase_info['MDtexDir']:
            lib.SetTextureCParameter(cr_idx, tex_idx, random.uniform(tex_par['min'], tex_par['max']))
            tex_idx += 1

        RIR = lib.CalcRIR(cr_idx)
        #print(RIR)

        L = get_random_value(phase_info['ReflectionProfileScherrer']['CS'])
        U = get_random_value(phase_info['ReflectionProfileScherrer']['Strain'])

        orig_peaks = lib.GetNormalizedPeaks(cr_idx)
        #print(orig_peaks)

        for wave in lam:  # Прибавляем Ka2

            peaks = calculate_peaks_with_angle(orig_peaks, wave[1])

            # Случайный сдвиг нуля
            peaks['angle'] += x_shift

            scale = weight * RIR * wave[0]

            diffraction_pattern = calculate_diffraction_pattern(two_theta_values,
                peaks, wave[1], wave[2], scale,
                two_theta_range, step, L, U, Lx, Rs
            )

            combined_intensity_values += diffraction_pattern

        cr_idx += 1

    # Добавляем начальный наклон фона
    max_intensity = np.max(combined_intensity_values)
    quarter_max_intensity = max_intensity * random.uniform(0.05, 0.6)
    combined_intensity_values += quarter_max_intensity / two_theta_values

    # Добавление линии фона
    max_profile_int = np.max(combined_intensity_values)
    ###bkg_weight = 0.0 # random.uniform(0.2, 5.0)
    dev_int = random.uniform(0.001, 0.002)
    std_dev = dev_int * max_profile_int  # стандартное отклонение 1% от максимальной интенсивности
    #background = np.random.normal(loc=bkg_weight, scale=std_dev, size=len(two_theta_values))
    background = np.random.normal(loc=0, scale=std_dev, size=len(two_theta_values))
    background += (-np.min(background)) ###+ bkg_weight * max_profile_int
    combined_intensity_values += background
    
    # Находим минимальное значение интенсивности профиля
    min_intensity = np.min(combined_intensity_values)
    
    # Вычитаем минимальное значение из каждой точки профиля, если оно больше нуля
    if min_intensity > 0:
        combined_intensity_values -= min_intensity
    
    # Удаление отрицательных значений
    combined_intensity_values = np.maximum(combined_intensity_values, 0)
    
    # Нормировка интенсивностей к 1
    max_total_int = np.max(combined_intensity_values)
    if max_total_int > 0:
        combined_intensity_values = (combined_intensity_values / max_total_int)

    return two_theta_values, combined_intensity_values, weights


def load_experimental_data(folder_path, cut_angle = 0.0):
    experimental_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.xy'):
            filepath = os.path.join(folder_path, filename)
            #print(filepath)
            try:
                data = np.loadtxt(filepath, delimiter=None, comments='#')  # Автоматическое определение разделителя
                #angles = data[:, 0]
                intensities = data[:, 1]

                # Определение минимальной интенсивности
                min_intensity = np.min(intensities)

                # Вычитание минимальной интенсивности из каждой точки
                normalized_intensities = intensities - min_intensity

                # Нормализация интенсивностей к значению 1.0
                max_intensity = np.max(normalized_intensities)
                if max_intensity > 0:
                    normalized_intensities /= max_intensity
                else:
                    normalized_intensities = np.zeros_like(normalized_intensities)

                if cut_angle > 0:
                    angles = data[:, 0]
                    start_index = np.where(angles > 5.8)[0][0]
                    normalized_intensities = normalized_intensities[start_index:]

                experimental_data[filename] = normalized_intensities
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return experimental_data
