import numpy as np
import matplotlib.pyplot as plt
from diffraction_utils import initialize_structures, generate_combined_diffraction_pattern, load_experimental_data
from load_xml_phases import *
import time
import os
import random
from typing import List

def generate_tshx_fractions(files) -> List[float]:
    """
    2. Создаёт массив нулей той же длины.
    3. Случайным образом выбирает 5–15 позиций и заполняет их
       случайными значениями в диапазоне 0…1.
    4. Нормирует массив так, чтобы сумма была равна 1.
    5. Возвращает полученный массив.

    Если .tshx-файлов нет, вернёт пустой список.
    """
    n = len(files)
    if n == 0:
        return []

    # 2. Массив нулей той же длины
    fractions = [0.0] * n

    # 3. Случайное количество активных фаз (5–15, но не более n)
    k = min(random.randint(5, 15), n)
    active_indices = random.sample(range(n), k)

    # Заполняем случайные значения
    for idx in active_indices:
        fractions[idx] = random.random()  # 0 ≤ val < 1

    # 4. Нормируем
    total = sum(fractions)
    if total == 0:  # На случай, если все случайные значения оказались 0
        return [1.0 / n] * n  # Равномерное распределение

    fractions = [v / total for v in fractions]

    # 5. Возвращаем
    return fractions

str_folder = r"./DB3/"

two_theta_range = (10, 90)  # in degrees
step = 0.02  # in degrees

#arg_wghts = [2.12, 3.48, 48.43, 27.17, 1.62, 15.24, 0.05, 1.63, 0.26, 0.00, 0.01, 0.00, 0.00]  # B508
arg_wghts = [1.30, 3.43, 67.70, 7.45, 2.77, 10.36, 0.05, 2.19, 0.40, 4.00, 0.00, 0.00, 0.00]  # K2019

arg_wghts = [x * 0.01 for x in arg_wghts]
print(arg_wghts)


def calculate_rwp(calculated_intensities, experimental_intensities, experimental_weights):
    numerator = np.sum(experimental_weights * (calculated_intensities - experimental_intensities) ** 2)
    denominator = np.sum(experimental_weights * experimental_intensities ** 2)
    rwp = np.sqrt(numerator / denominator)
    return rwp

def main():

    global arg_wghts

    experiment_folder = '.\\XY_OSO_test\\'

    experimental_data = load_experimental_data(experiment_folder)
    print(f'Loaded {len(experimental_data)} experimental diffraction patterns.')

    if experimental_data:
        # Получаем первую экспериментальную дифрактограмму из словаря
        first_experiment_filename = next(iter(experimental_data))
        first_experiment_intensities = experimental_data[first_experiment_filename]
        print("\nPlot of the sample", first_experiment_filename)

        start_time = time.time()  # Запоминаем время начала обучения

        # Список .tshx-файлов
        files = [
            os.path.join(str_folder, f)  # полный путь к файлу
            for f in os.listdir(str_folder)
            if f.lower().endswith('.tshx')
        ]
        files_n = len(files)
        if files_n == 0:
            exit(0)

        nonzero_indices = [i for i, w in enumerate(arg_wghts) if w > 0]
        arg_wghts = [arg_wghts[i] for i in nonzero_indices]
        files = [files[i] for i in nonzero_indices]
        print(arg_wghts)

        sample_info, PHASES_INFO = parse_xml_multi('.\\XRD_Data\\OSO_Topas_2022_stidy.tshx', files)
        print(f'Loaded {len(PHASES_INFO)} phases.')
        initialize_structures(PHASES_INFO, two_theta_range, step)

        start_time = time.time()  # Запоминаем время начала
        print('Начало вычисления профиля...')

        # Генерируем новую комбинированную дифрактограмму с оптимизированными параметрами
        combined_two_theta_values, combined_intensity_values, weights = generate_combined_diffraction_pattern(
            sample_info, PHASES_INFO, two_theta_range, step, arg_wghts
        )

        min_len = min(len(combined_two_theta_values),
                      len(combined_intensity_values),
                      len(first_experiment_intensities))
        combined_two_theta_values = combined_two_theta_values[:min_len]
        combined_intensity_values = combined_intensity_values[:min_len]
        first_experiment_intensities = first_experiment_intensities[:min_len]

        start_index = np.where(combined_two_theta_values > 5.8)[0][0]
        combined_two_theta_values = combined_two_theta_values[start_index:]
        combined_intensity_values = combined_intensity_values[start_index:]
        first_experiment_intensities = first_experiment_intensities[start_index:]

        print(f"Total len = {len(combined_two_theta_values)}")

        elapsed_time = time.time() - start_time  # Вычисляем прошедшее время
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Прошло времени: {int(hours)}:{int(minutes)}:{int(seconds)}')
        
        # Рассчитываем Rwp после оптимизации
        rwp = calculate_rwp(combined_intensity_values, first_experiment_intensities, first_experiment_intensities)
        print(f'Rwp factor after optimization: {(rwp*100):.2f}%')
        
        # Рассчитываем разницу между профилями
        difference = combined_intensity_values - first_experiment_intensities

        # Смещаем разностную линию, чтобы она была ниже нулевого уровня
        max_difference = np.max(difference)
        shifted_difference = difference - max_difference
        
        # Построение графика
        plt.figure(figsize=(17, 11))
        
        # График расчетной и экспериментальной дифрактограммы
        plt.plot(combined_two_theta_values, combined_intensity_values, label='Генерация', color='blue', linewidth=1)
        plt.plot(combined_two_theta_values, first_experiment_intensities, label=f'Образец {first_experiment_filename}', color='red', linewidth=1)
        
        # График разностной линии
        plt.plot(combined_two_theta_values, shifted_difference, label='Разностная линия', color='green', linestyle='--')
        
        plt.title('X-ray Diffraction Patterns')
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlim(two_theta_range)

        # Настройка расположения окна графика в углу экрана
        #manager = plt.get_current_fig_manager()
        # Для Windows
        #manager.window.wm_geometry("+0+0")
        
        plt.tight_layout()
        plt.show()
    else:
        print("No experimental data to plot.")


if __name__ == "__main__":
    main()