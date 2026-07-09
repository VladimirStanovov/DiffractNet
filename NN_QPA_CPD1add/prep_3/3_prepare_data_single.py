import os
import numpy as np
import torch
from torch.utils.data import Dataset
from diffraction_utils import initialize_structures, generate_combined_diffraction_pattern, erase_lib_memory, generate_ph_fractions
from load_xml_phases import *
import time

import struct

difr_dir = "D:\\nn_data\\"
os.makedirs(difr_dir, exist_ok=True)

input_size = 4001
num_nn_phases = 3

target_samples = 15000  # Целевое количество спектров
xml_name = f'CPD1_add7_{num_nn_phases}ph_p4_{target_samples}'

num_smp_per_run = 100

PHASES_INFO = []
sample_info = []

two_theta_range = (15, 95)  # in degrees
step = 0.02  # in degrees

folder = r"../DB3_add7/"
files = [
    os.path.join(folder, f)          # полный путь к файлу
    for f in os.listdir(folder)
    if f.lower().endswith('.tshx')
]
files = sorted(files)
files_n = len(files)
if files_n == 0:
    exit(0)

sample_info, PHASES_INFO = parse_xml_multi('..\\xrd_deivce_data.tshx', files)
print(f'Loaded {len(PHASES_INFO)} phases.')


class DiffractionDataset(Dataset):
    def __init__(self, _two_theta_range, _step, num_samples=100):
        self.two_theta_range = _two_theta_range
        self.step = _step
        self.num_samples = num_samples
        self.batch_inp = []
        self.batch_out = []
        # self.generate_all_patterns()  # Генерируем все дифрактограммы при инициализации

    def __len__(self):
        return len(self.batch_inp)  # Возвращаем фактическое количество спектров

    def __getitem__(self, idx):
        combined_intensity_values, weights = self.batch_inp[idx], self.batch_out[idx]
        combined_intensity_values = torch.tensor(combined_intensity_values).unsqueeze(0)  # Добавляем канал
        weights = torch.tensor(weights, dtype=torch.float32)
        return combined_intensity_values, weights

    def generate_all_patterns(self):
        self.batch_inp = []
        self.batch_out = []

        generated_count = 0  # Счетчик фактически сгенерированных спектров

        for iter in range(0, self.num_samples):
            arg_wghts = generate_ph_fractions(len(PHASES_INFO))

            # 1. Проверяем сумму первых трёх фаз
            sum_first_three = sum(arg_wghts[:num_nn_phases])
            if sum_first_three < 0.01:
                continue  # Пропускаем этот спектр

            # 2. Индексы ненулевых весов
            nonzero_indices = [i for i, w in enumerate(arg_wghts) if w > 0]
            # 3. Сокращаем оба списка
            curr_wghts = [arg_wghts[i] for i in nonzero_indices]
            CURR_PHASES = [PHASES_INFO[i] for i in nonzero_indices]

            initialize_structures(CURR_PHASES, two_theta_range, step)

            _, combined_intensity_values, weights = generate_combined_diffraction_pattern(sample_info, CURR_PHASES,
                                                                                          self.two_theta_range,
                                                                                          self.step, curr_wghts)
            self.batch_inp.append(np.array(combined_intensity_values, dtype=np.float32))

            # 4. Преобразуем выходной массив
            # Нормируем первые три фазы относительно друг друга
            first_three_weights = np.array(arg_wghts[:num_nn_phases], dtype=np.float32)
            norm_first_three = first_three_weights / sum_first_three if sum_first_three > 0 else first_three_weights

            # Создаём выходной массив: суммарная доля + нормированные доли
            output_array = np.concatenate([[sum_first_three], norm_first_three])
            self.batch_out.append(output_array.astype(np.float32))

            generated_count += 1  # Увеличиваем счетчик сгенерированных спектров

            erase_lib_memory()

        return generated_count  # Возвращаем количество фактически записанных спектров


def save_batch_inp_file(arg_inps):
    with open(f'{difr_dir}batch_inp_{xml_name}.bin', 'ab') as f:
        for inp in arg_inps:
            f.write(struct.pack(f'{input_size}f', *inp))


def save_batch_out_file(arg_outs):
    with open(f'{difr_dir}batch_out_{xml_name}.bin', 'ab') as f:
        for out in arg_outs:
            f.write(struct.pack(f'{num_nn_phases + 1}f', *out))


def main():
    # Вычисляем размер входа для свертки
    data_size = int((two_theta_range[1] - two_theta_range[0]) / step) + 1
    print('data_size =', data_size)

    # Создаем датасет и даталоадер
    dataset = DiffractionDataset(two_theta_range, step, num_samples=num_smp_per_run)

    total_generated = 0  # Общее количество сгенерированных спектров

    start_time = time.time()  # Запоминаем время начала обучения

    while total_generated < target_samples:
        # Генерируем пачку спектров и получаем количество фактически записанных
        generated_in_batch = dataset.generate_all_patterns()

        # Если были сгенерированы спектры - сохраняем их
        if generated_in_batch > 0:
            save_batch_inp_file(dataset.batch_inp)
            save_batch_out_file(dataset.batch_out)

            total_generated += generated_in_batch

            # Выводим информацию о прогрессе
            print(f'Generated {generated_in_batch} profiles in this batch')
            print(f'Total generated: {total_generated} / {target_samples}')

            if len(dataset.batch_out) > 0:
                print(f'Example output (sum + {num_nn_phases} normalized weights): {dataset.batch_out[0]}')
        else:
            print(f'No profiles generated in this batch (all had sum_first_three < 0.01)')

        # Выводим время выполнения
        elapsed_time = time.time() - start_time  # Вычисляем прошедшее время
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Прошло времени: {int(hours)}:{int(minutes):02d}:{int(seconds):02d}\n')

        # Защита от бесконечного цикла на случай, если почти все спектры отбрасываются
        if generated_in_batch == 0 and total_generated == 0:
            print("WARNING: No profiles generated at all. Check the condition sum_first_three < 0.01")
            break

    print(f'Generation completed! Total generated: {total_generated} profiles')


if __name__ == "__main__":
    main()