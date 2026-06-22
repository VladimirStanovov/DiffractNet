import os
import numpy as np
import torch
from torch.utils.data import Dataset
from diffraction_utils import initialize_structures, generate_combined_diffraction_pattern, erase_lib_memory, generate_ph_fractions
import random
from typing import List
import concurrent.futures
from load_xml_phases import *
import time

import struct

difr_dir = "D:\\nn_data\\"
os.makedirs(difr_dir, exist_ok=True)

# Автовыбор вычислительного устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('device =', device)

input_size = 4001
num_phases = 3

xml_name = f'CPD1_{num_phases}ph_p2'
model_ver = 'v10'

num_smp_per_run = 100

PHASES_INFO = []
sample_info = []

two_theta_range = (15, 95)  # in degrees
step = 0.02  # in degrees

folder = r"../DB3/"
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
    def __init__(self, _two_theta_range, _step, num_samples=500):
        self.two_theta_range = _two_theta_range
        self.step = _step
        self.num_samples = num_samples
        self.num_phases = len(PHASES_INFO)
        self.batch_inp = []
        self.batch_out = []
        #self.generate_all_patterns()  # Генерируем все дифрактограммы при инициализации

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        combined_intensity_values, weights = self.batch_inp[idx], self.batch_out[idx]
        combined_intensity_values = torch.tensor(combined_intensity_values).unsqueeze(0)  # Добавляем канал
        weights = torch.tensor(weights, dtype=torch.float32)
        return combined_intensity_values, weights

    def generate_all_patterns(self):
        self.batch_inp = []
        self.batch_out = []

        for iter in range(0, self.num_samples):
            arg_wghts = generate_ph_fractions(len(PHASES_INFO))
            # 1. Индексы ненулевых весов
            nonzero_indices = [i for i, w in enumerate(arg_wghts) if w > 0]
            # 2. Сокращаем оба списка
            curr_wghts = [arg_wghts[i] for i in nonzero_indices]
            CURR_PHASES = [PHASES_INFO[i] for i in nonzero_indices]
            # print(arg_wghts)
            # print(CURR_PHASES)

            initialize_structures(CURR_PHASES, two_theta_range, step)

            _, combined_intensity_values, weights = generate_combined_diffraction_pattern(sample_info, CURR_PHASES, self.two_theta_range, self.step, curr_wghts)
            self.batch_inp.append(np.array(combined_intensity_values, dtype=np.float32))
            self.batch_out.append(np.array(arg_wghts))

            erase_lib_memory()



def save_batch_inp_file(arg_inps):
    with open(f'{difr_dir}batch_inp_{xml_name}.bin', 'ab') as f:
        for inp in arg_inps:
            f.write(struct.pack(f'{input_size}f', *inp))


def save_batch_out_file(arg_outs):
    with open(f'{difr_dir}batch_out_{xml_name}.bin', 'ab') as f:
        for out in arg_outs:
            f.write(struct.pack(f'{num_phases}f', *out))


def main():
    # Вычисляем размер входа для свертки
    data_size = int((two_theta_range[1] - two_theta_range[0]) / step) + 1
    print('data_size =', data_size)

    # Создаем датасет и даталоадер
    dataset = DiffractionDataset(two_theta_range, step, num_samples=num_smp_per_run)

    i = 0 / num_smp_per_run

    start_time = time.time()  # Запоминаем время начала обучения

    while i*num_smp_per_run < 25000:
        dataset.generate_all_patterns()  # Перегенерируем все дифрактограммы

        save_batch_inp_file(dataset.batch_inp)
        save_batch_out_file(dataset.batch_out)

        print(dataset.batch_out[0])

        i += 1
        print('Generated profiles: ', i*num_smp_per_run)

        elapsed_time = time.time() - start_time  # Вычисляем прошедшее время
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'Прошло времени: {int(hours)}:{int(minutes)}:{int(seconds)}\n')



if __name__ == "__main__":
    main()