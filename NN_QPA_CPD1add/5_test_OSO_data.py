import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from diffraction_utils import generate_combined_diffraction_pattern, load_experimental_data
import concurrent.futures
from load_xml_phases import *

# Автовыбор вычислительного устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device =', device)

num_phases = 0
PHASES_INFO = []

xml_name = 'CPD-1'
wghts_fname = 'weights3_' + xml_name + '.pth'
experiment_folder = '.\\CPD1_XY\\'

two_theta_range = (15, 95)  # in degrees
step = 0.02  # in degrees

# Создаем словарь для сопоставления имен файлов и целевых значений
target_values = {       # Al2O3, CaF2, ZnO
    's1a_r1.xy': np.array([ 1.15, 94.81,  4.04]),
    's1b_r1.xy': np.array([94.31,  4.34,  1.36]),
    's1c_r1.xy': np.array([ 5.04,  1.36, 93.59]),
    's1d_r1.xy': np.array([13.53, 53.58, 32.89]),
    's1e_r1.xy': np.array([55.12, 29.62, 15.25]),
    's1f_r1.xy': np.array([27.06, 17.72, 55.22]),
    's1g_r1.xy': np.array([31.37, 34.42, 34.21]),
    's1h_r1.xy': np.array([35.12, 34.69, 30.19]),
}

class DiffractionNet(nn.Module):
    def __init__(self, input_size, num_phases):
        super(DiffractionNet, self).__init__()

        # Первый сверточный слой
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=36, kernel_size=20, stride=1, padding=9)
        self.relu = nn.ReLU()

        # Макс-пуллинг
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        # Второй сверточный слой
        self.conv2 = nn.Conv1d(in_channels=36, out_channels=72, kernel_size=5, stride=1, padding=2)

        # Вычисляем размер выходных данных после второй свертки
        # После первой свертки: (input_size - 20 + 2*9) / 1 + 1 = input_size
        # После maxpool: input_size / 4
        # После второй свертки: (input_size / 4 - 5 + 2*2) / 1 + 1 = input_size / 4
        conv_output_size = input_size // 4

        # Трансформер
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # Линейный слой
        self.fc_input_size = 72 * conv_output_size  # Размер после второй свертки
        self.fc = nn.Linear(self.fc_input_size, num_phases)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        # Трансформер
        x = self.transformer_encoder(x)

        # Линейный слой
        x = x.reshape(x.size(0), -1)  # Используем reshape вместо view
        x = self.fc(x)
        x = torch.sigmoid(x)  # Применяем sigmoid для получения значений между 0 и 1

        # Нормализация выхода, чтобы сумма всех значений была равна 1
        x = x / x.sum(dim=1, keepdim=True)

        return x



def predict_experimental_data(model, experimental_data, input_size):
    predictions = []
    total_difference = 0.0
    exp_samples_nb = 0
    for filename, intensities in experimental_data.items():

        interpolated_intensities = torch.tensor(intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        #interpolated_intensities = torch.tensor(interpolated_intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            predicted_weights = model(interpolated_intensities)

        # Умножаем предсказанные веса на 100
        predicted_weights *= 100
        
        predictions.append((filename, predicted_weights.squeeze().cpu().numpy()))

        # Сравнение с целевыми значениями, если они есть
        if filename in target_values:
            target = target_values[filename]
            difference = np.abs(predicted_weights.squeeze().cpu().numpy() - target).sum()
            total_difference += difference
            num_samples += 1

    return predictions, total_difference, exp_samples_nb


def save_predictions_to_csv(predictions, output_filename, PHASES_INFO):
    # Извлечение названий фаз
    phase_names = [phase_info['Name'] for phase_info in PHASES_INFO]
    
    # Создание списка заголовков
    headers = ['Filename'] + phase_names
    
    # Разбиение предсказанных весов на отдельные столбцы
    formatted_predictions = [(filename, *weights) for filename, weights in predictions]
    
    # Создание DataFrame
    df = pd.DataFrame(formatted_predictions, columns=headers)
    
    # Сохранение DataFrame в CSV файл с кодировкой UTF-8
    df.to_csv(output_filename, index=False, sep=';', header=True, encoding='utf-8-sig')
    print(f'Predictions saved to {output_filename}')



def main():
    global num_phases
    global PHASES_INFO

    # Загружаем фазы из файлов .phase
    PHASES_INFO = parse_xml(xml_name+'.tshx')
    num_phases = len(PHASES_INFO)


    # Вычисляем размер входа для свертки
    input_size = int((two_theta_range[1] - two_theta_range[0]) / step) + 1
    print('input_size =', input_size)

    # Создаем модель
    model = DiffractionNet(input_size, num_phases).to(device)

    # Проверяем, существует ли файл с весами, и загружаем его, если существует
    if os.path.exists(wghts_fname):
        model.load_state_dict(torch.load(wghts_fname, map_location=device, weights_only=True))
        print(f"Модель загружена из файла {wghts_fname}")
    else:
        print(f'No weights file found. Starting training from scratch.')
        return 0


    # Загружаем экспериментальные данные
    experimental_data = load_experimental_data(experiment_folder)
    print(f'Loaded {len(experimental_data)} experimental diffraction patterns.')

    # Прогоняем экспериментальные данные через модель
    # Предсказываем концентрации фаз
    model.eval()
    predictions, total_difference, num_samples = predict_experimental_data(model, experimental_data, input_size)

    # Сохраняем результаты в CSV файл
    output_filename = 'predictions.csv'
    save_predictions_to_csv(predictions, output_filename, PHASES_INFO)

    # Вывод средней разницы, если были найдены соответствия
    if num_samples > 0:
        average_difference = (total_difference / num_samples) / num_phases
        print(f'\nОбразцов для посчёта разницы: {num_samples}')
        print(f'Средняя разница между предсказанными и OSO на фазу: {average_difference:.3f}')
    else:
        print('Нет соответствий для подсчета разницы.')


if __name__ == "__main__":
    main()