import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from diffraction_utils import initialize_structures, generate_combined_diffraction_pattern, load_experimental_data
from load_xml_phases import *

# Автовыбор вычислительного устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device =', device)

num_phases = 0
PHASES_INFO = []

xml_name = 'Ores_96ph_p1'
model_ver = 'v1_n001'
wghts_fname = f'weights3_{xml_name}_{model_ver}.pth'

experiment_folder = '.\\exp_ores_xy\\'

two_theta_range = (5.0, 90.000)  # in degrees
step = 0.020000517647  # in degrees

folder = r"./DB3/"

files = [
    os.path.join(folder, f)          # полный путь к файлу
    for f in os.listdir(folder)
    if f.lower().endswith('.tshx')
]
files_n = len(files)
if files_n == 0:
    exit(0)


def predict_experimental_data(model, experimental_data, input_size):
    predictions = []
    total_difference = 0.0
    exp_samples_nb = 0
    for filename, intensities in experimental_data.items():
        # two_theta_values = np.arange(two_theta_range[0], two_theta_range[1] + step, step)
        # start_index = np.where(two_theta_values > 5.8)[0][0]
        # interpolated_intensities = torch.tensor(intensities[start_index:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        interpolated_intensities = torch.tensor(intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            predicted_weights = model(interpolated_intensities)

        # Умножаем предсказанные веса на 100
        predicted_weights *= 100
        
        predictions.append((filename, predicted_weights.squeeze().cpu().numpy()))

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


class DiffractionNet(nn.Module):
    def __init__(self, input_size, num_phases):
        super(DiffractionNet, self).__init__()

        # Сверточные слои для извлечения локальных признаков
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Размер выхода после сверточных слоев
        reduced_size = input_size // 8  # Поскольку мы делаем три пулинга с шагом 2

        # Трансформер для глобального понимания
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.0, batch_first=True),
            num_layers=2
        )

        # Полносвязные слои для получения итогового предсказания
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * reduced_size, 1024),
            nn.ReLU(),
            # nn.Dropout(0.0),
            nn.Linear(1024, num_phases)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Изменяем форму для трансформера (batch_size, sequence_length, features)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)  # Флэттенинг с использованием reshape
        x = self.fc_layers(x)
        x = torch.sigmoid(x)  # Применяем sigmoid для получения значений между 0 и 1
        # Нормализация выхода, чтобы сумма всех значений была равна 1
        x = x / x.sum(dim=1, keepdim=True)
        return x

def main():
    global num_phases
    global PHASES_INFO

    # Загружаем фазы из файлов .phase
    sample_info, PHASES_INFO = parse_xml_multi('.\\PolusD8_Cu.tshx', files)
    num_phases = len(PHASES_INFO)

    # initialize_structures(PHASES_INFO, two_theta_range, step)

    # Вычисляем размер входа для свертки
    data_size = int((two_theta_range[1] - two_theta_range[0]) / step) + 1
    print('data_size =', data_size)
    data_size = 4211

    # Создаем модель
    model = DiffractionNet(data_size, num_phases).to(device)

    # Проверяем, существует ли файл с весами, и загружаем его, если существует
    if os.path.exists(wghts_fname):
        model.load_state_dict(torch.load(wghts_fname, map_location=device, weights_only=True))
        print(f"Модель загружена из файла {wghts_fname}")
    else:
        print(f'No weights file found. Starting training from scratch.')
        return 0

    # Загружаем экспериментальные данные
    experimental_data = load_experimental_data(experiment_folder, 5.8)
    print(f'Loaded {len(experimental_data)} experimental diffraction patterns.')

    # Прогоняем экспериментальные данные через модель
    # Предсказываем концентрации фаз
    model.eval()
    predictions, total_difference, num_samples = predict_experimental_data(model, experimental_data, data_size)

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