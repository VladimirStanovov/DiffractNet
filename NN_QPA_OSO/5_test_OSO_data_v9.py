import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from diffraction_utils import initialize_structure, generate_combined_diffraction_pattern, load_experimental_data
from load_xml_phases import *

# Автовыбор вычислительного устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device =', device)

num_phases = 0
PHASES_INFO = []

xml_name = 'OSO_NN_midKO_11ph_noized001'
model_ver = 'v9exp1'
wghts_fname = f'weights3_{xml_name}_{model_ver}.pth'

experiment_folder = '.\\XY_OSO_midKO\\'

two_theta_range = (10, 90)  # in degrees
step = 0.02  # in degrees

# Создаем словарь для сопоставления имен файлов и целевых значений
target_values = {
    'Бо1125_180121.xy': np.array([0.42, 0.25, 42.72, 44.65, 2.22, 8.05, 0.00, 1.40, 0.28, 0.00, 0.01]),
    'Бо2074_200121.xy': np.array([0.50, 0.24, 58.48, 27.85, 2.88, 7.90, 0.05, 1.62, 0.48, 0.00, 0.01]),
    'Бо2077_200121.xy': np.array([0.40, 0.93, 67.24, 18.25, 3.63, 7.51, 0.03, 1.60, 0.42, 0.00, 0.00]),
    'Бр1026_181220.xy': np.array([2.72, 2.78, 47.84, 28.73, 1.20, 14.57, 0.07, 2.06, 0.03, 0.00, 0.00]),
    'Бр1071_241220.xy': np.array([1.22, 0.63, 50.30, 30.39, 1.60, 13.97, 0.06, 1.84, 0.00, 0.00, 0.00]),
    'Бр1307_2_181220.xy': np.array([0.68, 1.17, 86.07, 0.94, 3.65, 1.29, 4.55, 0.31, 0.79, 0.05, 0.52]),
    'Бр1356_251220.xy': np.array([1.67, 2.21, 78.72, 2.34, 7.51, 3.22, 2.43, 1.09, 0.51, 0.28, 0.02]),
    'Бр1425_241220.xy': np.array([0.90, 1.56, 81.95, 1.43, 6.32, 2.76, 3.40, 0.85, 0.68, 0.16, 0.00]),
    'Бр1428_241220.xy': np.array([1.07, 2.47, 63.14, 15.82, 0.80, 14.47, 0.66, 1.58, 0.00, 0.00, 0.00]),
    'Бр1566_281220.xy': np.array([0.72, 3.95, 70.88, 5.65, 6.65, 9.87, 0.12, 2.14, 0.00, 0.00, 0.01]),
    'В506_140121.xy': np.array([2.11, 2.59, 54.29, 21.35, 2.26, 15.51, 0.06, 1.59, 0.24, 0.00, 0.01]),
    'В508_140121.xy': np.array([2.12, 3.48, 48.43, 27.17, 1.62, 15.24, 0.05, 1.63, 0.26, 0.00, 0.01]),
    'В549_150121.xy': np.array([1.01, 3.74, 78.07, 0.43, 13.37, 0.44, 1.63, 0.24, 0.17, 0.00, 0.89]),
    'В574_150121.xy': np.array([0.52, 1.16, 52.24, 24.86, 1.70, 16.05, 0.03, 3.32, 0.11, 0.00, 0.00]),
    'В583_180121.xy': np.array([0.45, 1.25, 52.71, 24.86, 2.10, 16.14, 0.06, 2.38, 0.04, 0.00, 0.01]),
    'И103_291220.xy': np.array([1.02, 3.74, 53.23, 30.05, 1.25, 5.35, 4.20, 1.05, 0.00, 0.11, 0.00]),
    'И128_120121.xy': np.array([1.16, 3.16, 59.49, 23.44, 1.06, 7.46, 2.84, 1.33, 0.00, 0.06, 0.00]),
    'И538_110121.xy': np.array([1.42, 0.97, 72.78, 9.48, 2.78, 10.23, 0.42, 1.19, 0.00, 0.72, 0.00]),
    'И644_110121.xy': np.array([1.24, 4.00, 73.64, 6.08, 7.94, 4.33, 1.11, 0.93, 0.00, 0.63, 0.00]),
    'И743_211220.xy': np.array([0.39, 1.37, 88.97, 0.92, 2.73, 0.32, 4.03, 0.12, 0.42, 0.28, 0.44]),
    'И9085_130121.xy': np.array([0.48, 0.14, 42.95, 44.21, 1.04, 10.34, 0.07, 0.71, 0.06, 0.00, 0.00]),
    'К173_210121.xy': np.array([0.68, 2.13, 45.16, 36.51, 1.06, 10.94, 0.07, 2.73, 0.29, 0.43, 0.00]),
    'К1898_270121.xy': np.array([1.89, 1.83, 55.79, 17.94, 2.05, 13.11, 0.43, 2.62, 0.14, 4.21, 0.00]),
    'К2019_270121.xy': np.array([1.30, 3.43, 67.70, 7.81, 2.77, 10.36, 0.05, 2.19, 0.40, 4.00, 0.00]),
    'К2022_280121.xy': np.array([1.18, 1.49, 58.94, 15.68, 2.12, 13.57, 0.08, 2.58, 0.46, 3.91, 0.00]),
    'К2035_280121.xy': np.array([0.99, 1.88, 57.56, 16.03, 2.68, 12.59, 0.11, 2.36, 0.43, 5.37, 0.00]),
    'К2041_280121.xy': np.array([0.95, 2.44, 49.16, 25.80, 1.25, 13.73, 0.07, 2.76, 0.22, 3.62, 0.00]),
    'К2042_290121.xy': np.array([0.87, 0.92, 83.39, 0.71, 0.34, 0.11, 6.30, 0.00, 0.24, 4.62, 2.51]),
    'С152_161220.xy': np.array([0.26, 1.65, 33.93, 52.38, 0.73, 9.34, 0.27, 0.99, 0.04, 0.38, 0.04]),
    'С207_181220.xy': np.array([0.32, 0.61, 42.08, 44.57, 0.83, 9.56, 0.29, 1.31, 0.00, 0.42, 0.00]),
    'С294_211220.xy': np.array([0.77, 2.01, 57.04, 26.08, 2.25, 8.96, 0.34, 1.74, 0.00, 0.83, 0.00]),
    'С343_221220.xy': np.array([1.84, 1.80, 51.13, 33.75, 1.11, 7.74, 0.75, 1.31, 0.00, 0.56, 0.00]),
    'С367_231220.xy': np.array([0.47, 1.94, 56.72, 26.99, 1.58, 9.51, 0.39, 1.60, 0.00, 0.81, 0.00]),
    'С455_171220.xy': np.array([0.20, 2.09, 32.42, 53.03, 0.80, 9.33, 0.46, 1.01, 0.00, 0.28, 0.38]),
    'С479_231220.xy': np.array([0.73, 1.32, 53.55, 31.52, 1.41, 9.09, 0.29, 1.45, 0.00, 0.63, 0.00]),
    'С490_231220.xy': np.array([1.05, 2.49, 55.01, 30.80, 1.07, 4.45, 3.10, 1.48, 0.01, 0.55, 0.00]),
    'дБо2058д_240521.xy': np.array([0.00, 0.74, 10.33, 59.61, 2.16, 5.09, 0.13, 21.81, 0.15, 0.00, 0.00]),
    'дБо2082д_280521.xy': np.array([0.62, 0.35, 85.91, 0.00, 0.00, 0.00, 3.22, 0.00, 9.42, 0.06, 0.17]),
    'дБр1307д_220421.xy': np.array([0.73, 0.45, 86.21, 0.00, 0.00, 0.00, 9.47, 0.00, 0.69, 0.00, 0.99]),
    'дБр1552д_020621.xy': np.array([0.69, 0.08, 85.75, 0.00, 0.00, 0.00, 6.05, 0.00, 1.05, 0.00, 0.21]),
    'дК1132д_180521.xy': np.array([2.49, 1.68, 75.77, 0.00, 0.00, 0.00, 9.79, 0.00, 5.89, 0.22, 1.78]),
    'дК1585д_170521.xy': np.array([0.43, 1.17, 84.69, 0.00, 0.00, 0.00, 9.71, 0.00, 1.83, 1.65, 0.46]),
    'дК180д_290421.xy': np.array([0.19, 1.93, 26.94, 42.17, 4.99, 17.44, 0.06, 5.90, 0.00, 0.40, 0.00]),
    'дС132д_210421.xy': np.array([0.29, 2.38, 26.13, 41.49, 2.17, 16.56, 0.10, 10.58, 0.27, 0.03, 0.00]),
    'дС321д_120521.xy': np.array([0.42, 1.90, 20.31, 38.45, 3.73, 20.36, 0.10, 14.56, 0.00, 0.02, 0.16]),
    'дС323д_170521.xy': np.array([0.17, 1.65, 21.48, 30.25, 7.69, 19.54, 0.03, 18.75, 0.14, 0.07, 0.24])
}

def predict_experimental_data(model, experimental_data, input_size):
    predictions = []
    total_difference = 0.0
    exp_samples_nb = 0
    for filename, intensities in experimental_data.items():

        interpolated_intensities = torch.tensor(intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
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
            exp_samples_nb += 1

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
            #nn.Dropout(0.0),
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
    sample_info,  PHASES_INFO = parse_xml(xml_name+'.tshx')
    num_phases = len(PHASES_INFO)

    initialize_structure(PHASES_INFO, two_theta_range, step)


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