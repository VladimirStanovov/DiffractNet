import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import time
import csv
import glob

difr_dir = "D:\\nn_data\\"
os.makedirs(difr_dir, exist_ok=True)

# Автовыбор вычислительного устройства
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device =', device)

# Вычисляем размер входа для свертки
input_size = 4211
num_phases = 96

xml_name = f'Ores_{num_phases}ph_p1'
model_ver = 'v01'
wghts_fname = f'weights3_{xml_name}_{model_ver}.pth'

# Имя файла для логов CSV
csv_log_fname = f'log_{xml_name}_{model_ver}.csv'

inp_data = []
out_data = []

def load_batch_inp_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(struct.calcsize(f'{input_size}f'))
                if not data:
                    break
                inp = struct.unpack(f'{input_size}f', data)
                inp_data.append(np.array(inp))


def load_batch_out_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(struct.calcsize(f'{num_phases}f'))
                if not data:
                    break
                out = struct.unpack(f'{num_phases}f', data)
                out_data.append(np.array(out))

# Загрузка данных
inp_files = glob.glob(f"{difr_dir}batch_inp_Ores_{num_phases}ph_p*.bin")
out_files = glob.glob(f"{difr_dir}batch_out_Ores_{num_phases}ph_p*.bin")

for file in inp_files:
    load_batch_inp_file(file)

for file in out_files:
    load_batch_out_file(file)

print(f"Размер массива out_data: {len(out_data)}\n")

if len(out_data) < 1:
    print("Внимание - нет данных для обучения !")

lim_len = 500000
if len(inp_data) > lim_len:
    inp_data = inp_data[-lim_len:]
if len(out_data) > lim_len:
    out_data = out_data[-lim_len:]

# Определение класса для датасета
class CandleDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Возвращаем данные без изменения формы
        return torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.outputs[idx], dtype=torch.float32)

# Разделение данных на тренировочную и тестовую выборки
indices = list(range(len(inp_data)))
test_indices = indices[::20]
train_indices = [i for i in indices if i not in test_indices]

train_inputs = [inp_data[i] for i in train_indices]
train_outputs = [out_data[i] for i in train_indices]
test_inputs = [inp_data[i] for i in test_indices]
test_outputs = [out_data[i] for i in test_indices]

train_dataset = CandleDataset(train_inputs, train_outputs)
test_dataset = CandleDataset(test_inputs, test_outputs)

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

# Создание экземпляра модели
model = DiffractionNet(input_size, num_phases).to(device)

# Проверяем, существует ли файл с весами, и загружаем его, если существует
if os.path.exists(wghts_fname):
    model.load_state_dict(torch.load(wghts_fname, map_location=device, weights_only=True))
    print(f"Модель загружена из файла {wghts_fname}")
else:
    print(f'No weights file found. Starting training from scratch.')

print('Start training')

learn_rate = 2E-5
    
# Определяем критерий потерь и оптимизатор
criterion = nn.MSELoss(reduction='none')  # Используем reduction='none' для получения ошибки для каждого элемента в батче
optimizer = optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=learn_rate*0.1)

def find_index_of_max(float_array):
    max_index = np.argmax(float_array)
    return max_index

# Определяем категории для балансировки данных
def get_category(out):
    min_val = np.min(out)
    max_val = np.max(out)
    delta = min_val / max_val if max_val > 0 else 0.0
    if delta < 0.001:
        return 0
    elif 0.001 <= delta < 0.01:
        return 1
    elif 0.01 <= delta < 0.1:
        return 2
    else:
        return 3

# Определяем метки классов для балансировки
train_labels = np.array([find_index_of_max(out) for out in train_outputs])
class_counts = np.bincount(train_labels)
print(f'Category class_counts = {class_counts}')
# Добавляем сглаживание для предотвращения деления на ноль
#smoothing_factor = 1e-3
#class_weights = 1. / (class_counts + smoothing_factor)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Загрузка данных
train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

n_train_iter = len(train_dataloader)
n_test_iter = len(test_dataloader)
print(f'train_iter: {n_train_iter}, test_iter: {n_test_iter}')

# Инициализация списков для хранения значений потерь
train_losses = []
test_losses = []

# Инициализация списков для хранения значений потерь для каждого параметра
train_param_losses = [[] for _ in range(num_phases)]
test_param_losses = [[] for _ in range(num_phases)]

# Открываем CSV файл для записи и записываем заголовок
fieldnames = ['epoch', 'learn_rate', 'train_loss', 'test_loss'] + \
             [f'train_phase_{i+1}' for i in range(num_phases)] + \
             [f'test_phase_{i+1}' for i in range(num_phases)] + \
             ['elapsed_time']
with open(csv_log_fname, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()

num_epochs = 300
pre_fitness = 1E6

start_time = time.time()  # Запоминаем время начала обучения

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    running_train_param_losses = [0.0 for _ in range(num_phases)]
    iters = 0
    for inputs, outputs in train_dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, outputs)
        loss.mean().backward()  # Средняя ошибка по всем элементам в батче
        # Градиенты масштабируются, если их общая норма > 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_train_loss += loss.mean().item()
        for i in range(num_phases):
            running_train_param_losses[i] += loss[:, i].mean().item()
        iters += 1

        if iters % 500 == 0:
            print(f"Iteration {iters}")

    train_loss = running_train_loss / n_train_iter
    train_losses.append(train_loss)
    for i in range(num_phases):
        train_param_losses[i].append(running_train_param_losses[i] / n_train_iter)

    if n_test_iter > 0:
        model.eval()
        running_test_loss = 0.0
        running_test_param_losses = [0.0 for _ in range(num_phases)]
        with torch.no_grad():
            for inputs, outputs in test_dataloader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, outputs)
                running_test_loss += loss.mean().item()
                for i in range(num_phases):
                    running_test_param_losses[i] += loss[:, i].mean().item()

        test_loss = running_test_loss / n_test_iter
        test_losses.append(test_loss)
        for i in range(num_phases):
            test_param_losses[i].append(running_test_param_losses[i] / n_test_iter)

        if test_loss < pre_fitness:
            torch.save(model.state_dict(), wghts_fname)
            print(f'*** Weights are saved to {wghts_fname} after epoch {epoch + 1}')
            pre_fitness = test_loss

    else:
        test_loss = 0.0
        test_losses.append(test_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], learn_rate: {learn_rate}, Train Loss: {100*np.sqrt(train_loss):.5f}, Test Loss: {100*np.sqrt(test_loss):.5f}')

    # Вывод ошибок для каждого параметра
    for i in range(num_phases):
        print(f'Фаза {i+1}: Train: {100*np.sqrt(train_param_losses[i][-1]):.5f}, Test: {100*np.sqrt(test_param_losses[i][-1]):.5f}')

    # Вычисляем и выводим прошедшее время
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Прошло времени: {int(hours)}:{int(minutes)}:{int(seconds)}\n')

    # Подготовка строки для CSV
    row_dict = {
        'epoch': epoch + 1,
        'learn_rate': learn_rate,
        'train_loss': 100*np.sqrt(train_loss),
        'test_loss': 100*np.sqrt(test_loss)
    }
    for i in range(num_phases):
         row_dict[f'train_phase_{i+1}'] = 100*np.sqrt(train_param_losses[i][-1])
         row_dict[f'test_phase_{i+1}'] = 100*np.sqrt(test_param_losses[i][-1])

    # Добавляем время в CSV
    with open(csv_log_fname, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        row_dict['elapsed_time'] = f'{int(hours)}:{int(minutes)}:{int(seconds)}'
        writer.writerow(row_dict)