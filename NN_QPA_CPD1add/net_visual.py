import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from graphviz import Digraph
import os
from collections import OrderedDict
import json

num_phases = 4

# Копия класса DiffractionNet из вашего кода
class DiffractionNet(nn.Module):
    def __init__(self, input_size, num_phases):
        super(DiffractionNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        reduced_size = input_size // 8

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True),
            num_layers=2
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * reduced_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_phases)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        total_sum = torch.sigmoid(x[:, 0:1])
        normalized = F.softmax(x[:, 1:], dim=1)
        output = torch.cat([total_sum, normalized], dim=1)
        return output


def visualize_architecture():
    """Главная функция для визуализации архитектуры"""

    # Параметры модели
    input_size = 4001
    num_phases = 4

    # Создаем модель
    model = DiffractionNet(input_size, num_phases)

    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ АРХИТЕКТУРЫ DIFFRACTIONNET")
    print("=" * 80)
    print(f"Размер входа: {input_size}")
    print(f"Количество фаз: {num_phases}")
    print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Обучаемые параметры: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 80)

    # Создаем папку для результатов
    os.makedirs("visualization_results", exist_ok=True)

    # Вариант 1: Детальное текстовое описание
    print("\n1. СОЗДАНИЕ ТЕКСТОВОГО ОПИСАНИЯ...")
    create_text_description(model, input_size)

    # Вариант 2: Визуализация графа вычислений с torchviz
    print("\n2. СОЗДАНИЕ ГРАФА ВЫЧИСЛЕНИЙ (TORCHVIZ)...")
    create_torchviz_graph(model, input_size)

    # Вариант 3: Визуализация через TensorBoard
    print("\n3. СОЗДАНИЕ ВИЗУАЛИЗАЦИИ ДЛЯ TENSORBOARD...")
    create_tensorboard_graph(model, input_size)

    # Вариант 4: Схематичная визуализация с matplotlib
    print("\n4. СОЗДАНИЕ СХЕМАТИЧНОЙ ДИАГРАММЫ (MATPLOTLIB)...")
    create_matplotlib_diagram(model, input_size)

    # Вариант 5: Иерархическое дерево слоев
    print("\n5. СОЗДАНИЕ ИЕРАРХИЧЕСКОГО ДЕРЕВА СЛОЕВ...")
    create_layer_hierarchy(model)

    # Вариант 6: Блок-схема архитектуры
    print("\n6. СОЗДАНИЕ БЛОК-СХЕМЫ АРХИТЕКТУРЫ...")
    create_block_diagram(model, input_size)

    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("Результаты сохранены в папке 'visualization_results/'")
    print("=" * 80)


def create_text_description(model, input_size):
    """Создание подробного текстового описания архитектуры"""

    output = []
    output.append("=" * 80)
    output.append("ТЕКСТОВОЕ ОПИСАНИЕ АРХИТЕКТУРЫ DIFFRACTIONNET")
    output.append("=" * 80)

    # Общая информация
    output.append(f"\nОБЩАЯ ИНФОРМАЦИЯ:")
    output.append(f"  • Размер входа: (batch, 1, {input_size})")
    output.append(f"  • Выход: (batch, {model.fc_layers[-1].out_features})")

    # Подсчет параметров по слоям
    output.append(f"\nРАСПРЕДЕЛЕНИЕ ПАРАМЕТРОВ:")

    total_params = 0
    layer_info = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.split('.')[0] if '.' in name else name
            params_count = param.numel()
            total_params += params_count
            layer_info.append(f"  • {name}: {params_count:,} параметров")

    for line in layer_info:
        output.append(line)

    output.append(f"\n  Всего параметров: {total_params:,}")

    # Подробное описание каждого слоя
    output.append(f"\nДЕТАЛЬНАЯ СТРУКТУРА СЛОЕВ:")

    # Сверточные слои
    output.append(f"\n1. СВЕРТОЧНЫЕ СЛОИ (conv_layers):")

    conv_layers = model.conv_layers
    current_size = input_size

    for i, layer in enumerate(conv_layers):
        if isinstance(layer, nn.Conv1d):
            output.append(f"   • Conv1d-{i // 3 + 1}:")
            output.append(f"     - Вход: (batch, {layer.in_channels}, {current_size})")
            output.append(f"     - Фильтры: {layer.out_channels}")
            output.append(f"     - Ядро: {layer.kernel_size[0]}")
            output.append(f"     - Сдвиг: {layer.stride[0]}")
            output.append(f"     - Паддинг: {layer.padding[0]}")
            output.append(f"     - Параметры: {sum(p.numel() for p in layer.parameters()):,}")
            current_size = current_size // layer.stride[0] if layer.stride[0] > 1 else current_size

        elif isinstance(layer, nn.MaxPool1d):
            output.append(f"   • MaxPool1d-{(i - 2) // 3 + 1}:")
            output.append(f"     - Ядро: {layer.kernel_size}")
            output.append(f"     - Сдвиг: {layer.stride}")
            current_size = current_size // layer.stride

    # Трансформер
    output.append(f"\n2. ТРАНСФОРМЕРНЫЙ ЭНКОДЕР (transformer):")
    transformer = model.transformer
    layer = transformer.layers[0]
    output.append(f"   • Количество слоев: {len(transformer.layers)}")
    output.append(f"   • Размер модели: {layer.self_attn.embed_dim}")
    output.append(f"   • Количество голов: {layer.self_attn.num_heads}")
    output.append(f"   • Размер FFN: {layer.linear1.out_features}")
    output.append(f"   • Вход/выход: (batch, {current_size}, 256)")

    # Полносвязные слои
    output.append(f"\n3. ПОЛНОСВЯЗНЫЕ СЛОИ (fc_layers):")
    for i, layer in enumerate(model.fc_layers):
        if isinstance(layer, nn.Linear):
            output.append(f"   • Linear-{i // 3 + 1}:")
            output.append(f"     - Входные нейроны: {layer.in_features}")
            output.append(f"     - Выходные нейроны: {layer.out_features}")
            output.append(f"     - Параметры: {sum(p.numel() for p in layer.parameters()):,}")

    # Форвард проход
    output.append(f"\n4. ПРОЦЕСС ФОРВАРД ПРОХОДА:")
    output.append(f"   1. Вход: (batch, 1, {input_size})")
    output.append(f"   2. Conv слои → (batch, 256, {input_size // 8})")
    output.append(f"   3. Permute → (batch, {input_size // 8}, 256)")
    output.append(f"   4. Transformer → (batch, {input_size // 8}, 256)")
    output.append(f"   5. Reshape → (batch, {256 * (input_size // 8)})")
    output.append(f"   6. FC слои → (batch, {num_phases})")
    output.append(f"   7. Sigmoid + Softmax → (batch, {num_phases})")

    # Сохраняем в файл
    with open("visualization_results/architecture_description.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    # Выводим в консоль
    print("\n".join(output[:50]))  # Первые 50 строк
    print("... (полное описание сохранено в файл)")

    return output


def create_torchviz_graph(model, input_size):
    """Создание графа вычислений с помощью torchviz"""

    try:
        # Создаем фиктивный вход
        x = torch.randn(1, 1, input_size)

        # Прямой проход
        y = model(x)

        # Создаем граф
        dot = make_dot(y, params=dict(model.named_parameters()),
                       show_attrs=True,
                       show_saved=True)

        # Настройка внешнего вида
        dot.format = 'png'
        dot.attr('graph', rankdir='TB', size='20,20')  # TB - сверху вниз
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

        # Сохраняем
        dot.render("visualization_results/computation_graph", cleanup=True)
        print("Граф вычислений сохранен как: visualization_results/computation_graph.png")

    except Exception as e:
        print(f"Ошибка при создании графа torchviz: {e}")
        print("Убедитесь, что установлены graphviz и torchviz")


def create_tensorboard_graph(model, input_size):
    """Создание визуализации для TensorBoard"""

    try:
        writer = SummaryWriter('visualization_results/tensorboard_logs')

        # Создаем фиктивный вход
        x = torch.randn(1, 1, input_size)

        # Добавляем граф в TensorBoard
        writer.add_graph(model, x)
        writer.close()

        print("Граф TensorBoard создан. Запустите команду:")
        print("tensorboard --logdir=visualization_results/tensorboard_logs")
        print("Затем откройте http://localhost:6006 в браузере")

    except Exception as e:
        print(f"Ошибка при создании графа TensorBoard: {e}")


def create_matplotlib_diagram(model, input_size):
    """Создание схематичной диаграммы с matplotlib"""

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Архитектура DiffractionNet', fontsize=16, fontweight='bold')

        # 1. Общая схема
        ax1 = axes[0, 0]
        layers = [
            ("Input", 1, f"(1, {input_size})"),
            ("Conv1D\n(64, k=7)", 64, f"(64, {input_size})"),
            ("BN + ReLU", 64, f"(64, {input_size})"),
            ("MaxPool\n(stride=2)", 64, f"(64, {input_size // 2})"),
            ("Conv1D\n(128, k=5)", 128, f"(128, {input_size // 2})"),
            ("BN + ReLU", 128, f"(128, {input_size // 2})"),
            ("MaxPool\n(stride=2)", 128, f"(128, {input_size // 4})"),
            ("Conv1D\n(256, k=3)", 256, f"(256, {input_size // 4})"),
            ("BN + ReLU", 256, f"(256, {input_size // 4})"),
            ("MaxPool\n(stride=2)", 256, f"(256, {input_size // 8})"),
            ("Transformer\n(2 layers)", 256, f"({input_size // 8}, 256)"),
            ("Flatten", 256 * (input_size // 8), f"({256 * (input_size // 8)})"),
            ("FC\n(1024)", 1024, "(1024)"),
            ("FC Output", 4, "(4)"),
            ("Sigmoid\n+ Softmax", 4, "(4)")
        ]

        x_pos = range(len(layers))
        heights = [layer[1] for layer in layers]

        # Создаем цветовую карту для слоев
        colors = []
        for i, layer in enumerate(layers):
            layer_name = layer[0].lower()
            if 'input' in layer_name:
                colors.append('#FFE4B5')  # Мокасиновый для входа
            elif 'conv' in layer_name:
                colors.append('#ADD8E6')  # Светло-голубой для сверток
            elif 'pool' in layer_name:
                colors.append('#90EE90')  # Светло-зеленый для пулинга
            elif 'bn' in layer_name or 'norm' in layer_name:
                colors.append('#FFB6C1')  # Светло-розовый для норм.
            elif 'relu' in layer_name:
                colors.append('#DDA0DD')  # Сливовый для активаций
            elif 'transformer' in layer_name:
                colors.append('#F0E68C')  # Хаки для трансформера
            elif 'flatten' in layer_name:
                colors.append('#87CEEB')  # Небесно-голубой
            elif 'fc' in layer_name:
                colors.append('#87CEEB')  # Небесно-голубой для FC
            elif 'output' in layer_name:
                colors.append('#98FB98')  # Бледно-зеленый для выхода
            elif 'sigmoid' in layer_name or 'softmax' in layer_name:
                colors.append('#DDA0DD')  # Сливовый для активаций
            else:
                colors.append('#D3D3D3')  # Светло-серый по умолчанию

        bars = ax1.bar(x_pos, heights, color=colors)
        ax1.set_xlabel('Слои')
        ax1.set_ylabel('Количество каналов/нейронов')
        ax1.set_title('Общая схема архитектуры')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([layer[0] for layer in layers], rotation=45, ha='right', fontsize=8)

        # Добавляем аннотации с размерами
        for i, (bar, layer) in enumerate(zip(bars, layers)):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., height + max(heights) * 0.01,
                         layer[2], ha='center', va='bottom', fontsize=7, rotation=90)

        # 2. Распределение параметров
        ax2 = axes[0, 1]
        param_counts = []
        layer_names = []

        # Собираем параметры по основным модулям
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_counts.append(params)
                # Укорачиваем имя для лучшего отображения
                short_name = name[:15] + '...' if len(name) > 15 else name
                layer_names.append(short_name)

        if param_counts:
            colors = plt.cm.Set3(np.linspace(0, 1, len(param_counts)))
            ax2.pie(param_counts, labels=layer_names, autopct='%1.1f%%', colors=colors,
                    startangle=90, counterclock=False)
            ax2.set_title('Распределение параметров по модулям')
        else:
            ax2.text(0.5, 0.5, 'Нет параметров', ha='center', va='center')
            ax2.set_title('Распределение параметров')

        # 3. Размерности тензоров
        ax3 = axes[1, 0]
        tensor_shapes = [
            f"Вход: (batch, 1, {input_size})",
            f"После Conv1: (batch, 64, {input_size})",
            f"После Pool1: (batch, 64, {input_size // 2})",
            f"После Conv2: (batch, 128, {input_size // 2})",
            f"После Pool2: (batch, 128, {input_size // 4})",
            f"После Conv3: (batch, 256, {input_size // 4})",
            f"После Pool3: (batch, 256, {input_size // 8})",
            f"После Permute: (batch, {input_size // 8}, 256)",
            f"После Transformer: (batch, {input_size // 8}, 256)",
            f"После Reshape: (batch, {256 * (input_size // 8)})",
            f"После FC1: (batch, 1024)",
            f"После FC2: (batch, 4)",
            f"Выход: (batch, 4)"
        ]

        y_pos = range(len(tensor_shapes))
        ax3.barh(y_pos, [1] * len(tensor_shapes), color='skyblue', height=0.6)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(tensor_shapes, fontsize=8)
        ax3.set_xlabel('Шаг обработки')
        ax3.set_title('Преобразование размерностей тензоров')
        ax3.invert_yaxis()
        ax3.set_xlim(0, 1.2)

        # 4. Информация о трансформере
        ax4 = axes[1, 1]
        transformer_info = [
            ("Количество слоев", 2),
            ("Размер модели", 256),
            ("Количество голов", 8),
            ("Размер FFN", 512),
            ("Dropout", 0.1)
        ]

        categories = [info[0] for info in transformer_info]
        values = [info[1] for info in transformer_info]

        bars_trans = ax4.bar(categories, values, color='lightcoral')
        ax4.set_title('Параметры трансформерного энкодера')
        ax4.set_ylabel('Значение')
        ax4.tick_params(axis='x', rotation=45, labelsize=8)

        # Добавляем значения на столбцы
        for bar, value in zip(bars_trans, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     str(value), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('visualization_results/architecture_diagram.png', dpi=300, bbox_inches='tight')
        plt.savefig('visualization_results/architecture_diagram.pdf', bbox_inches='tight')
        print("Диаграмма сохранена как: visualization_results/architecture_diagram.png")

        # Закрываем фигуру, чтобы освободить память
        plt.close(fig)

        # Дополнительная детальная схема
        print("Создание детальной схемы...")
        create_detailed_schematic(input_size)

    except Exception as e:
        print(f"Ошибка при создании диаграммы matplotlib: {e}")
        import traceback
        traceback.print_exc()


def create_detailed_schematic(input_size):
    """Создание детальной схемы архитектуры"""

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Создаем схему потоков данных с правильной структурой
        layers_detailed = [
            {"name": "Input Layer", "layer_type": "input", "shape": f"(1, {input_size})", "color": "#FFE4B5"},
            {"name": "Conv1D", "params": "64 filters, k=7", "layer_type": "conv", "shape": f"(64, {input_size})",
             "color": "#ADD8E6"},
            {"name": "BatchNorm + ReLU", "layer_type": "activation", "shape": f"(64, {input_size})",
             "color": "#DDA0DD"},
            {"name": "MaxPool1D", "params": "stride=2", "layer_type": "pool", "shape": f"(64, {input_size // 2})",
             "color": "#90EE90"},
            {"name": "Conv1D", "params": "128 filters, k=5", "layer_type": "conv", "shape": f"(128, {input_size // 2})",
             "color": "#ADD8E6"},
            {"name": "BatchNorm + ReLU", "layer_type": "activation", "shape": f"(128, {input_size // 2})",
             "color": "#DDA0DD"},
            {"name": "MaxPool1D", "params": "stride=2", "layer_type": "pool", "shape": f"(128, {input_size // 4})",
             "color": "#90EE90"},
            {"name": "Conv1D", "params": "256 filters, k=3", "layer_type": "conv", "shape": f"(256, {input_size // 4})",
             "color": "#ADD8E6"},
            {"name": "BatchNorm + ReLU", "layer_type": "activation", "shape": f"(256, {input_size // 4})",
             "color": "#DDA0DD"},
            {"name": "MaxPool1D", "params": "stride=2", "layer_type": "pool", "shape": f"(256, {input_size // 8})",
             "color": "#90EE90"},
            {"name": "Permute", "layer_type": "reshape", "shape": f"({input_size // 8}, 256)", "color": "#87CEEB"},
            {"name": "Transformer\nEncoder", "params": "2 layers, 8 heads", "layer_type": "transformer",
             "shape": f"({input_size // 8}, 256)", "color": "#F0E68C"},
            {"name": "Flatten", "layer_type": "reshape", "shape": f"({256 * (input_size // 8)})", "color": "#87CEEB"},
            {"name": "Fully Connected", "params": "1024 neurons", "layer_type": "fc", "shape": "(1024)",
             "color": "#87CEEB"},
            {"name": "BatchNorm + ReLU\n+ Dropout", "layer_type": "activation", "shape": "(1024)", "color": "#DDA0DD"},
            {"name": "Output Layer", "params": "4 neurons", "layer_type": "fc", "shape": "(4)", "color": "#87CEEB"},
            {"name": "Sigmoid\n(1st output)", "layer_type": "activation", "shape": "(1)", "color": "#DDA0DD"},
            {"name": "Softmax\n(2nd-4th)", "layer_type": "activation", "shape": "(3)", "color": "#DDA0DD"},
            {"name": "Concatenate", "layer_type": "reshape", "shape": "(4)", "color": "#87CEEB"},
            {"name": "Output", "layer_type": "output", "shape": "(4)", "color": "#98FB98"}
        ]

        # Распределяем слои по вертикали
        y_pos = np.arange(len(layers_detailed)) * 1.5
        box_height = 0.8
        box_width = 5

        for i, layer in enumerate(layers_detailed):
            # Рисуем прямоугольник для слоя
            rect = plt.Rectangle((-box_width / 2, y_pos[i] - box_height / 2),
                                 box_width, box_height,
                                 facecolor=layer['color'],
                                 edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(rect)

            # Добавляем основное название слоя
            ax.text(0, y_pos[i], layer['name'],
                    ha='center', va='center',
                    fontweight='bold', fontsize=9)

            # Добавляем параметры, если они есть
            if 'params' in layer:
                ax.text(0, y_pos[i] - 0.15, layer['params'],
                        ha='center', va='top',
                        fontsize=7, style='italic')

            # Добавляем форму тензора справа
            ax.text(box_width / 2 + 0.5, y_pos[i], layer['shape'],
                    ha='left', va='center',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightgrey',
                              alpha=0.5))

            # Рисуем стрелки между слоями (кроме последнего)
            if i < len(layers_detailed) - 1:
                ax.arrow(0, y_pos[i] + box_height / 2,
                         0, y_pos[i + 1] - y_pos[i] - box_height / 2,
                         head_width=0.15, head_length=0.2,
                         fc='gray', ec='gray', alpha=0.7)

        # Настройки отображения
        ax.set_xlim(-box_width / 2 - 1, box_width / 2 + 6)
        ax.set_ylim(-1, y_pos[-1] + 1)
        ax.set_aspect('auto')
        ax.axis('off')
        ax.set_title('Детальная схема DiffractionNet',
                     fontsize=14, fontweight='bold', pad=20)

        # Добавляем легенду цветов
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#FFE4B5', edgecolor='black', label='Вход'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ADD8E6', edgecolor='black', label='Свертка'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='black', label='Пулинг'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#DDA0DD', edgecolor='black', label='Активация/Нормализация'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#F0E68C', edgecolor='black', label='Трансформер'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#87CEEB', edgecolor='black', label='Полносвязный/Reshape'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#98FB98', edgecolor='black', label='Выход')
        ]

        ax.legend(handles=legend_elements,
                  loc='upper left',
                  bbox_to_anchor=(1.02, 1),
                  borderaxespad=0.,
                  fontsize=8,
                  title='Типы слоев',
                  title_fontsize=9)

        plt.tight_layout()
        plt.savefig('visualization_results/detailed_architecture.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig('visualization_results/detailed_architecture.pdf',
                    bbox_inches='tight')
        print("Детальная схема сохранена как: visualization_results/detailed_architecture.png")

        # Показываем и закрываем
        plt.close(fig)

    except Exception as e:
        print(f"Ошибка при создании детальной схемы: {e}")
        import traceback
        traceback.print_exc()


def create_layer_hierarchy(model):
    """Создание иерархического дерева слоев"""

    def print_tree(module, name, depth=0, max_depth=3):
        """Рекурсивная функция для печати дерева"""
        indent = "  " * depth
        module_type = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters())

        if depth == 0:
            tree_str = f"{indent}📁 {name} ({module_type})"
            if num_params > 0:
                tree_str += f" - {num_params:,} params"
        else:
            tree_str = f"{indent}├─ {name} ({module_type})"
            if num_params > 0:
                tree_str += f" - {num_params:,} params"

        print(tree_str)

        # Если это Sequential или ModuleList, показываем содержимое
        if isinstance(module, (nn.Sequential, nn.ModuleList)) and depth < max_depth:
            for i, child in enumerate(module):
                print_tree(child, f"{name}[{i}]", depth + 1, max_depth)

        # Для других модулей показываем детей, если они есть
        elif hasattr(module, '_modules') and module._modules and depth < max_depth:
            for child_name, child_module in module._modules.items():
                if child_module is not None:
                    print_tree(child_module, child_name, depth + 1, max_depth)

    print("\n" + "=" * 80)
    print("ИЕРАРХИЧЕСКОЕ ДЕРЕВО СЛОЕВ")
    print("=" * 80)

    print_tree(model, "DiffractionNet")

    # Сохраняем в файл
    import sys
    original_stdout = sys.stdout
    with open("visualization_results/layer_hierarchy.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print_tree(model, "DiffractionNet")
        sys.stdout = original_stdout

    print("\nИерархия сохранена в: visualization_results/layer_hierarchy.txt")


def create_block_diagram(model, input_size):
    """Создание блок-схемы архитектуры"""

    try:
        # Используем graphviz для создания блок-схемы
        dot = Digraph(comment='DiffractionNet Architecture',
                      format='png',
                      graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5'},
                      node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Arial'})

        # Цвета для разных типов слоев
        colors = {
            'input': '#FFE4B5',  # Мокасиновый
            'conv': '#ADD8E6',  # Светло-голубой
            'pool': '#90EE90',  # Светло-зеленый
            'norm': '#FFB6C1',  # Светло-розовый
            'activation': '#DDA0DD',  # Сливовый
            'transformer': '#F0E68C',  # Хаки
            'fc': '#87CEEB',  # Небесно-голубой
            'output': '#98FB98'  # Бледно-зеленый
        }

        # Добавляем узлы
        nodes = [
            ('input', f'Input\n(1, {input_size})', 'input'),
            ('conv1', 'Conv1D\n64 filters\nk=7, s=1, p=3', 'conv'),
            ('bn1', 'BatchNorm1d\n64', 'norm'),
            ('relu1', 'ReLU', 'activation'),
            ('pool1', 'MaxPool1d\nk=2, s=2', 'pool'),
            ('conv2', 'Conv1D\n128 filters\nk=5, s=1, p=2', 'conv'),
            ('bn2', 'BatchNorm1d\n128', 'norm'),
            ('relu2', 'ReLU', 'activation'),
            ('pool2', 'MaxPool1d\nk=2, s=2', 'pool'),
            ('conv3', 'Conv1D\n256 filters\nk=3, s=1, p=1', 'conv'),
            ('bn3', 'BatchNorm1d\n256', 'norm'),
            ('relu3', 'ReLU', 'activation'),
            ('pool3', 'MaxPool1d\nk=2, s=2', 'pool'),
            ('permute', 'Permute\n(batch, seq_len, features)', 'activation'),
            ('transformer', 'Transformer Encoder\n2 layers, 8 heads\nd_model=256, ff_dim=512', 'transformer'),
            ('flatten', f'Flatten\n({256 * (input_size // 8)} neurons)', 'activation'),
            ('fc1', 'Fully Connected\n1024 neurons', 'fc'),
            ('bn_fc', 'BatchNorm1d\n1024', 'norm'),
            ('relu_fc', 'ReLU', 'activation'),
            ('dropout', 'Dropout\np=0.1', 'activation'),
            ('fc2', 'Output Layer\n4 neurons', 'fc'),
            ('sigmoid', 'Sigmoid\n(1st output)', 'activation'),
            ('softmax', 'Softmax\n(2nd-4th outputs)', 'activation'),
            ('concat', 'Concatenate', 'activation'),
            ('output', 'Output\n(4 values)', 'output')
        ]

        for node_id, label, node_type in nodes:
            dot.node(node_id, label, fillcolor=colors[node_type])

        # Добавляем связи
        edges = [
            ('input', 'conv1'),
            ('conv1', 'bn1'),
            ('bn1', 'relu1'),
            ('relu1', 'pool1'),
            ('pool1', 'conv2'),
            ('conv2', 'bn2'),
            ('bn2', 'relu2'),
            ('relu2', 'pool2'),
            ('pool2', 'conv3'),
            ('conv3', 'bn3'),
            ('bn3', 'relu3'),
            ('relu3', 'pool3'),
            ('pool3', 'permute'),
            ('permute', 'transformer'),
            ('transformer', 'flatten'),
            ('flatten', 'fc1'),
            ('fc1', 'bn_fc'),
            ('bn_fc', 'relu_fc'),
            ('relu_fc', 'dropout'),
            ('dropout', 'fc2'),
            ('fc2', 'sigmoid'),
            ('fc2', 'softmax'),
            ('sigmoid', 'concat'),
            ('softmax', 'concat'),
            ('concat', 'output')
        ]

        for src, dst in edges:
            dot.edge(src, dst)

        # Добавляем заголовок
        dot.attr(label='DiffractionNet Architecture\n\n', labelloc='t', fontsize='20')

        # Сохраняем
        dot.render('visualization_results/block_diagram', cleanup=True)
        print("Блок-схема сохранена как: visualization_results/block_diagram.png")

        # Создаем упрощенную версию
        dot_simple = Digraph(comment='DiffractionNet Simplified',
                             format='png',
                             graph_attr={'rankdir': 'TB'},
                             node_attr={'shape': 'box', 'style': 'filled'})

        simple_nodes = [
            ('input', 'Input\n1D Signal', '#FFE4B5'),
            ('conv_block', 'Conv Blocks\n(3x Conv1D + BN + ReLU + Pool)', '#ADD8E6'),
            ('transformer', 'Transformer\nEncoder', '#F0E68C'),
            ('fc_block', 'FC Layers\n(1024 → 4 neurons)', '#87CEEB'),
            ('activation', 'Activation\n(Sigmoid + Softmax)', '#DDA0DD'),
            ('output', 'Output\nPhase Fractions', '#98FB98')
        ]

        for node_id, label, color in simple_nodes:
            dot_simple.node(node_id, label, fillcolor=color)

        simple_edges = [
            ('input', 'conv_block'),
            ('conv_block', 'transformer'),
            ('transformer', 'fc_block'),
            ('fc_block', 'activation'),
            ('activation', 'output')
        ]

        for src, dst in simple_edges:
            dot_simple.edge(src, dst)

        dot_simple.attr(label='DiffractionNet (Упрощенная схема)\n\n', labelloc='t', fontsize='16')
        dot_simple.render('visualization_results/block_diagram_simple', cleanup=True)
        print("Упрощенная блок-схема сохранена как: visualization_results/block_diagram_simple.png")

    except Exception as e:
        print(f"Ошибка при создании блок-схемы: {e}")
        print("Убедитесь, что установлен graphviz")


def export_model_summary(model, input_size):
    """Экспорт сводки модели в JSON"""

    summary = {
        "model_name": "DiffractionNet",
        "input_size": input_size,
        "num_phases": model.fc_layers[-1].out_features,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": []
    }

    # Собираем информацию о слоях
    for name, module in model.named_modules():
        if name:  # Пропускаем корневой модуль
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable": any(p.requires_grad for p in module.parameters())
            }
            summary["layers"].append(layer_info)

    # Сохраняем в JSON
    import json
    with open("visualization_results/model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Сводка модели сохранена в: visualization_results/model_summary.json")

    return summary


if __name__ == "__main__":
    print("Запуск визуализации архитектуры DiffractionNet...")
    print("Это может занять несколько секунд...\n")

    # Запускаем все методы визуализации
    visualize_architecture()

    # Дополнительно: экспорт в JSON
    print("\n7. СОЗДАНИЕ JSON-СВОДКИ МОДЕЛИ...")
    model = DiffractionNet(4001, 4)
    export_model_summary(model, 4001)

    print("\n" + "=" * 80)
    print("ВСЕ ФАЙЛЫ СОХРАНЕНЫ В ПАПКЕ 'visualization_results/':")
    print("=" * 80)
    print("1. architecture_description.txt - Текстовое описание")
    print("2. computation_graph.png - Граф вычислений (torchviz)")
    print("3. tensorboard_logs/ - Данные для TensorBoard")
    print("4. architecture_diagram.png - Схематичная диаграмма")
    print("5. detailed_architecture.png - Детальная схема")
    print("6. layer_hierarchy.txt - Иерархия слоев")
    print("7. block_diagram.png - Блок-схема архитектуры")
    print("8. block_diagram_simple.png - Упрощенная блок-схема")
    print("9. model_summary.json - JSON-сводка модели")
    print("=" * 80)
    print("\nДля просмотра в TensorBoard выполните:")
    print("tensorboard --logdir=visualization_results/tensorboard_logs")