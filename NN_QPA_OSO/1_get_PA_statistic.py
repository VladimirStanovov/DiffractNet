import os
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

# Определяем структуру заголовка и атомов
PHASE_HEADER_LINE = "SRfactor SZero SDispl STransp + C Wght ESU SF A B C Alpha Beta Gamma NA NB NC P1 P2 P3 U V W Z hh kk ll hk hl kl s1 s2 s3 s4 s5 s6"
PHASE_PARAMS = PHASE_HEADER_LINE.split()[7:]  # Начинаем с A
ATOM_PARAMS = ["Name", "Type", "X", "Y", "Z", "B", "N"]

# Словарь соответствия параметров статистики и XML
PHASE_PARAM_MAPPING = {
    "A": ("CellPar", "a"),
    "B": ("CellPar", "b"),
    "C": ("CellPar", "c"),
    "Alpha": ("CellPar", "alpha"),
    "Beta": ("CellPar", "beta"),
    "Gamma": ("CellPar", "gamma"),
    "NA": ("ReflectionProfileScherrer", "Par[@Name='CS']"), # Требует специальной обработки
    "W": ("ReflectionProfileScherrer", "Par[@Name='Strain']"), # Требует специальной обработки
    "hh": ("MDtexDir", "[1]"), # Требует специальной обработки (первый)
    "kk": ("MDtexDir", "[2]"), # Требует специальной обработки (второй)
    "ll": ("MDtexDir", "[3]"), # Требует специальной обработки (третий)
}

ATOM_PARAM_MAPPING = {
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "B": "Beq",
    "N": "Occ"
}

# Функция для проверки, является ли файл валидным
def is_valid_file(file_path):
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            return first_line == PHASE_HEADER_LINE
    except Exception:
        return False


# Функция для парсинга файла
def parse_file(file_path):
    phases_data = []

    if not is_valid_file(file_path):
        print(f"Пропущен файл (неверный формат): {file_path}")
        return phases_data

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return phases_data

    # Пропускаем первую строку (заголовок)
    i = 2
    while i < len(lines):
        phase_name = lines[i]
        i += 1
        # Ожидаем строку фазы
        phase_line_parts = lines[i].split()
        if len(phase_line_parts) < 15:
            i += 1
            continue

        try:
            # Проверяем, является ли первый элемент числом (индекс фазы)
            int(phase_line_parts[0])
        except ValueError:
            i += 1
            continue  # Не строка фазы, пропускаем

        # Извлекаем параметры фазы
        # У нас есть 35 параметров фазы, плюс индекс, имя, AtomCount и C (конц-я)
        # Всего в строке 39 элементов. Если меньше, считаем файл поврежденным для этой фазы.
        if len(phase_line_parts) < 32:
            print(f"Предупреждение: Недостаточно параметров для фазы '{phase_name}' в файле {file_path}. Пропущено.")
            i += 1
            # Пропускаем атомы этой фазы
            atom_count = int(phase_line_parts[0]) if len(phase_line_parts) > 2 and phase_line_parts[2].isdigit() else 0
            i += 1 + atom_count
            continue

        phase_data = {
            "file": file_path,
            "name": phase_name,
            "concentration": float(phase_line_parts[1]),  # Параметр C
            "params": {name: float(val) for name, val in zip(PHASE_PARAMS, phase_line_parts[2:32])},
            "atoms": []
        }

        # Извлекаем атомы
        atom_count = int(phase_line_parts[0])
        for j in range(1, atom_count + 1):
            if i + j >= len(lines):
                print(f"Ошибка: Неожиданный конец файла при чтении атомов фазы '{phase_name}' в {file_path}")
                break
            atom_line_parts = lines[i + j].split()
            if len(atom_line_parts) < 7:
                print(
                    f"Предупреждение: Недостаточно данных для атома в фазе '{phase_name}' в файле {file_path}. Пропущено.")
                continue
            atom_data = {
                "name": atom_line_parts[0],
                "type": atom_line_parts[1],
                "params": {name: float(val) for name, val in zip(ATOM_PARAMS[2:], atom_line_parts[2:8])}
                # X, Y, Z, B, N
            }
            if atom_data['params']['N'] < 0.0:
                atom_data['params']['N'] = 0.0
            if atom_data['params']['N'] > 1.0:
                atom_data['params']['N'] = 1.0
            phase_data["atoms"].append(atom_data)

        phases_data.append(phase_data)
        i += 1 + atom_count

    return phases_data


# --- Статистические функции ---
def weighted_average(values, weights):
    """Вычисляет средневзвешенное значение."""
    if not values or not weights or len(values) != len(weights):
        return None
    try:
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)
    except ZeroDivisionError:
        return None


def weighted_std(values, weights, weighted_avg):
    """Вычисляет взвешенное стандартное отклонение."""
    if not values or not weights or len(values) != len(weights) or weighted_avg is None:
        return None
    if len(values) < 2:
        return 0.0

    sum_weights = sum(weights)
    if sum_weights == 0:
        return None

    # Формула для взвешенного стандартного отклонения
    variance = sum(w * (v - weighted_avg) ** 2 for v, w in zip(values, weights)) / sum_weights
    # Для несмещенной оценки часто используется деление на (sum_weights - 1),
    # но для простоты и соответствия стандартному отклонению будем делить на sum_weights
    # Если нужно несмещенное, раскомментируйте строку ниже и закомментируйте следующую
    # variance = sum(w * (v - weighted_avg) ** 2 for v, w in zip(values, weights)) / (sum_weights - 1) if sum_weights > 1 else 0
    return math.sqrt(variance)


def calculate_param_stats(values, weights, std_mult = 1.5, min_value = None):
    """Вычисляет статистику для одного параметра."""
    if not values:
        return {"min": None, "max": None, "avg": None, "std": None, "range": None}

    min_val = min(values)
    max_val = max(values)

    if len(set(values)) < 3:  # Все значения одинаковы
        return {"min": min_val, "max": max_val, "avg": (min_val+max_val)/2.0, "std": 0.0, "range": [min_val, max_val]}

    avg_val = weighted_average(values, weights)
    std_val = weighted_std(values, weights, avg_val) if avg_val is not None else None
    if avg_val is not None and std_val is not None:
        range_min = avg_val - std_mult * std_val
        range_max = avg_val + std_mult * std_val
        if min_value and range_min < range_min:
            range_min = range_min
        # Убедимся, что диапазон в пределах min/max
        range_min = max(range_min, min_val)
        range_max = min(range_max, max_val)
        return {
            "min": min_val, "max": max_val,
            "avg": avg_val, "std": std_val,
            "range": [range_min, range_max]
        }


# --- Основная логика ---
def collect_statistics(base_folder):
    all_phases = defaultdict(list)  # {phase_name: [phase_data1, phase_data2, ...]}

    # 1. Поиск и парсинг файлов
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Обработка файла: {file_path}")
                phases = parse_file(file_path)
                for phase in phases:
                    all_phases[phase["name"]].append(phase)

    # 2. Формирование статистики и запись JSON
    phase_stats_db = {} # Для использования в дальнейшем
    for phase_name, phase_entries in all_phases.items():
        print(f"Формирование статистики для фазы: {phase_name} (найдено {len(phase_entries)} записей)")

        # Собираем концентрации для взвешивания
        concentrations = [p["concentration"] for p in phase_entries]

        # --- Статистика по параметрам фазы ---
        phase_stats = {}
        for param_name in PHASE_PARAMS:
            values = [p["params"].get(param_name) for p in phase_entries]
            # Фильтруем None, если были ошибки парсинга
            valid_data = [(v, c) for v, c in zip(values, concentrations) if v is not None]
            if valid_data:
                valid_values, valid_weights = zip(*valid_data)
                if param_name == 'NA':
                    phase_stats[param_name] = calculate_param_stats(valid_values, valid_weights, min_value=20)
                elif param_name == 'W':
                    phase_stats[param_name] = calculate_param_stats(valid_values, valid_weights, min_value=0.0001)
                else:
                    phase_stats[param_name] = calculate_param_stats(valid_values, valid_weights)
            else:
                phase_stats[param_name] = {"min": None, "max": None, "avg": None, "std": None, "range": None}

        # --- Статистика по атомам ---
        # Предполагаем, что структура атомов одинакова во всех записях одной фазы
        # (одинаковый набор атомов с одинаковыми именами)
        atom_stats = {}
        if phase_entries and phase_entries[0]["atoms"]:
            num_atoms = len(phase_entries[0]["atoms"])

            for atom_idx in range(num_atoms):
                # Проверяем, что у всех записей есть атом с этим индексом
                if not all(len(p["atoms"]) > atom_idx for p in phase_entries):
                    print(
                        f"Предупреждение: Несовпадение структуры атомов для фазы {phase_name} в разных файлах. Статистика по атомам может быть неполной.")
                    continue

                atom_name = phase_entries[0]["atoms"][atom_idx]["name"]
                atom_type = phase_entries[0]["atoms"][atom_idx]["type"]
                atom_key = f"{atom_name} ({atom_type})"

                atom_stats[atom_key] = {}
                for param_name in ATOM_PARAMS[2:]:  # X, Y, Z, B, N
                    values = [p["atoms"][atom_idx]["params"].get(param_name) for p in phase_entries]
                    valid_data = [(v, c) for v, c in zip(values, concentrations) if v is not None]
                    if valid_data:
                        valid_values, valid_weights = zip(*valid_data)
                        atom_stats[atom_key][param_name] = calculate_param_stats(valid_values, valid_weights)
                    else:
                        atom_stats[atom_key][param_name] = {"min": None, "max": None, "avg": None, "std": None,
                                                            "range": None}

        # --- Создание JSON ---
        output_data = {
            "phase_name": phase_name,
            "total_entries": len(phase_entries),
            "phase_parameters": phase_stats,
            "atom_parameters": atom_stats
        }
        phase_stats_db[phase_name] = output_data # Сохраняем в памяти

        safe_filename = "".join(c for c in phase_name if c.isalnum() or c in (" ", "_")).rstrip()
        output_filename = f"./Statistic/{safe_filename}.json"

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Статистика для фазы '{phase_name}' сохранена в '{output_filename}'")

    return phase_stats_db

# --- Логика для работы с .tshx файлами ---
def update_tshx_files(tshx_folder, stats_db):
    print("\n--- Обновление .tshx файлов ---")
    for root, _, files in os.walk(tshx_folder):
        for file in files:
            if file.endswith(".tshx"):
                file_path = os.path.join(root, file)
                print(f"Обработка .tshx файла: {file_path}")
                try:
                    tree = ET.parse(file_path)
                    root_element = tree.getroot()
                    
                    # Найти все фазы в .tshx
                    phases = root_element.findall(".//Phase")
                    modified = False
                    
                    for phase in phases:
                        phase_name = phase.get("Name")
                        if not phase_name:
                            continue
                        
                        # Найти статистику для этой фазы
                        if phase_name not in stats_db:
                            print(f"  Предупреждение: Статистика для фазы '{phase_name}' не найдена.")
                            continue
                        
                        stats = stats_db[phase_name]
                        print(f"  Обновление фазы: {phase_name}")
                        
                        # Обновление параметров фазы
                        for stat_param_name, (xml_tag, xml_attr_or_name) in PHASE_PARAM_MAPPING.items():
                            stat_info = stats["phase_parameters"].get(stat_param_name)
                            if not stat_info or not stat_info["range"]:
                                print(f"    Пропущен параметр {stat_param_name} (нет данных)")
                                continue

                            # Проверка наличия статистики и диапазона
                            if not stat_info:
                                print(f"    Пропущен параметр {stat_param_name} (нет данных в статистике)")
                                continue

                            range_data = stat_info.get("range")
                            # print(f"  DEBUG: Диапазон из статистики: {range_data}")

                            if not range_data:
                                print(f"    Пропущен параметр {stat_param_name} (нет данных о диапазоне)")
                                continue

                            if len(range_data) < 2:
                                 print(f"    Пропущен параметр {stat_param_name} (диапазон некорректный: {range_data})")
                                 continue
                            
                            min_val, max_val = range_data[0], range_data[1]
                            if min_val is None or max_val is None:
                                print(f"    Пропущен параметр {stat_param_name} (min или max равен None)")
                                continue

                            min_val = round(min_val, 6)
                            max_val = round(max_val, 6)

                            elem = None
                            # Специальная обработка для сложных путей
                            if stat_param_name in ["NA", "W"]:
                                # Ищем Par с конкретным атрибутом Name
                                attr_name = "CS" if stat_param_name == "NA" else "Strain"
                                elem = phase.find(f".//{xml_tag}/Par[@Name='{attr_name}']")
                            elif stat_param_name in ["hh", "kk", "ll"]:
                                # Ищем MDtexDir по порядковому номеру
                                index_map = {"hh": 1, "kk": 2, "ll": 3}
                                index = index_map[stat_param_name]
                                md_dirs = phase.findall(".//MDtexDir")
                                if len(md_dirs) >= index:
                                    elem = md_dirs[index - 1]
                                else:
                                    elem = None
                                    print(f"    Пропущен параметр {stat_param_name} (не найден MDtexDir #{index})")
                            else:
                                # Простой случай
                                elem = phase.find(f".//{xml_tag}[@Name='{xml_attr_or_name}']")
                            
                            if elem is not None:
                                # Убедимся, что значения строковые
                                elem.set("Min", str(min_val))
                                elem.set("Max", str(max_val))
                                modified = True
                                print(f"    Обновлен параметр {stat_param_name}: [{min_val}, {max_val}]")
                            else:
                                print(f"    Не найден XML элемент для параметра {stat_param_name} (XPath: ...//{xml_tag}[@Name='{xml_attr_or_name}'] или аналог)")

                        # Обновление параметров атомов
                        atoms = phase.findall(".//Atom")
                        for atom in atoms:
                            atom_name = atom.get("Name")
                            atom_type = atom.get("Type")
                            atom_key = f"{atom_name} ({atom_type})"
                            
                            if atom_key not in stats["atom_parameters"]:
                                print(f"    Предупреждение: Статистика для атома '{atom_key}' не найдена.")
                                continue
                            
                            atom_stats = stats["atom_parameters"][atom_key]
                            for stat_param_name, xml_attr_name in ATOM_PARAM_MAPPING.items():
                                stat_info = atom_stats.get(stat_param_name)
                                if not stat_info or not stat_info["range"]:
                                    print(f"      Пропущен атомный параметр {atom_key}/{stat_param_name} (нет данных)")
                                    continue
                                
                                min_val, max_val = stat_info["range"]
                                if min_val is None or max_val is None:
                                    print(f"      Пропущен атомный параметр {atom_key}/{stat_param_name} (диапазон None)")
                                    continue

                                # Для X, Y, Z используем @ перед именем атрибута, если это значение уточняется
                                if stat_param_name in ["X", "Y", "Z"]:
                                    current_val_str = atom.get(xml_attr_name)
                                    if current_val_str and current_val_str.startswith("@"):
                                        atom.set(f"{xml_attr_name}min", str(round(min_val, 6)))
                                        atom.set(f"{xml_attr_name}max", str(round(max_val, 6)))
                                        modified = True
                                        print(f"      Обновлен атомный параметр {atom_key}/{stat_param_name}: [{min_val:.6f}, {max_val:.6f}]")
                                    else:
                                        # Если параметр не помечен как уточняемый (@), пропускаем
                                        print(f"      Пропущен атомный параметр {atom_key}/{stat_param_name} (не уточняемый)")
                                else:
                                    # Для Occ и Beq устанавливаем Min/Max напрямую
                                    atom.set(f"{xml_attr_name}Min", str(round(min_val, 6)))
                                    atom.set(f"{xml_attr_name}Max", str(round(max_val, 6)))
                                    modified = True
                                    print(f"      Обновлен атомный параметр {atom_key}/{stat_param_name}: [{min_val:.6f}, {max_val:.6f}]")

                    # Сохранить файл, если были изменения
                    if modified:
                        # Создаем резервную копию
                        # backup_path = file_path + ".bak"
                        # os.rename(file_path, backup_path)
                        # Сохраняем с оригинальным именем
                        tree.write(file_path, encoding="utf-8", xml_declaration=True)
                        # print(f"  Файл {file_path} успешно обновлен (резервная копия: {backup_path})")
                        print(f"  Файл {file_path} успешно обновлен.")
                    else:
                        print(f"  Файл {file_path} не был изменен.")

                except ET.ParseError as e:
                    print(f"Ошибка парсинга XML в файле {file_path}: {e}")
                except Exception as e:
                    print(f"Ошибка обработки файла {file_path}: {e}")


# --- Запуск ---
if __name__ == "__main__":
    os.makedirs("./Statistic/", exist_ok=True)

    txt_folder_path = "./XRD_Data/" # input("Введите путь к папке с файлами .txt: ").strip()
    if not txt_folder_path:
        txt_folder_path = "."  # Текущая директория по умолчанию

    if not os.path.isdir(txt_folder_path):
        print("Ошибка: Указанный путь к .txt файлам не является директорией.")
        exit(0)
    else:
        stats_database = collect_statistics(txt_folder_path)
        print("Этап 1 (Сбор статистики) завершен.")

    # Этап 2: Обновление .tshx файлов
    tshx_folder_path = "./DB3/" # input("Введите путь к папке с файлами .tshx: ").strip()
    if not tshx_folder_path:
        tshx_folder_path = "." # Текущая директория по умолчанию

    if not os.path.isdir(tshx_folder_path):
        print("Ошибка: Указанный путь к .tshx файлам не является директорией или не существует.")
    else:
        update_tshx_files(tshx_folder_path, stats_database)
        print("Этап 2 (Обновление .tshx) завершен.")

    print("Вся обработка завершена.")
