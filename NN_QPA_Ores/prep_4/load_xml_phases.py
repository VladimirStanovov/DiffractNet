import xml.etree.ElementTree as ET


def correct_space_group(space_group):
    # Убираем крайние пробелы из строки
    space_group = space_group.strip()

    # Проверяем, заканчивается ли строка на 'H' или 'Z'
    if space_group.endswith('Z'):
        # Убираем последний символ (H или Z) из строки
        space_group = space_group[:-1]
        space_group = space_group + ':2'

    """
    # Проверяем, заканчивается ли строка на 'H' или 'Z'
    if space_group.endswith('H') or space_group.endswith('Z'):
        # Убираем последний символ (H или Z) из строки
        space_group = space_group[:-1]
        space_group = space_group.strip()
    """

    return space_group

def correct_atom_type(atom_type):
    # Проверяем наличие знака '+' или '-' в строке
    if '+' in atom_type:
        # Разделяем строку на две части по знаку '+'
        parts = atom_type.split('+')
        # Убираем возможные пробелы и объединяем части с '+' в конце
        corrected_type = ''.join(parts).strip() + '+'
    elif '-' in atom_type:
        # Разделяем строку на две части по знаку '-'
        parts = atom_type.split('-')
        # Убираем возможные пробелы и объединяем части с '-' в конце
        corrected_type = ''.join(parts).strip() + '-'
    else:
        # Если знака нет, оставляем строку без изменений
        corrected_type = atom_type.strip()

    return corrected_type

def convert_to_number(s, default_value = 0):
    if s is None:
        return default_value

    try:
        s = s.strip()
        return float(s)
    except ValueError:
        return default_value

def parse_param(value_text, min_text, max_text):
    if '@' in value_text:
        value = float(value_text.split('@ ')[-1])
        # Если параметр варьируется, используем начальное значение для генерации диапазона
        if min_text is None or min_text == '':
            min_val = value * 0.99  # -1%
        else:
            min_val = float(min_text)
        if max_text is None or max_text == '':
            max_val = value * 1.01  # +1%
        else:
            max_val = float(max_text)
    else:
        # Проверяем наличие специальных подстрок и вычисляем значение
        if '1/3' in value_text:
            value = 1 / 3
        elif '2/3' in value_text:
            value = 2 / 3
        elif '1/6' in value_text:
            value = 1 / 6
        elif '5/6' in value_text:
            value = 5 / 6
        else:
            value = float(value_text)
        # Если параметр не варьируется, убираем мин и макс значения
        min_val = None
        max_val = None

    return {'min': min_val, 'max': max_val, 'value': value}


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    sample = root.find('Sample')
    ze_min = sample.find('ZErrorMin').text
    ze_max = sample.find('ZErrorMax').text

    ze = max( abs(convert_to_number(ze_min, -0.05)), abs(convert_to_number(ze_max, 0.05)) )

    sample_info = {
        'Rs': convert_to_number(sample.find('Rs').text, 200),
        'SAM': convert_to_number(sample.find('SAM').text, 12),
        'ZE': ze
    }

    phases_info = []

    for phase in root.findall('Phase'):
        if phase.get('Exported') == '-1':

            phase_info = {
                'Name': phase.get('Name'),
                'SpaceGroup': phase.get('SpaceGroup'),
                'CellPar': {},
                'ReflectionProfileScherrer': {},
                'MDtexDir': [],
                'Atoms': [],
                'RIR': -1
            }

            # Парсим параметры ячейки
            for cellpar in phase.findall('CellPar'):
                name = cellpar.get('Name')
                min_val = cellpar.get('Min')
                max_val = cellpar.get('Max')
                value_text = cellpar.text.strip()
                parsed_param = parse_param(value_text, min_val, max_val)
                phase_info['CellPar'][name] = parsed_param

            # Парсим параметры профиля разброса
            for par in phase.find('ReflectionProfileScherrer').findall('Par'):
                name = par.get('Name')
                min_val = par.get('Min')
                max_val = par.get('Max')
                value_text = par.text.strip()
                parsed_param = parse_param(value_text, min_val, max_val)
                phase_info['ReflectionProfileScherrer'][name] = parsed_param

            # Проверяем значение TextureChoise
            texture_choise = phase.find('TextureChoise')
            if texture_choise is not None and texture_choise.text.strip() == '1':
                # Парсим параметры MDtexDir
                for mdtexdir in phase.findall('MDtexDir'):
                    name = mdtexdir.get('Name')
                    min_val = mdtexdir.get('Min')
                    max_val = mdtexdir.get('Max')
                    value_text = '@ '+mdtexdir.text.strip()
                    parsed_param = parse_param(value_text, min_val, max_val)
                    phase_info['MDtexDir'].append({'name': name, **parsed_param})

            # Парсим атомы
            for atom in phase.findall('Atom'):
                atom_info = {
                    'Type': correct_atom_type(atom.get('Type')),
                    'X': parse_param(atom.get('X'), atom.get('Xmin'), atom.get('Xmax')),
                    'Y': parse_param(atom.get('Y'), atom.get('Ymin'), atom.get('Ymax')),
                    'Z': parse_param(atom.get('Z'), atom.get('Zmin'), atom.get('Zmax')),
                    'Occ': parse_param(atom.get('Occ'), atom.get('OccMin'), atom.get('OccMax'))
                }
                phase_info['Atoms'].append(atom_info)

            print(phase_info, '\n')

            phases_info.append(phase_info)

    return sample_info, phases_info


def parse_xml_multi(sample_file, ph_files):
    tree = ET.parse(sample_file)
    root = tree.getroot()

    sample = root.find('Sample')
    ze_min = sample.find('ZErrorMin').text
    ze_max = sample.find('ZErrorMax').text

    ze = max( abs(convert_to_number(ze_min, -0.05)), abs(convert_to_number(ze_max, 0.05)) )

    sample_info = {
        'Rs': convert_to_number(sample.find('Rs').text, 200),
        'SAM': convert_to_number(sample.find('SAM').text, 12),
        'ZE': ze
    }

    phases_info = []

    for ph_file in ph_files:
        print(f"Read file: {ph_file}")
        tree = ET.parse(ph_file)
        root = tree.getroot()

        phase = root.find('Phase')
        if phase.get('Exported') == '-1':

            phase_info = {
                'Name': phase.get('Name'),
                'SpaceGroup': phase.get('SpaceGroup'),
                'CellPar': {},
                'ReflectionProfileScherrer': {},
                'MDtexDir': [],
                'Atoms': [],
                'RIR': -1
            }

            # Парсим параметры ячейки
            for cellpar in phase.findall('CellPar'):
                name = cellpar.get('Name')
                min_val = cellpar.get('Min')
                max_val = cellpar.get('Max')
                value_text = cellpar.text.strip()
                parsed_param = parse_param(value_text, min_val, max_val)
                phase_info['CellPar'][name] = parsed_param

            # Парсим параметры профиля разброса
            for par in phase.find('ReflectionProfileScherrer').findall('Par'):
                name = par.get('Name')
                min_val = par.get('Min')
                max_val = par.get('Max')
                value_text = par.text.strip()
                parsed_param = parse_param(value_text, min_val, max_val)
                phase_info['ReflectionProfileScherrer'][name] = parsed_param

            # Проверяем значение TextureChoise
            texture_choise = phase.find('TextureChoise')
            if texture_choise is not None and texture_choise.text.strip() == '1':
                # Парсим параметры MDtexDir
                for mdtexdir in phase.findall('MDtexDir'):
                    name = mdtexdir.get('Name')
                    min_val = mdtexdir.get('Min')
                    max_val = mdtexdir.get('Max')
                    value_text = '@ '+mdtexdir.text.strip()
                    parsed_param = parse_param(value_text, min_val, max_val)
                    phase_info['MDtexDir'].append({'name': name, **parsed_param})

            # Парсим атомы
            for atom in phase.findall('Atom'):
                atom_info = {
                    'Type': correct_atom_type(atom.get('Type')),
                    'X': parse_param(atom.get('X'), atom.get('Xmin'), atom.get('Xmax')),
                    'Y': parse_param(atom.get('Y'), atom.get('Ymin'), atom.get('Ymax')),
                    'Z': parse_param(atom.get('Z'), atom.get('Zmin'), atom.get('Zmax')),
                    'Occ': parse_param(atom.get('Occ'), atom.get('OccMin'), atom.get('OccMax'))
                }
                phase_info['Atoms'].append(atom_info)

            print(phase_info)

            phases_info.append(phase_info)

    return sample_info, phases_info

