import ctypes
import json
import os
import shutil

class CrystLib:
    #def __del__(self):
    #    # Удаляем временную директорию и файл DLL, если они существуют
    #    if self.temp_dir and os.path.exists(self.temp_dir):
    #        shutil.rmtree(self.temp_dir)

    def __init__(self, proc_id = None):
        self.proc_id = proc_id
        self.dll_path = './libCIFtoDB.so'
        self.temp_dir = None

        if proc_id is not None:
            # Создаем директорию с именем proc_id
            self.temp_dir = os.path.join(os.getcwd(), str(proc_id))
            os.makedirs(self.temp_dir, exist_ok=True)
            # Копируем DLL в созданную директорию
            self.dll_path = os.path.join(self.temp_dir, str(proc_id)+'.so')
            shutil.copyfile('./libCIFtoDB.so', self.dll_path)

        # Загрузка библиотеки
        self.lib = ctypes.CDLL(self.dll_path)

        # Определение типов аргументов и возвращаемых значений функций
        self.lib.AddCrystal.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                       ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                       ctypes.c_char_p]
        self.lib.AddCrystal.restype = None

        self.lib.SetCrystalCellParameters.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                     ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                     ctypes.c_double]
        self.lib.SetCrystalCellParameters.restype = None

        self.lib.AddScatteringPowerToCrystal.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
        self.lib.AddScatteringPowerToCrystal.restype = None

        self.lib.AddAtomToCrystal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                             ctypes.c_double, ctypes.c_double]
        self.lib.AddAtomToCrystal.restype = None

        self.lib.SetAtomParameters.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                              ctypes.c_double, ctypes.c_double]
        self.lib.AddAtomToCrystal.restype = None

        self.lib.InitializePowderData.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        self.lib.InitializePowderData.restype = None

        self.lib.AddTextureToDiffData.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.AddTextureToDiffData.restype = None

        self.lib.SetTextureCParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.lib.SetTextureCParameter.restype = None

        self.lib.GetMaxInt.argtypes = [ctypes.c_int]
        self.lib.GetMaxInt.restype = ctypes.c_double

        self.lib.GetNormalizedPeaks.argtypes = [ctypes.c_int]
        self.lib.GetNormalizedPeaks.restype = ctypes.c_char_p

        self.lib.CalcRIR.argtypes = [ctypes.c_int]
        self.lib.CalcRIR.restype = ctypes.c_double

        self.lib.FreeMemory.argtypes = []
        self.lib.FreeMemory.restype = None

        self.lib.FreeString.argtypes = []
        self.lib.FreeString.restype = None

    def AddCrystal(self, a, b, c, alpha, beta, gamma, spacegroup):
        self.lib.AddCrystal(a, b, c, alpha, beta, gamma, str(spacegroup).encode('utf-8'))

    def SetCrystalCellParameters(self, index, a, b, c, alpha, beta, gamma):
        self.lib.SetCrystalCellParameters(index, a, b, c, alpha, beta, gamma)

    def AddScatteringPowerToCrystal(self, index, sp_name, sp_type, sp_Biso):
        self.lib.AddScatteringPowerToCrystal(index, str(sp_name).encode('utf-8'), str(sp_type).encode('utf-8'), sp_Biso)

    def AddAtomToCrystal(self, index, scatteringPowerIndex, x, y, z, occupancy):
        self.lib.AddAtomToCrystal(index, scatteringPowerIndex, x, y, z, occupancy)

    def SetAtomParameters(self, index, at_index, x, y, z, occupancy):
        self.lib.SetAtomParameters(index, at_index, x, y, z, occupancy)

    def InitializePowderData(self, wavelength, param1, param2, param3):
        # print(f"{wavelength} {param1} {param2} {param3}")
        self.lib.InitializePowderData(wavelength, param1, param2, param3)

    def AddTextureToDiffData(self, index, c, h, k, l):
        self.lib.AddTextureToDiffData(index, c, h, k, l)

    def SetTextureCParameter(self, diffIndex, phaseIndex, c):
        self.lib.SetTextureCParameter(diffIndex, phaseIndex, c)

    def GetMaxInt(self, diffIndex):
        return self.lib.GetMaxInt(diffIndex)

    def GetNormalizedPeaks(self, diffIndex):
        peaks_json = self.lib.GetNormalizedPeaks(diffIndex)
        peaks = json.loads(peaks_json.decode('utf-8'))
        # Освобождение памяти для строки
        self.lib.FreeString()
        return peaks

    def CalcRIR(self, index):
        return self.lib.CalcRIR(index)

    def FreeMemory(self):
        self.lib.FreeMemory()
