import numpy as np
import matplotlib.pyplot as plt
from typing import List
import random


# Определяем те же функции, что и в исходном коде
def _lognormal_weights(k: int) -> List[float]:
    """Случайные веса из логнормального распределения."""
    mu, sigma = 0, 1.2
    vals = np.random.lognormal(mu, sigma, k)
    return vals.tolist()


def _pareto_weights(k: int, alpha: float = 1.0) -> List[float]:
    """Случайные веса из распределения Парето."""
    vals = np.random.pareto(alpha, k) + 1
    return vals.tolist()


def _dirichlet_weights(k: int) -> List[float]:
    """Генерация через Dirichlet с маленькими alpha (неравномерно)."""
    alpha = np.random.uniform(0.05, 0.8, k)
    vals = np.random.dirichlet(alpha, size=1)[0]
    return vals.tolist()


def _uniform_weights(k: int) -> List[float]:
    """Равномерное распределение для сравнения."""
    return [random.random() for _ in range(k)]


def generate_distribution_samples(distribution_func, n_phases=10, n_samples=10000):
    """
    Генерирует образцы массовых долей для заданной функции распределения.

    Args:
        distribution_func: Функция генерации случайных весов
        n_phases: Количество фаз
        n_samples: Количество образцов для статистики

    Returns:
        Массив отсортированных массовых долей для каждого образца
    """
    samples = []

    for _ in range(n_samples):
        # Генерируем веса для всех фаз
        raw_weights = distribution_func(n_phases)

        # Нормализуем, чтобы сумма была равна 1
        total = sum(raw_weights)
        if total == 0:
            normalized = [1.0 / n_phases] * n_phases
        else:
            normalized = [w / total for w in raw_weights]

        # Сортируем по убыванию
        sorted_weights = sorted(normalized, reverse=True)
        samples.append(sorted_weights)

    return np.array(samples)


# Параметры визуализации
N_PHASES = 10
N_SAMPLES = 5000
PHASE_INDICES = np.arange(1, N_PHASES + 1)

# Настройки шрифтов для научной публикации
plt.rcParams.update({
    'font.size': 14,  # Базовый размер шрифта увеличен в 2 раза (было 7-10)
    'axes.titlesize': 18,  # Заголовки подграфиков
    'axes.labelsize': 16,  # Подписи осей
    'xtick.labelsize': 14,  # Метки на оси X
    'ytick.labelsize': 14,  # Метки на оси Y
    'legend.fontsize': 12,  # Легенда
    'figure.titlesize': 20,  # Общий заголовок
})

# Создаем фигуру с 4 подграфиками - увеличиваем размер для читаемости
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Цвета для разных распределений
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 1. Равномерное распределение
print("Генерация образцов для равномерного распределения...")
uniform_samples = generate_distribution_samples(_uniform_weights, N_PHASES, N_SAMPLES)
uniform_means = np.mean(uniform_samples, axis=0)
uniform_stds = np.std(uniform_samples, axis=0)

ax = axes[0, 0]
uniform_err_lower = np.clip(uniform_stds, 0, uniform_means)
bars = ax.bar(PHASE_INDICES, uniform_means, yerr=[uniform_err_lower, uniform_stds],
              color=colors[0], alpha=0.8, capsize=6, error_kw={'elinewidth': 2.5, 'capthick': 2})
ax.set_title('Равномерное распределение', fontweight='bold')
ax.set_xlabel('Фаза', fontsize=18)  # Увеличено дополнительно
ax.set_ylabel('Массовая доля', fontsize=18)  # Увеличено дополнительно
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(PHASE_INDICES)
ax.set_ylim(bottom=0)
# Добавляем значения над столбцами для лучшей читаемости
# for i, (bar, mean) in enumerate(zip(bars, uniform_means)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
#             f'{mean:.3f}', ha='center', va='bottom', fontsize=12, rotation=0)

# 2. Логнормальное распределение
print("Генерация образцов для логнормального распределения...")
lognormal_samples = generate_distribution_samples(_lognormal_weights, N_PHASES, N_SAMPLES)
lognormal_means = np.mean(lognormal_samples, axis=0)
lognormal_stds = np.std(lognormal_samples, axis=0)

ax = axes[1, 0]
lognormal_err_lower = np.clip(lognormal_stds, 0, lognormal_means)
bars = ax.bar(PHASE_INDICES, lognormal_means, yerr=[lognormal_err_lower, lognormal_stds],
              color=colors[1], alpha=0.8, capsize=6, error_kw={'elinewidth': 2.5, 'capthick': 2})
ax.set_title('Логнормальное (μ=0, σ=1.2)', fontweight='bold')
ax.set_xlabel('Фаза', fontsize=18)
ax.set_ylabel('Массовая доля', fontsize=18)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(PHASE_INDICES)
ax.set_ylim(bottom=0)
# Добавляем значения над столбцами
# for i, (bar, mean) in enumerate(zip(bars, lognormal_means)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
#             f'{mean:.3f}', ha='center', va='bottom', fontsize=12, rotation=0)

# 3. Распределение Парето
print("Генерация образцов для распределения Парето...")
pareto_samples = generate_distribution_samples(lambda k: _pareto_weights(k, 1.0), N_PHASES, N_SAMPLES)
pareto_means = np.mean(pareto_samples, axis=0)
pareto_stds = np.std(pareto_samples, axis=0)

ax = axes[1, 1]
pareto_err_lower = np.clip(pareto_stds, 0, pareto_means)
bars = ax.bar(PHASE_INDICES, pareto_means, yerr=[pareto_err_lower, pareto_stds],
              color=colors[2], alpha=0.8, capsize=6, error_kw={'elinewidth': 2.5, 'capthick': 2})
ax.set_title('Парето (α=1.0)', fontweight='bold')
ax.set_xlabel('Фаза', fontsize=18)
ax.set_ylabel('Массовая доля', fontsize=18)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(PHASE_INDICES)
ax.set_ylim(bottom=0)
# Добавляем значения над столбцами
# for i, (bar, mean) in enumerate(zip(bars, pareto_means)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
#             f'{mean:.3f}', ha='center', va='bottom', fontsize=12, rotation=0)

# 4. Распределение Дирихле
print("Генерация образцов для распределения Дирихле...")
dirichlet_samples = generate_distribution_samples(_dirichlet_weights, N_PHASES, N_SAMPLES)
dirichlet_means = np.mean(dirichlet_samples, axis=0)
dirichlet_stds = np.std(dirichlet_samples, axis=0)

ax = axes[0, 1]
dirichlet_err_lower = np.clip(dirichlet_stds, 0, dirichlet_means)
bars = ax.bar(PHASE_INDICES, dirichlet_means, yerr=[dirichlet_err_lower, dirichlet_stds],
              color=colors[3], alpha=0.8, capsize=6, error_kw={'elinewidth': 2.5, 'capthick': 2})
ax.set_title('Дирихле (α∈[0.05, 0.8])', fontweight='bold')
ax.set_xlabel('Фаза', fontsize=18)
ax.set_ylabel('Массовая доля', fontsize=18)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(PHASE_INDICES)
ax.set_ylim(bottom=0)
# Добавляем значения над столбцами
# for i, (bar, mean) in enumerate(zip(bars, dirichlet_means)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
#             f'{mean:.3f}', ha='center', va='bottom', fontsize=12, rotation=0)

# Убираем общий заголовок (по требованию)
# Настраиваем расположение подграфиков
plt.tight_layout()
plt.subplots_adjust(top=0.96, hspace=0.25, wspace=0.25)  # Увеличено расстояние между графиками

print("Сохранение графиков...")
plt.savefig('phase_distributions.png', dpi=400, bbox_inches='tight')  # Увеличено DPI
plt.savefig('phase_distributions.pdf', bbox_inches='tight', format='pdf')

plt.show()

print("Готово! Графики сохранены как 'phase_distributions.png/pdf'")