import matplotlib.pyplot as plt
import networkx as nx

new_nodes = ["Input\n(1, 4001)",
             "Conv1\n(36, 4001)",
             "ReLU 1",
             "MaxPool\n(36, 1000)",
             "Conv2\n(72, 1000)",
             "ReLU ",
             "Transformer\n(72, 1000)",
             "FC\n(72000, 11)",
             "Output\n(11)"]



# Создание графа
G = nx.DiGraph()

# Добавление узлов
G.add_node(new_nodes[0], pos=(0, 0))
G.add_node(new_nodes[1], pos=(1, 0))
G.add_node(new_nodes[2], pos=(2, 0))
G.add_node(new_nodes[3], pos=(3, 0))
G.add_node(new_nodes[4], pos=(4, 0))
G.add_node(new_nodes[5], pos=(5, 0))
G.add_node(new_nodes[6], pos=(6, 0))
G.add_node(new_nodes[7], pos=(7, 0))
G.add_node(new_nodes[8], pos=(8, 0))

# Добавление ребер
edges = [
    (new_nodes[0], new_nodes[1]),
    (new_nodes[1], new_nodes[2]),
    (new_nodes[2], new_nodes[3]),
    (new_nodes[3], new_nodes[4]),
    (new_nodes[4], new_nodes[5]),
    (new_nodes[5], new_nodes[6]),
    (new_nodes[6], new_nodes[7]),
    (new_nodes[7], new_nodes[8])
]

G.add_edges_from(edges)

# Получение позиций узлов
pos = nx.get_node_attributes(G, 'pos')

# Рисование графа
plt.figure(figsize=(12, 4))
nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
plt.title("Структура нейронной сети для анализа дифрактограмм")
plt.show()