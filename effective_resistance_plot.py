# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pos = {'L0': np.array([0.27623617, 0.88746711]),
 'L1': np.array([0.17756036, 0.65809066]),
 'L2': np.array([0.41241181, 0.69533594]),
 'L3': np.array([0.57694415, 0.82418349]),
 'L4': np.array([0.47217663, 0.9807163 ]),
 'R0': np.array([-0.40891482, -0.68843508]),
 'R1': np.array([-0.29782751, -0.44243271]),
 'R2': np.array([-0.13644417, -0.6280399 ]),
 'R3': np.array([-0.04935565, -0.83658502]),
 'R4': np.array([-0.16424932, -1.        ]),
 'R5': np.array([-0.36122932, -0.92226245]),
 0: np.array([-0.03072258,  0.42894798]),
 1: np.array([-0.19102918,  0.16887793]),
 2: np.array([-0.27555657, -0.12586423])}

path_length = 3
clique_1 = nx.cycle_graph(5)
clique_2 = nx.cycle_graph(6)
path = nx.path_graph(path_length)
cmap = "inferno"

g = nx.union(clique_1, clique_2, rename=("L", "R"))
g = nx.union(g, path)
g.add_edge("L1", 0)
g.add_edge("R1", path_length - 1)

edge_width = 1.2
edge_alpha = 0.8
edge_color = 'k'

plt.figure(figsize=(20, 10))
reference_node = "L2"

pos_remove_reference_node = pos.copy()
pos_remove_reference_node[reference_node] = pos["L1"]
pos_collapse_reference_node = {node: pos[reference_node] for node, i in pos.items()}

def get_resistances(graph, reference_node):
    resistances = {}

    for node in graph.nodes:
        if node != reference_node:
            resistances[node] = nx.resistance_distance(graph, reference_node, node)
        else:
            resistances[node] = 0
    return resistances

def resistances_to_commute_times(resistances, graph):
    n_edges = graph.number_of_edges()
    return {node : res * 2 * n_edges for node, res in resistances.items()}

resistances = get_resistances(g, reference_node)
commute_times = resistances_to_commute_times(resistances, g)
nodes = g.nodes()
colors = [commute_times[n] for n in nodes]

font = {'family': 'Verdana',
        'weight': 'bold',
        'size': 14,
        }
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

vmin, vmax = min(colors), max(colors)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
fig.tight_layout()
ax = plt.subplot(131)
#pos = nx.spring_layout(g)
ec = nx.draw_networkx_edges(g, pos, alpha=edge_alpha, width=edge_width, edge_color=edge_color)
nc_base = nx.draw_networkx_nodes(g, pos_remove_reference_node, nodelist=nodes, node_color=colors,
                            node_size=150, cmap=cmap, vmin=vmin, vmax=vmax)
# nc_base = nx.draw_networkx_nodes(g, pos_collapse_reference_node, nodelist=nodes, node_color=[0 for _ in nodes],
#                             node_size=150, cmap=cmap, vmin=vmin, vmax=vmax, node_shape="^")
plt.scatter(pos[reference_node][0], pos[reference_node][1], s=200, marker="*", color="k")


# plt.colorbar(nc_base)
plt.axis('off')

A = nx.adjacency_matrix(g).todense()
A_pow = np.linalg.matrix_power(A, 3)
np.fill_diagonal(A_pow, 0)
A_pow[A_pow > 1] = 1

def get_graph_from_adj(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    return gr

g2 = get_graph_from_adj(A_pow)
mapping = {}

mapping = {i: node for i, (node, _) in enumerate(g.nodes.items())}
g2 = nx.relabel_nodes(g2, mapping)
resistances = get_resistances(g2, reference_node)
commute_times = resistances_to_commute_times(resistances, g2)
nodes = g2.nodes()
colors = [commute_times[n] for n in nodes]
ax.set_title("Base Graph", font)

ax = plt.subplot(132)
ec = nx.draw_networkx_edges(g2, pos, alpha=edge_alpha, width=edge_width, edge_color=edge_color)
nc = nx.draw_networkx_nodes(g2, pos_remove_reference_node, nodelist=nodes, node_color=colors,
                            node_size=150, cmap=cmap, vmin=vmin, vmax=vmax)
# nc = nx.draw_networkx_nodes(g2, pos_collapse_reference_node, nodelist=nodes, node_color=[0 for _ in nodes],
#                             node_size=150, cmap=cmap, vmin=vmin, vmax=vmax, node_shape="^")
plt.scatter(pos[reference_node][0], pos[reference_node][1], s=200, marker="*", color="k")
plt.axis('off')
#plt.colorbar(nc_base)
# plt.show()

g3 = g.copy()
g3.add_edge("L3", "R4")
# g3.add_edge("L4", "R2")
g3.add_edge("L1", 1)
resistances = get_resistances(g3, reference_node)
commute_times = resistances_to_commute_times(resistances, g3)
nodes = g3.nodes()
colors = [commute_times[n] for n in nodes]
ax.set_title("Spatially Rewired Graph", font)
ax = plt.subplot(133)
ec = nx.draw_networkx_edges(g3, pos, alpha=edge_alpha, width=edge_width, edge_color=edge_color)
nc = nx.draw_networkx_nodes(g3, pos_remove_reference_node, nodelist=nodes, node_color=colors,
                            node_size=150, cmap=cmap, vmin=vmin, vmax=vmax)
# nc = nx.draw_networkx_nodes(g3, pos_collapse_reference_node, nodelist=nodes, node_color=[0 for _ in nodes],
#                             node_size=150, cmap=cmap, vmin=vmin, vmax=vmax, node_shape="^")
plt.scatter(pos[reference_node][0], pos[reference_node][1], s=200, marker="*", color="k")
ax.set_title("Spectrally Rewired Graph", font)

cbar = plt.colorbar(nc_base)
cbar.ax.tick_params(labelsize=11)
cbar.ax.set_ylabel('Commute Time', font)
plt.axis('off')
plt.savefig("effective_resistance.pdf", bbox_inches='tight')

