import networkx as nx
from matplotlib import pyplot, patches
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[], save_path='./adj_visual', save_name='trial'):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(15, 15)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)

    font_board = 2
    ax.spines['bottom'].set_linewidth(font_board)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(font_board)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_linewidth(font_board)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(font_board)
    ax.spines['right'].set_color('black')

    plt.tight_layout()
    if not osp.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(osp.join(save_path, save_name+'.pdf'))



from scipy import io

# def delete_first_three_lines(filename):
#   A = io.mmread(filename)
#   print(A)
#   comments = '%%MatrixMarket matrix coordinate real symmetric'
#   dges = nx.read_edgelist('./pretrain/CiteSeer_adj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
#   # lines = (line if isinstance(line, str) else line.decode(encoding) for line in filename)
#   for line in filename:
#     if isinstance(line, str):
#       print(line)
#     else:
#       print(line.decode(encoding))
#   exit()

# delete_first_three_lines('./pretrain/CiteSeer_adj.mtx')

# A = np.load("./adj_visual/origin_adj1.npy")
# A = 1 - A
# G = nx.from_numpy_matrix(A)

# A = io.mmread("./adj_visual/socfb-Caltech36.mtx")
# G = nx.from_scipy_sparse_matrix(A)

if not osp.exists('./adj_visual'):
    os.mkdir('./adj_visual')

### CiteSeer

# G = nx.Graph()
# comments = '%%MatrixMarket matrix coordinate real symmetric'
# edges = nx.read_edgelist('./pretrain/CiteSeer_adj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='CiteSeer_metis')

# G = nx.Graph()
# edges = nx.read_edgelist('./pretrain/CiteSeer_oriadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='CiteSeer_origin')

# G = nx.Graph()
# edges = nx.read_edgelist('./pretrain/CiteSeer_newadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='CiteSeer_newadj')

### Caltech36
G = nx.Graph()
comments = '%%MatrixMarket matrix coordinate real symmetric'
edges = nx.read_edgelist('./pretrain/Caltech36_adj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
G.add_edges_from(edges.edges())
num_nodes = len(list(G.nodes))
print(num_nodes)
nodes_list = np.array(list(G.nodes))
draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Caltech36_metis')

G = nx.Graph()
edges = nx.read_edgelist('./pretrain/Caltech36_oriadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
G.add_edges_from(edges.edges())
num_nodes = len(list(G.nodes))
print(num_nodes)
nodes_list = np.array(list(G.nodes))
draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Caltech36_origin')

G = nx.Graph()
edges = nx.read_edgelist('./pretrain/Caltech36_newadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
G.add_edges_from(edges.edges())
num_nodes = len(list(G.nodes))
print(num_nodes)
nodes_list = np.array(list(G.nodes))
draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Caltech36_newadj')


### Cora
# G = nx.Graph()
# comments = '%%MatrixMarket matrix coordinate real symmetric'
# edges = nx.read_edgelist('./pretrain/Cora_adj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Cora_metis')

# G = nx.Graph()
# edges = nx.read_edgelist('./pretrain/Cora_oriadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Cora_origin')

# G = nx.Graph()
# edges = nx.read_edgelist('./pretrain/Cora_newadj.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/', save_name='Cora_newadj')

### test partition
# G = nx.Graph()
# comments = '%%MatrixMarket matrix coordinate real symmetric'
# edges = nx.read_edgelist('./partition/before.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/test/', save_name='before')

# G = nx.Graph()
# edges = nx.read_edgelist('./partition/after.mtx', comments=comments, nodetype=int, data=(('weight', float),))
# G.add_edges_from(edges.edges())
# num_nodes = len(list(G.nodes))
# print(num_nodes)
# nodes_list = np.array(list(G.nodes))
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/test/', save_name='after')


exit()

# np.random.shuffle(nodes_list)
# draw_adjacency_matrix(G, nodes_list, save_path='./adj_visual/Caltech/', save_name='random_order')


def divide_communities(G, nodes_partitions, alg='greedy_modularity'):

  if alg == 'greedy_modularity':
    # assign communities (greedy_modularity)
    from collections import defaultdict
    nodes_partitions = defaultdict(list)

    from networkx.algorithms.community.modularity_max import greedy_modularity_communities
    communities_gmc = greedy_modularity_communities(G)

    communities_sum = 0

    for i, community_gmc in enumerate(communities_gmc):
      communities_sum += 1
      print("Community ", communities_sum)
      print(community_gmc)
      print("Total number of nodes: ", len(community_gmc), "\n")
      nodes_partitions[i].extend(community_gmc)

    print("Total number of communities: ", communities_sum)

  elif alg == 'girvan_newman':
    # assign communities (girvan_newman)
    from networkx.algorithms.community.centrality import girvan_newman
    communities_iter = girvan_newman(G)

    communities_sum = 0
    communities_gn = []

    i = 0
    for community_gn in next(communities_iter):
      communities_sum += 1
      communities_gn.append(community_gn)
      print("Community ", communities_sum)
      print(community_gn)
      print("Total number of nodes: ", len(community_gn), "\n")
      nodes_partitions[i].extend(community_gn)
      i += 1

    print("Total number of communities: ", communities_sum)

  elif alg == 'louvain':
    import community
    louvain_community_dict = community.best_partition(G)
    print(louvain_community_dict)

    for node_index, comm_id in louvain_community_dict.items():
      nodes_partitions[comm_id].append(node_index)


  nodes_partitions = nodes_partitions.values()
  return nodes_partitions




from collections import defaultdict
nodes_partitions = defaultdict(list)

nodes_partitions = divide_communities(G, nodes_partitions, alg='greedy_modularity')

nodes_dorm_ordered = [node for dorm in nodes_partitions for node in dorm]
print(len(partition))
draw_adjacency_matrix(G, nodes_dorm_ordered, [nodes_partitions], ["blue"], save_path='./adj_visual/Caltech/', save_name='greedy_modularity'+'_order')


##################

# import numpy as np
# from collections import defaultdict

# def assignmentArray_to_lists(assignment_array):
#     by_attribute_value = defaultdict(list)
#     for node_index, attribute_value in enumerate(assignment_array):
#         by_attribute_value[attribute_value].append(node_index)
#     return by_attribute_value.values()

# # Load in array which maps node index to dorm number
# # Convert this to a list of lists indicating dorm membership
# dorm_assignment = np.genfromtxt("caltech_dorms_blanksInferred.txt", dtype="u4")
# dorm_lists = assignmentArray_to_lists(dorm_assignment)

# # Create a list of all nodes sorted by dorm, and plot
# # adjacency matrix with this ordering
# nodes_dorm_ordered = [node for dorm in dorm_lists for node in dorm]
# draw_adjacency_matrix(G, nodes_dorm_ordered, [dorm_lists],["blue"])

##################

