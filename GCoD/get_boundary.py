import numpy as np

def my_get_boundary(n_subgraphs, class_graphs, groups):

    for i in range(groups-1):
        class_graphs.extend(class_graphs)

    n_classes = []
    for i in range(len(class_graphs)):
        if i == 0:
            count = sum(n_subgraphs[:class_graphs[i]])
            n_classes.append(count)
        else:
            new_count = sum(n_subgraphs[sum(class_graphs[:i]):sum(class_graphs[:i+1])])
            count += new_count
            n_classes.append(count)

    n_groups = []
    par = int(len(n_classes)/groups)
    for i in range(groups):
        n_groups.append(n_classes[par*(i+1)-1])

    return n_subgraphs, n_classes, n_groups

if __name__ == '__main__':
    n_subgraphs = [694, 306, 319, 114, 215, 685, 291, 317, 158, 228]
    class_graphs = [1, 2, 2]
    groups = 2
    n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, groups)
    print(n_subgraphs)
    print(n_classes)
    print(n_groups)