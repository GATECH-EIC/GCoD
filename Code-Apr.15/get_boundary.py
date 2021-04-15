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

def my_get_bd(n_group, n_class, bd1):
    bd2 = []
    now = 0
    for _ in range(n_group):
        for nc in n_class:
            tot = 0
            for __ in range(nc):
                tot += bd1[now]
                now += 1
            bd2.append(tot)
    # print(bd2)
    bd3 = []
    now = 0
    for _ in range(n_group):
        tot = 0
        for nc in n_class:
            for __ in range(nc):
                tot += bd1[now]
                now += 1
        bd3.append(tot)
    # print(bd3)
    for i in range(1, len(bd1)):
        bd1[i] += bd1[i - 1]
    for i in range(1, len(bd2)):
        bd2[i] += bd2[i - 1]
    for i in range(1, len(bd3)):
        bd3[i] += bd3[i - 1]

    return bd1, bd2, bd3

if __name__ == '__main__':
    n_subgraphs = [694, 306, 319, 114, 215, 685, 291, 317, 158, 228]
    class_graphs = [1, 2, 2]
    groups = 2
    n_subgraphs, n_classes, n_groups = my_get_boundary(n_subgraphs, class_graphs, groups)
    print(n_subgraphs)
    print(n_classes)
    print(n_groups)