import os
import ssl
import torch
import pandas as pd
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_scipy_sparse_matrix


class Facebook100(InMemoryDataset):
    r"""
    This dataset contains the Facebook networks (from a date in
    Sept. 2005) for 100 colleges and universities from the
    `"Social Structure of Facebook Networks"
    <https://arxiv.org/abs/1102.2166>`_ paper.
    Each dataset only includes
    intra-school links. Note that these are the full sets of links
    inside each school, ignoring isolated nodes; the datasets are
    not restricted to the largest connected component. The institutions
    are described in the paper.

    Each node (user) has the following one-hot encoded features:
    a student/faculty status, gender, major, minor, housing status, and year.
    One of these features can be chosen as the node label. Missing node labels
    are coded -1.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): name of the college/university (refer to the paper to
            see the list of names.)
        target (string): The node attribute to be used at the target node label.
            One of the 'status', 'gender', 'major', 'minor', 'housing', or 'year'.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://escience.rpi.edu/data/DA/fb100/'
    targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']

    def __init__(self, root, name, target, transform=None, pre_transform=None):
        self.name = name
        self.target = target
        assert target in self.targets
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return self.name + '.mat'

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        ssl._create_default_https_context = context

    def process(self):
        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names))
        features = pd.DataFrame(mat['local_info'][:, :-1], columns=self.targets)
        if self.target == 'year':
            features.loc[(features['year'] < 2004) | (features['year'] > 2009), 'year'] = 0
        y = torch.from_numpy(LabelEncoder().fit_transform(features[self.target]))
        if 0 in features[self.target].values:
            y = y - 1

        x = features.drop(columns=self.target).replace({0: pd.NA})
        x = torch.tensor(pd.get_dummies(x).values, dtype=torch.float)
        edge_index = from_scipy_sparse_matrix(mat['A'])[0]
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(y))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Facebook100-{self.name}()'