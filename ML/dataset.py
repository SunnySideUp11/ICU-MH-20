import numpy as np
from sklearn.model_selection import KFold
from utils import get_person_data, get_step_data


class BaseDataset:
    def __init__(self, view_num=[1, 2, 3], dims=3):
        self.d = dims
        self._load_data(view_num)
        
    def _load_data(self, view_num):
        self.data = {}
        for v in view_num:
            X = np.concatenate([get_step_data(v, s, self.d)[0] for s in range(1, 8)], dtype=object)
            Y = np.concatenate([get_step_data(v, s, self.d)[1] for s in range(1, 8)], dtype=object)   
            self.data[v] = (X, Y.astype("int"))
    
    def generate_5_fold_data(self):
        train_test_split_data = {}
        k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
        for view in self.data.keys():
            view_data = []
            for train_idx, test_idx in k_fold.split(self.data[view][0]):
                x_train, y_train = self.data[view][0][train_idx], self.data[view][1][train_idx]
                x_test, y_test = self.data[view][0][test_idx], self.data[view][1][test_idx]
                view_data.append([(x_train, y_train), (x_test, y_test)])
            train_test_split_data[view] = view_data
        
        return train_test_split_data            
    

class CrossObjectDataset:
    def __init__(self, view_num=[1, 2, 3], dims=3, index=None):
        self.d = dims
        self._load_data(view_num)
        self._5_fold_index = index if index else self._5_fold()
    
    def _load_data(self, view_num):
        self.data = {}
        for v in view_num:
            self.data[v] = np.array([get_person_data(v, p, self.d) for p in range(1, 21)], dtype=object)
    
    def _5_fold(self):
        index = np.arange(20)
        np.random.shuffle(index)
        _5_fold_index = [
            (np.concatenate((index[0: i], index[i + 4: 20])), index[i: i + 4]) for i in range(0, 20, 4)
        ]
        return _5_fold_index
    
    def generate_5_fold_data(self):
        train_test_split_data = {}
        for view in self.data.keys():
            view_data = []
            for train_idx, test_idx in self._5_fold_index:
                train_data, test_data = self.data[view][train_idx], self.data[view][test_idx]
                train_x, train_y = self._concatenate(train_data)
                test_x, test_y = self._concatenate(test_data)
                view_data.append(((train_x, train_y), (test_x, test_y)))
            train_test_split_data[view] = view_data
        return train_test_split_data
    
    def _concatenate(self, data):
        x, y = data[:, 0], data[:, 1]
        return np.concatenate(x), np.concatenate(y)
    
