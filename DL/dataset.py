import glob
import os
from math import ceil

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, root, num_view, transform=None):
        self.root = root
        self.transform = transform
        
        # num_people = num_people if isinstance(num_people, (tuple, list)) else [num_people]
        self.image_path, self.image_class = self.merge_data(num_view)
            
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img = Image.open(self.image_path[index])
        label = self.image_class[index]
        if self.transform:
            img = self.transform(img)

        return img, label  
    
    def merge_data(self, num_view):
        image_path = []
        image_class = []
        for people in range(1, 21):
            for i in range(1, 8):
                class_ = i
                folder = os.path.join(self.root, f"view{num_view}", f"p{people}", str(class_))
                image_path += [os.path.join(folder, file) for file in os.listdir(folder)]
                image_class += [class_ - 1 for _ in range(len(os.listdir(folder)))]
        return image_path, image_class



class CrossObjectDataSet(Dataset):
    def __init__(self, root, num_view, num_people, transform=None):
        self.root = root
        self.transform = transform
        
        # num_people = num_people if isinstance(num_people, (tuple, list)) else [num_people]
        self.image_path, self.image_class = self.merge_data(num_view, num_people)
            
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        img = Image.open(self.image_path[index])
        label = self.image_class[index]
        if self.transform:
            img = self.transform(img)

        return img, label  
    
    def merge_data(self, num_view, num_people):
        image_path = []
        image_class = []
        for people in num_people:
            for i in range(1, 8):
                class_ = i
                folder = os.path.join(self.root, f"view{num_view}", f"p{people}", str(class_))
                image_path += [os.path.join(folder, file) for file in os.listdir(folder)]
                image_class += [class_ - 1 for _ in range(len(os.listdir(folder)))]
        return image_path, image_class
    
    
    
class MyDataset(Dataset):
    def __init__(self, root, num_view, num_people, transform=None, len_frames=16):
        super().__init__()
        self.root = root
        self.num_view = num_view
        self.num_people = num_people 
        
        self.transform = transform
        self.len_frames = len_frames
        # self.is_TimeSformer = is_TimeSformer
        self.data, self.labels = self._load_data()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    
    def _load_data(self):
        data, labels = [], []
        for p in self.num_people:
            for s in range(1, 8):
                folder = os.path.join(self.root, f"view{self.num_view}", f"p{p}", str(s))
                X, y = self._get_frames(folder, s)
                data.append(X)
                labels.append(y)
        data, labels = torch.cat(data, dim=0), torch.cat(labels)
        # if not self.is_TimeSformer:
        data = data.reshape(-1, 3, 16, 224, 224)
        return data, labels
    
    def _get_frames(self, folder, class_):
        jpgs = np.array(glob.glob(folder + "/*.jpg"))
        indexs = self._get_indexs(jpgs)
        X = []
        # X = [self._cat_frames(jpgs, idx).unsqueeze(0) for idx in indexs]
        for idx in indexs:
            frames = self._cat_frames(jpgs, idx)
            if len(frames) == 0:
                continue
            frames = frames.unsqueeze(0)
            X.append(frames)
        y = [class_ - 1 for _ in range(len(X))]
        return torch.cat(X, dim=0), torch.tensor(y)
    
    def _get_indexs(self, images: list):
        len_frames = self.len_frames
        num_frames = ceil(len(images) / len_frames)
        
        index = [i for i in range(len(images))]
        if len(index) == 0:
            return []
        index += [index[-1]] * (num_frames * len_frames - len(images))
        
        res = []
        for i in range(num_frames):
            res.append([index[j] for j in range(len_frames * i, len_frames * (i + 1))])
        
        return res
        
        
    def _cat_frames(self, jpgs, index):
        frames = []
        if len(index) == 0:
            return frames
        for jpg in jpgs[index]:
            img = Image.open(jpg)
            if self.transform:
                img = self.transform(img)
            frames.append(img.unsqueeze(0))
        if frames:
            frames = torch.cat(frames, dim=0)
        return frames
    
    
    
class DatasetFor3D(Dataset):
    def __init__(self, root, num_view, num_people, transform=None, len_frames=16):
        super().__init__()
        self.root = root
        self.num_view = num_view
        self.num_people = num_people 
        
        self.transform = transform
        self.len_frames = len_frames
        self.data = self._load_data()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_paths = self.data[index]
        label = torch.tensor(int(img_paths[0].split("\\")[-2]) - 1, dtype=torch.long) 
        data = self._get_frames(img_paths)
        
        return data, label
    
    def _load_data(self):
        data = []
        for p in self.num_people:
            for s in range(1, 8):
                folder = os.path.join(self.root, f"view{self.num_view}", f"p{p}", str(s))
                jpgs = np.array(glob.glob(folder + "/*.jpg"))
                if len(jpgs) != 0:
                    indexs = self._get_indexs(jpgs)
                    data += [jpgs[idx] for idx in indexs]
    
        return data
    
    def _get_frames(self, paths):
        frames = []
        for path in paths:
            img = Image.open(path)
            if self.transform:
                img = self.transform(img)
            frames.append(img.unsqueeze(1))
        return torch.cat(frames, dim=1)
    
    def _get_indexs(self, images: list):
        len_frames = self.len_frames
        num_frames = ceil(len(images) / len_frames)
        
        index = [i for i in range(len(images))]
        # print(index)
        index += [index[-1]] * (num_frames * len_frames - len(images))
        
        res = []
        for i in range(num_frames):
            res.append([index[j] for j in range(len_frames * i, len_frames * (i + 1))])
        
        return res