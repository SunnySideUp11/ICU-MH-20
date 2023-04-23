import os

import numpy as np
import pandas as pd


def get_dir(root, *paths, inc=True):
    outdir = os.path.join(root, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count += 1
            outdir_inc = outdir + '-' + str(count)
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def get_csv_data(path: str, left: bool = True):
    df = pd.read_csv(path)
    x = "Left" if left else "Right"
    csv_data = [] 
    for item in df[x]:
        data = item[1: -1]
        if len(data) == 0:
            csv_data.append([0 for _ in range(63)])
        else:
            data = [float(i) for i in data.split(", ")]
            csv_data.append(data)
    return csv_data

def get_csv_data_2d(path: str, left: bool = True):
    data = get_csv_data(path, left)
    data = np.array(data)
    new_data = np.zeros(shape=(data.shape[0], 42))
    for i in range(new_data.shape[0]):
        for j in range(new_data.shape[1]):
            new_data[i, j] = data[i, j + j // 2]  
    return new_data.tolist()

def get_person_data(view_num: int, p_idx: int, dims: int = 3):
    root = "../image/"
    _x, _y =[], []
    for i in range(1, 8):
        path_csv = os.path.join(root, f"view{view_num}", f"p{p_idx}", str(i), "data.csv")
        left = get_csv_data(path_csv, left=True) if dims == 3 else get_csv_data_2d(path_csv, left=True)
        right = get_csv_data(path_csv, left=False) if dims == 3 else get_csv_data_2d(path_csv, left=False)
        x = np.concatenate([left, right], axis=1)
        y = np.array([i for _ in range(len(left))])
        _x.append(x)
        _y.append(y)
    return np.concatenate(_x), np.concatenate(_y)

def get_step_data(view_num: int, step_num: int, dims: int = 3):
    root = "../image/"
    _x, _y =[], []
    for i in range(1, 21):
        path_csv = os.path.join(root, f"view{view_num}", f"p{i}", str(step_num), "data.csv")
        left = get_csv_data(path_csv, left=True) if dims == 3 else get_csv_data_2d(path_csv, left=True)
        right = get_csv_data(path_csv, left=False) if dims == 3 else get_csv_data_2d(path_csv, left=False)
        x = np.concatenate([left, right], axis=1)
        y = np.array([step_num for _ in range(len(left))])
        _x.append(x)
        _y.append(y)
    return np.concatenate(_x), np.concatenate(_y)