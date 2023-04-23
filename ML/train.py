import os

import joblib
import numpy as np
from dataset import BaseDataset, CrossObjectDataset
from utils import get_dir

from ml import myModel

np.random.seed(3)

# MODELS = ["KNN", "RF", "SVC", "MLP"]
MODELS = ["MLP"]



def main(view_num=[1, 2, 3], cross=True, dims=3, index=None):
    exp_type = "cross" if cross else "base"
    outdir = get_dir("./output", exp_type, f"{dims}d")
    for i in view_num:
        os.makedirs(os.path.join(outdir, f"view{i}"))
        
    dataset = CrossObjectDataset(view_num=view_num, dims=dims, index=index) if cross else BaseDataset(view_num=view_num, dims=dims)
    for view, data in dataset.generate_5_fold_data().items():
        results = {"KNN": [], "RF": [], "SVC": [], "MLP": []}
        for idx, (train_data, test_data) in enumerate(data):
            for model_name in MODELS:
                print(f"{view}-{idx}: {model_name}")
                model = myModel(model_name)
                model.train(train_data)
                res = model.test(test_data)
                results[model_name].append(res)
        for k, v in results.items():
            joblib.dump(v, os.path.join(outdir, f"view{view}", f"{k}.pth"))

            
if __name__ == '__main__':
    index = joblib.load("./index.pth")
    main(cross=False, view_num=[1, 2, 3], dims=3, index=index)
    main(cross=True, view_num=[1, 2, 3], dims=3, index=index)

    