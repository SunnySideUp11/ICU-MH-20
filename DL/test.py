import os

import joblib
import torch
import torch.nn as nn
from dataset import BaseDataset, CrossObjectDataSet, DatasetFor3D, MyDataset
from models import create_model
from models.resnet import resnet3d101
from timm import utils
from torch.utils.data import random_split
from torchvision import transforms
from tqdm import tqdm
from utils import MetricCalculator

# from pytorchvideo.models import create_resnet


view_num = 3
index = joblib.load("./index.pth")
idx = index[-1]

device = torch.device("cuda:0")

model = create_model("vitb8_dino")
# weights = torch.load("./output/train/latest_hand-vitb8_dino/0921-0637/model_best.pth.tar") # v1
# weights = torch.load("./output/train/latest_hand-vitb8_dino/1006-2343/model_best.pth.tar") # v2
weights = torch.load("./output/train/latest_hand-vitb8_dino/1010-2211/model_best.pth.tar") # v3

model.load_state_dict(weights["state_dict"])
model.to(device)

data_transforms = transforms.Compose([
    transforms.CenterCrop(1080),
    transforms.Resize(224),
    transforms.ToTensor()
])

root = "../data/image"
# dataset = BaseDataset(root=root, num_view=view_num, transform=data_transforms)
# n_val = int(len(dataset) * 0.2)
# n_tarin = len(dataset) - n_val
# train_dataset, eval_dataset = random_split(dataset, [n_tarin, n_val])


eval_dataset = CrossObjectDataSet(root=root,
                                num_view=view_num, 
                                num_people=idx[1],
                                transform=data_transforms)

eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=8,
                                                shuffle=False)




labels, predicts = [], []

with torch.no_grad():
    for input, target in tqdm(eval_dataloader, ncols=120):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            predict = output.argmax(dim=1)
            
            labels += target.tolist()
            predicts += predict.tolist()

joblib.dump({"labels": labels, "predicts": predicts}, f"./output/test/vitb8_v{view_num}_gen.pkl")



        