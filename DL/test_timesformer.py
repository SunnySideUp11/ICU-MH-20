import os

import joblib
import torch
import torch.nn as nn
from dataset import DatasetFor3D
from models.timesformer import timesformer
from timm import utils
from torchvision import transforms
from tqdm import tqdm

index = joblib.load("./index.pth")
idx = index[-1]

device = torch.device("cuda:0")

model = timesformer()

view_num = 2
# weights = torch.load("./output/train/3d_hand-timesformer/1027-1615/model_best.pth.tar") # v1
weights = torch.load("./output/train/3d_hand-timesformer/1102-1104/model_best.pth.tar") # v2
# weights = torch.load("./output/train/3d_hand-timesformer/1030-2307/model_best.pth.tar") # v3

model.load_state_dict(weights["state_dict"])
model.to(device)

data_transforms = transforms.Compose([
    transforms.CenterCrop(1080),
    transforms.Resize(448),
    transforms.ToTensor()
])

root = "../data/image"

train_dataset = DatasetFor3D(root=root,
                              num_view=view_num,
                              num_people=index[0],
                              transform=data_transforms)

eval_dataset = DatasetFor3D(root=root,
                             num_view=view_num,
                             num_people=index[1],
                             transform=data_transforms)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=1,
                                                shuffle=False)

eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=1,
                                                shuffle=False)


labels, predicts = [], []

with torch.no_grad():
    # for input, target in tqdm(train_dataloader, ncols=120):
    for input, target in tqdm(eval_dataloader, ncols=120):
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            predict = output.argmax(dim=1)
            
            labels += target.tolist()
            predicts += predict.tolist()
            

# joblib.dump({"labels": labels, "predicts": predicts}, f"./output/test/latest/timesformer_v{view_num}_per.pkl")
joblib.dump({"labels": labels, "predicts": predicts}, f"./output/test/latest/timesformer_v{view_num}_gen.pkl")



        