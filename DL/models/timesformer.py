import torch.nn as nn
from timesformer.models.vit import TimeSformer


def timesformer(img_size=448, num_classes=7, num_frames=16, attention_type="divided_space_time", pretrained=True):
    if pretrained:
        model = TimeSformer(img_size=img_size, num_classes=400, num_frames=num_frames, 
                            attention_type=attention_type, 
                            pretrained_model="./models/pretrained_weights/TimeSformer_divST_16x16_448_K400.pyth")

        for n, m in model.model.named_children():
            if n != "head":
                for p in m.parameters():
                    p.requires_grad = False
        in_features = model.model.head.in_features
        model.model.head = nn.Linear(in_features, num_classes)
        return model