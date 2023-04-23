import os

import torch
import torch.nn as nn


class Vits_dino(nn.Module):
    def __init__(self, patch=8, arch="s", pretrained=True, num_classes=7):
        super(Vits_dino, self).__init__()
        self.arch = arch
        self.vits = torch.hub.load('facebookresearch/dino:main', f'dino_vit{arch}{patch}')
        # self.classifier = nn.Linear(384, num_classes)
        in_features = 384 if arch == "s" else 768
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, num_classes)
        )
        
        self._init_weights(pretrained=pretrained, patch=patch)
        
    def _init_weights(self, pretrained=True, patch=8):
        if pretrained:
            filename = f"dino_deitsmall{patch}_pretrain.pth" if self.arch == "s" else f"dino_vitbase{patch}_pretrain.pth"
            weights_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "pretrained_weights", 
                                    filename)
            self.vits.load_state_dict(torch.load(weights_file))
            for p in self.vits.parameters():
                p.requires_grad = False
        for layer in self.classifier:
            layer.weight.data.normal_(mean=0.0, std=0.01)
            layer.bias.data.zero_()
        # self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        # self.classifier.bias.data.zero_()     

    def forward(self, x):
        x = self.vits(x)
        return self.classifier(x)



def vits8_dino(pretrained=True, num_classes=7):
    return Vits_dino(patch=8, pretrained=pretrained, num_classes=num_classes)

def vits16_dino(pretrained=True, num_classes=7):
    return Vits_dino(patch=16, pretrained=pretrained, num_classes=num_classes)

def vitb8_dino(pretrained=True, num_classes=7):
    return Vits_dino(patch=8, arch="b", pretrained=pretrained, num_classes=num_classes)


if __name__ == "__main__":
    print(vits16_dino())
    # print(os.getcwd())
    
    
    
