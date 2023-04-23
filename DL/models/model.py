from timm.models.helpers import load_checkpoint

from .dino import vitb8_dino, vits8_dino, vits16_dino
from .resnet import resnet101_torchvision

# from .timesformer import timesformer


__models__ = {
    "vits8_dino": vits8_dino,
    "vits16_dino": vits16_dino, 
    "vitb8_dino": vitb8_dino,
    "resnet101_tv": resnet101_torchvision,
}

def get_model(model_name, **kwargs):
    assert model_name in __models__, "There is no model for you"
    model =  __models__.get(model_name)
    return model(**kwargs)

def create_model(model_name,
              pretrained=False,
              checkpoint_path='',
              **kwargs):
    
    model = get_model(model_name, pretrained=pretrained, **kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model