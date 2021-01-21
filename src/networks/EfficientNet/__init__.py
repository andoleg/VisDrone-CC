from .network import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

from torch.nn import Module


class EfficientNetModel(Module):
    def __init__(self, model_name, weights_path=None, advprop=False,
                 in_channels=3, num_classes=1, pretrained=True, **override_params):
        super().__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, weights_path, advprop=advprop,
                                                      in_channels=in_channels, num_classes=num_classes,
                                                      **override_params)
        else:
            self.model = EfficientNet.from_name(model_name, num_classes=num_classes, in_channels=in_channels,
                                                **override_params)

    def forward(self, inputs):
        return self.model(inputs)
