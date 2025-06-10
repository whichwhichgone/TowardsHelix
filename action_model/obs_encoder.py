import torch
import torch.nn as nn


class SpatialSoftmaxPooling(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        # [batch_size, channels, height*width]
        features = x.view(batch_size, channels, -1)
        weights = torch.softmax(features / self.temperature, dim=2)
        weighted_features = torch.bmm(features, weights.transpose(1, 2))
        
        # [batch_size, channels, channels] -> [batch_size, channels]
        pooled_features = torch.diagonal(weighted_features, dim1=1, dim2=2)
        return pooled_features


class ObsImgEncoder(nn.Module):
    def __init__(self,
            backbone_model: nn.Module,
            input_shape=(3, 224, 224),
            resize_shape=None,
            crop_shape=None,
            noise_random=False,
            random_crop=True,
            use_group_norm=False,
            imagenet_norm=False,
            output_dim=None
        ):
        super().__init__()

        # images have been augmented in the data chunking part
        self.input_shape = input_shape
        if use_group_norm:
            backbone_model = self.replace_batchnorm_with_groupnorm(backbone_model)
        backbone_model.avgpool = nn.Identity()
        backbone_model.fc = nn.Identity()

        self.backbone = backbone_model
        self.spatial_softmax = SpatialSoftmaxPooling(temperature=1.0)
        self.out = nn.Linear(512, output_dim, bias=False)
    
    def replace_batchnorm_with_groupnorm(self, model, num_groups=32):
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                groups = min(num_groups, num_channels)
                while num_channels % groups != 0 and groups > 1:
                    groups -= 1
                setattr(model, name, nn.GroupNorm(num_groups=groups, num_channels=num_channels))
            elif len(list(child.children())) > 0:
                self.replace_batchnorm_with_groupnorm(child, num_groups)
        return model
    
    def forward(self, x):
        assert x.shape[1] == 3 and x.shape[2] == 224 and x.shape[3] == 224, "Input image must be 3x224x224"
        x = self.backbone(x)
        x = x.view(x.size(0), 512, 7, 7)
        x = self.spatial_softmax(x)

        # (batch_size, 512) -> (batch_size, output_dim)
        x = self.out(x)
        return x
