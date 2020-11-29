from torch import nn

from torchvision.models import resnet34, resnet50


BACKONE_HEAD = {
    'resnet34': (resnet34, (512, 256)),
    'resnet50': (resnet50, (2048, 512))
}


class Classifier(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=100):
        super().__init__()

        backbone_name, head_dimensions = BACKONE_HEAD[backbone]
        self.backbone = nn.Sequential(*list(backbone_name(pretrained=True).children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(in_features=head_dimensions[0], out_features=head_dimensions[1], bias=False),
            nn.Dropout(p=.5),
            nn.Linear(in_features=head_dimensions[1], out_features=num_classes, bias=False),
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out.view(out.size()[0], -1))
        return out
