import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.wasp import build_wasp
from model.modules.decoder import build_decoder
from model.modules.backbone import build_backbone
import time


class unipose(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=15,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(unipose, self).__init__()
        self.stride = stride

        BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes

        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.wasp = build_wasp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)  # Adjusted to remove dataset dependency

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = F.interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)
        return x


# Load the pre-trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = unipose(num_classes=15).to(device)  # Example usage without the dataset parameter
model.load_state_dict(torch.load('/home/ps332/myViT/UniPose/UniPose_MPII.pth'))
model.eval()

# Create a dummy input tensor
input_tensor = torch.randn(1, 3, 368, 368).to(device)

# Warm up
for _ in range(10):
    _ = model(input_tensor)

# Measure throughput
num_tests = 100
start_time = time.time()
for _ in range(num_tests):
    _ = model(input_tensor)
total_time = time.time() - start_time

print(f"Average inference time per input: {total_time / num_tests:.6f} seconds")
print(f"Inference per second: {num_tests / total_time:.2f} inferences per second")