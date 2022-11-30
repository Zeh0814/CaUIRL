# From https://github.com/xternalz/WideResNet-pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x_mask):
        x = x_mask['x']
        mask = x_mask['mask']
        if not self.equalInOut:
            if mask is not None:
                x = dar_bn(self.bn1, x, mask)
                x = self.relu1(x)
            else:
                x = self.relu1(self.bn1(x))

        else:
            if mask is not None:
                out = dar_bn(self.bn1, x, mask)
                out = self.relu1(out)
            else:
                out = self.relu1(self.bn1(x))
        if mask is not None:
            out = self.conv1(out if self.equalInOut else x)
            out = self.relu2(dar_bn(self.bn2, out, mask))
        else:
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return {'x': torch.add(x if self.equalInOut else self.convShortcut(x), out), 'mask': mask}


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        return self.layer({'x': x, 'mask': mask})


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, mask=None):
        if mask is not None:
            out = self.conv1(x)
            out = self.block1(out, mask=mask)
            out = self.block2(out['x'], mask=mask)
            out = self.block3(out['x'], mask=mask)
            out = dar_bn(self.bn1, out['x'], mask)
            out = self.relu(out)

            out = F.avg_pool2d(out, 8)
            fea_out = out.view(-1, self.nChannels)
            out = self.fc(fea_out)
            return out, fea_out
        else:
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out['x'])
            out = self.block3(out['x'])
            out = self.relu(self.bn1(out['x']))

            out = F.avg_pool2d(out, 8)
            fea_out = out.view(-1, self.nChannels)
            out = self.fc(fea_out)
            return out, fea_out


def dar_bn(bn_layer, x, noise_mask):
    """
  Applies DAR-BN normalization to a 4D input (a mini-batch of 2D inputs with
  additional channel dimension)
  bn_layer : torch.nn.BatchNorm2d
  Batch norm layer operating on activation maps of natural images
  x : torch.FloatTensor of size: (N, C, H, W)
  2D activation maps obtained from both natural images and noise images
  noise_mask: torch.BoolTensor of size: (N)
  Boolean 1D tensor that indicates which activation map is obtained from noise
  """
    # Batch norm for activation maps of natural images
    out_natural = bn_layer(x[torch.logical_not(noise_mask)])
    # Batch norm for activation maps of noise images
    # Do not compute gradients for this operation
    with torch.no_grad():
        adaptive_params = {"weight": bn_layer.weight,
                           "bias": bn_layer.bias,
                           "eps": bn_layer.eps}
    out_noise = batch_norm_with_adaptive_parameters(x[noise_mask], adaptive_params)
    # Concatenate activation maps in original order
    out = torch.empty_like(torch.cat([out_natural, out_noise], dim=0))
    out[torch.logical_not(noise_mask)] = out_natural
    out[noise_mask] = out_noise
    return out


def batch_norm_with_adaptive_parameters(x_noise, adaptive_parameters):
    """
  Applies batch normalization to x_noise according to the adaptive_parameters
  x_noise : torch.FloatTensor of size: (N, C, H, W)
  2D activation maps obtained from noise images only
  adaptive_parameters:
  a dictionary containing:
  weight: scale parameter for the adaptive affine
  bias: bias parameter for the adaptive affine
  eps: a value added to the denominator for numerical stability.
  """

    # Calculate mean and variance for the noise activations batch per channel
    mean = x_noise.mean([0, 2, 3])
    var = x_noise.var([0, 2, 3], unbiased=False)

    # Normalize the noise activations batch per channel
    out = (x_noise - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + adaptive_parameters["eps"]))
    # Scale and shift using adaptive affine per channel
    out = out * adaptive_parameters["weight"][None, :, None, None] + adaptive_parameters["bias"][None, :, None, None]

    return out


class WideResNet_copy(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet_copy, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
