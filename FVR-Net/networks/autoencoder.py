import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time
import tools

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # if stride == 2:
        #     stride = (1, 2, 2)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        # print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        # self.avgpool = nn.AvgPool3d(
        #     (last_duration, last_size, last_size), stride=1)
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        # self.avgpool = nn.AvgPool3d((1, 10, 18), stride=1)
        self.conv2 = nn.Conv3d(in_channels=2048, out_channels=128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)

        # self.fc_mu = nn.Linear(cardinality * 32 * block.expansion, 128)
        # self.fc_logvar = nn.Linear(cardinality * 32 * block.expansion, 128)

        self.attention = nn.Sequential(
            nn.BatchNorm3d(2048),
            nn.Conv3d(in_channels=2048, out_channels=1024, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        """ Decoder starts here """
        self.deconv1 = nn.ConvTranspose3d(in_channels=2048, out_channels=1024,
                                          kernel_size=(2, 4, 4), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=1024, out_channels=512,
                                          kernel_size=(2, 4, 4), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.deconv3 = nn.ConvTranspose3d(in_channels=512, out_channels=256,
                                          kernel_size=(3, 4, 4), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.deconv4 = nn.ConvTranspose3d(in_channels=256, out_channels=64,
                                          kernel_size=(1, 4, 4), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.deconv5 = nn.ConvTranspose3d(in_channels=64, out_channels=1,
                                          kernel_size=(1, 4, 4), stride=(1, 2, 2),
                                          padding=(0, 1, 1))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def encode(self, x):
        h1 = self.relu(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        # x = self.sigmoid(x)

        # time.sleep(30)
        return x

    def forward(self, x):

        show_size = False
        # show_size = True
        if show_size:
            print('input shape {}'.format(x.shape))
            x = self.conv1(x)
            print('conv1 shape {}'.format(x.shape))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # x = self.conv_pool(x)
            print('maxpool shape {}'.format(x.shape))

            x = self.layer1(x)
            print('layer1 shape {}'.format(x.shape))
            # time.sleep(30)
            x = self.layer2(x)
            print('layer2 shape {}'.format(x.shape))
            x = self.layer3(x)
            print('layer3 shape {}'.format(x.shape))
            x = self.layer4(x)
            print('layer4 shape {}'.format(x.shape))

            at_map = self.attention(x)
            print('attention_shape {}'.format(at_map.shape))
            x = x * at_map
            print('x*at shape {}'.format(x.shape))

            recon_im = self.decode(x)
            mp = self.relu(x)
            print('x relu {}'.format(x.shape))

            x = self.avgpool(mp)
            print('avgpool shape {}'.format(x.shape))

            x = x.view(x.size(0), -1)
            print('flatten shape {}'.format(x.shape))

            # mu, logvar = self.encode(x)
            # print('mu {}, logvar {}'.format(mu.shape, logvar.shape))
            # x = self.reparameterize(mu, logvar)
            # print('reparameter {}'.format(x.shape))

            x = self.fc(x)
            print('output shape {}'.format(x.shape))
            print('autoencoder')
            time.sleep(30)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # x = self.conv_pool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            at_map = self.attention(x)
            x = x * at_map
            mp = self.relu(x)
            recon_im = self.decode(x)

            x = self.avgpool(mp)
            x = x.view(x.size(0), -1)

            # mu, logvar = self.encode(x)
            # x = self.reparameterize(mu, logvar)

            x = self.fc(x)
        # return x, mp
        return x, recon_im


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
