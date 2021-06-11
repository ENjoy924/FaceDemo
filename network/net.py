import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter


def conv_bn(inp, oup, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()

        self.conv3x3 = conv_bn_no_relu(in_channel, out_channel // 2, 1, 1)
        self.conv5x5_1 = conv_bn(in_channel, out_channel // 4, 1, 1)
        self.conv5x5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, 1, 1)
        self.conv7x7_1 = conv_bn(out_channel // 4, out_channel // 4, 1, 1)
        self.conv7x7_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, 1, 1)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)
        conv7x7_1 = self.conv7x7_1(conv5x5)
        conv7x7 = self.conv7x7_2(conv7x7_1)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return F.relu(out)


class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(FPN, self).__init__()
        self.output1 = conv_bn(in_channel_list[0], out_channel)
        self.output2 = conv_bn(in_channel_list[1], out_channel)
        self.output3 = conv_bn(in_channel_list[2], out_channel)

        self.merge1 = conv_bn1x1(out_channel, out_channel)
        self.merge2 = conv_bn1x1(out_channel, out_channel)

    def forward(self, x):
        out1 = self.output1(x[1])
        out2 = self.output2(x[2])
        out3 = self.output3(x[3])

        up2 = F.interpolate(out3, (out2.size(2), out2.size(3)), mode='nearest')
        out2 += up2
        up1 = F.interpolate(out2, (out1.size(2), out1.size(3)), mode='nearest')
        out1 += up1

        out2 = self.merge2(out2)
        out1 = self.merge1(out1)

        return [out1, out2, out3]


class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1)
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1)
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, 1000)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avg(out)
        out = out.reshape(-1, 256)
        out = self.linear(out)
        return out


class BBoxHead(nn.Module):
    def __init__(self, in_channel=64, anchor_num=2):
        super(BBoxHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, anchor_num * 4, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = torch.reshape(out, (out.shape[0], -1, 4))
        return out


class LandmarkHead(nn.Module):
    def __init__(self, in_channel=64, anchor_num=2):
        super(LandmarkHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, anchor_num * 10, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 10)
        return out


class ClassHead(nn.Module):
    def __init__(self, in_channel, anchor_num=2):
        super(ClassHead, self).__init__()
        self.conv = nn.Conv2d(in_channel, anchor_num * 2, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 2)
        return out


class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet, self).__init__()
        backone = Mobilenet()
        self.body = IntermediateLayerGetter(backone, {'stage1': 1, 'stage2': 2, 'stage3': 3})
        in_channel = 32
        out_channel = 64
        in_channel_list = [
            in_channel * 2,
            in_channel * 4,
            in_channel * 8
        ]
        self.fpn = FPN(in_channel_list, out_channel)
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)
        self.class_head = self.make_class_head()
        self.bbox_head = self.make_bbox_head()
        self.landmark_head = self.make_landmark_head()

    def make_class_head(self, in_channel=64, anchor_num=2, fpn_num=3):
        modules = nn.ModuleList()
        for _ in range(fpn_num):
            modules.append(ClassHead(in_channel, anchor_num))
        return modules

    def make_bbox_head(self, in_channel=64, anchor_num=2, fpn_num=3):
        modules = nn.ModuleList()
        for _ in range(fpn_num):
            modules.append(BBoxHead(in_channel, anchor_num))
        return modules

    def make_landmark_head(self, in_channel=64, anchor_num=2, fpn_num=3):
        modules = nn.ModuleList()
        for _ in range(fpn_num):
            modules.append(LandmarkHead(in_channel, anchor_num))
        return modules

    def forward(self, x):
        out = self.body(x)
        out = self.fpn(out)
        ssh1 = self.ssh1(out[0])
        ssh2 = self.ssh2(out[1])
        ssh3 = self.ssh3(out[2])
        features = [ssh1, ssh2, ssh3]
        class_output = torch.cat([self.class_head[i](f) for i, f in enumerate(features)], dim=1)
        bbox_output = torch.cat([self.bbox_head[i](f) for i, f in enumerate(features)], dim=1)
        landmark_output = torch.cat([self.landmark_head[i](f) for i, f in enumerate(features)], dim=1)
        return bbox_output, landmark_output, class_output


def test_mobilenet():
    input = torch.rand(size=(10, 640, 640, 3))
    input = input.permute(0, 3, 1, 2)
    net = Mobilenet()
    output = net(input)
    print(output.size())


def test_retinanet():
    net = RetinaNet()
    input = torch.rand(size=(1, 640, 640, 3))
    input = input.permute(0, 3, 1, 2)
    print(input.size())
    output = net(input)
    for out in output:
        print(out.size())


if __name__ == '__main__':
    test_retinanet()
