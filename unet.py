""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        # 将一个不可训练的tensor转换成可以训练的类型parameter，
        # 并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)

        out = self.alpha * feat_e + x

        return out


class ChannelAttentionModule(nn.Module, ):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)

        out = self.beta * feat_e + x

        return out


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.pa = PositionAttentionModule(in_channels)
        self.ca = ChannelAttentionModule()
        self.convRes = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.dia1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dia2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dia5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5)

    def forward(self, x):
        px = self.pa(x)
        cx = self.ca(x)
        res = px + cx
        feat_1 = self.conv1x1(x)
        dia_1 = self.dia1(x)
        dia_2 = self.dia2(x)
        dia_5 = self.dia5(x)

        res = self.convRes(res)
        res = res+ dia_1 + dia_2 + dia_5 + feat_1
        return res

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class Soft_Branch(nn.Module):
    def __init__(self, in_channels, out_channels, size1, size2, size3):
        super(Soft_Branch, self).__init__()

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)#
        self.skip1_connection_residual_block = ResidualBlock(out_channels, out_channels)#

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#
        self.softmax2_blocks = ResidualBlock(out_channels, out_channels)#
        self.skip2_connection_residual_block = ResidualBlock(out_channels, out_channels)#

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#
        self.softmax3_blocks = ResidualBlock(out_channels, out_channels)#
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)#

        self.softmax4_blocks = ResidualBlock(out_channels, out_channels)#
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)#
        self.softmax5_blocks = ResidualBlock(out_channels, out_channels)#
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)#
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )


    def forward(self, x):
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2

        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5)
        out_softmax6 = self.softmax6_blocks(out_interp1)

        return out_softmax6


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Maxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.maxpool(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class DropOut(nn.Module):
    def __init__(self,in_channels,out_channels,keep_prob=0.5):
        super(DropOut, self).__init__()
        self.dropout=nn.Dropout(keep_prob)

    def forward(self,x):
        return self.dropout(x)
# 上采样结构
# 如果 bilinear == True, 用双线性差值进行上采样, 尺寸 * 2
# 如果 bilinear == False, 用转置卷积进行上采样, 其输入通道数 = 输出通道数 = in_ch // 2, stride=2，表示尺寸 * 2
# 该层的输入是相邻的两个下采样层的输出
# x1 是由 x2 下采样得到的
# 先对 x1 进行上采样，比较上采样后的 x1 与 x2 的尺寸, 如果不同那么一定是 x1 的尺寸大于 x2 的尺寸
# 在 x2 的四周进行补 0, 使其与 x1 有相同的尺寸
# 对 x1 和 x2 进行级联，级联后的维度就是 in_ch
# 然后对 cat(x1, x2) 进行卷积，卷积后的维度为 out_ch

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 1
        self.inc = DoubleConv(n_channels, 64)
        # self.soft1 = Soft_Branch(n_channels, 64, size1=(448, 448), size2=(224, 224), size3=(112, 112))
        self.soft1 = Soft_Branch(n_channels, 64, size1=(224, 224), size2=(112, 112),size3=(56, 56))
        #2
        self.maxpool1 = Maxpool()
        self.down1 = Down(64, 128)
        # self.soft2 = Soft_Branch(64, 128, size1=(224, 224), size2=(112, 112), size3=(56, 56))
        self.soft2 = Soft_Branch(64, 128, size1=(112, 112), size2=(56, 56), size3=(28, 28))
        #3
        self.maxpool2 = Maxpool()
        self.down2 = Down(128, 256)
        # self.soft3 = Soft_Branch(128,256, size1=(112, 112), size2=(56, 56), size3=(28, 28))
        self.soft3 = Soft_Branch(128,256,  size1=(56, 56), size2=(28, 28), size3=(14, 14))

        #4
        self.maxpool3 = Maxpool()
        self.down3 = Down(256, 512)
        # self.soft4 = Soft_Branch(256, 512, size1=(56, 56), size2=(28, 28), size3=(14, 14))
        self.soft4 = Soft_Branch(256, 512, size1=(28, 28), size2=(14, 14), size3=(7, 7))

        factor = 2 if bilinear else 1
        # 5
        self.maxpool4 = Maxpool()
        self.down4 = Down(512,1024)
        self.df5 = Fusion(512, 1024)
        self.soft5 = Soft_Branch(512, 1024, size1=(14, 14), size2=(7, 7), size3=(4, 4))

        #6
        self.maxpool5 = Maxpool()
        self.down5 = Down(1024, 2048 // factor)

        self.drop = DropOut(2048,2048)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128// factor, bilinear)
        self.up5 = Up(128, 64,bilinear)
        self.outc = OutConv(64, n_classes)



    def forward(self, x):
        x1 = self.inc(x)
        s1 = self.soft1(x)
        last_x1 = (1+s1)*x1

        x2 = self.maxpool1(last_x1)
        s2 = self.soft2(x2)
        x2 = self.down1(x2)
        last_x2 = (1+s2)*x2
        # last_x2 = x2

        x3 = self.maxpool2(last_x2)
        s3 = self.soft3(x3)
        x3 = self.down2(x3)
        last_x3 = (1+s3)*x3
        # last_x3 = x3

        x4 = self.maxpool3(last_x3)
        s4 = self.soft4(x4)
        x4 = self.down3(x4)
        last_x4 = (1+s4)*x4
        # last_x4 = x4

        x5 = self.maxpool4(last_x4)
        s5 = self.soft5(x5)
        x5 = self.down4(x5)
        # last_x5 = (1 + d5) * x5
        last_x5 = (1 + s5) * x5
        # last_x5 = x5

        x6 = self.maxpool5(last_x5)
        x6 = self.down5(x6)
        x6 = self.drop(x6)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits

