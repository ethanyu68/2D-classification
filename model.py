
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class conv_layer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(conv_layer, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features, num_output_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_output_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def forward(self, x):
        return self.conv(x)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseCBAMLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseCBAMLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.add_module('cbam', CBAM(growth_rate))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.cbam(self.conv2(self.relu2(self.norm2(bottleneck_output))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _DenseCBAMBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size=4, growth_rate=32, drop_rate=0.3, memory_efficient=False):
        super(_DenseCBAMBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseCBAMLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _SoftMaskBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, drop_rate, memory_efficient=False):
        super(_SoftMaskBlock, self).__init__()
        self.down = nn.Sequential(OrderedDict([
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('res0', _ResidualLayer(num_input_features, num_input_features))]))
        self.down_up = nn.Sequential(OrderedDict([
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        for i in range(num_layers):
            self.down_up.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_input_features))
        self.down_up.add_module('upsample0', nn.UpsamplingBilinear2d(scale_factor=2))
        self.skip = _ResidualLayer(num_input_features, num_input_features)
        self.up = nn.Sequential(OrderedDict([
            ('res{}'.format(num_layers + 2), _ResidualLayer(num_input_features, num_input_features)),
            ('upsample1', nn.UpsamplingBilinear2d(scale_factor=2)),
            ('conv0', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
                                padding=0, bias=False)),
            ('conv1', nn.Conv2d(num_input_features, num_input_features, kernel_size=1, stride=1,
                                padding=0, bias=False)),
            ('sig', nn.Sigmoid())
        ]))

    def forward(self, init_features):
        out_down = self.down(init_features)
        out_downup = self.down_up(out_down)
        out_skip = self.skip(out_downup)
        out_up = self.up(out_downup + out_skip)
        return out_up


class _ResidualLayer(nn.Module):  #@save
    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class _ResidualLayer_CBAM(nn.Module):
    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Ycbam = self.cbam(Y)
        if self.conv3:
            X = self.conv3(X)
        Ycbam += X
        return F.relu(Ycbam)


class _ResidualBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, first_block = False):
        super(_ResidualBlock, self).__init__()
        self.ResBlock = nn.Sequential(OrderedDict([]))
        for i in range(num_layers):
            if i == 0 and not first_block:
                self.ResBlock.add_module('res{}'.format(i), _ResidualLayer(num_input_features, num_output_features,
                                                                           use_1x1conv=True, strides=2))
            else:
                self.ResBlock.add_module('res{}'.format(i), _ResidualLayer(num_output_features, num_output_features))

    def forward(self, x):
        x = self.ResBlock(x)
        return x


class _ResidualBlock_CBAM(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, first_block = False):
        super(_ResidualBlock_CBAM, self).__init__()
        self.ResBlock = nn.Sequential(OrderedDict([]))
        for i in range(num_layers):
            if i == 0 and not first_block:
                self.ResBlock.add_module('res{}'.format(i), _ResidualLayer_CBAM(num_input_features, num_output_features,
                                                                           use_1x1conv=True, strides=2))
            else:
                self.ResBlock.add_module('res{}'.format(i), _ResidualLayer_CBAM(num_output_features, num_output_features))

    def forward(self, x):
        x = self.ResBlock(x)
        return x


class _ResAttentionBlock(nn.Module):
    def __init__(self, num_input_features, num_layers_trunk=2, num_layers_mask=2, drop_rate=0, memory_efficient=False):
        super(_ResAttentionBlock, self).__init__()
        self.Trunk = nn.Sequential(OrderedDict([]))
        for i in range(num_layers_trunk):
            self.Trunk.add_module('res{}'.format(i), _ResidualLayer(num_input_features))
        self.MaskBlock = _SoftMaskBlock(num_layers_mask, num_input_features, drop_rate)
        self.resunit0 = _ResidualLayer(num_input_features, drop_rate)
        self.resunit1 = _ResidualLayer(num_input_features, drop_rate)
        self.resunit2 = _ResidualLayer(num_input_features, drop_rate)
        self.relu = nn.ReLU()

    def forward(self, init_features):
        out_res = self.resunit1(self.resunit0(init_features))
        out_trunk = self.Trunk(out_res)
        out_mask = self.MaskBlock(out_res)
        out_combine = out_trunk * out_mask + out_trunk
        out = self.resunit2(out_combine)
        out = self.relu(out)
        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class CAM(nn.Module):

    """Channel Attention Module

    """

    def __init__(self, in_channels, reduction_ratio=16):

        """
        :param in_channels: int

            Number of input channels.

        :param reduction_ratio: int

            Channels reduction ratio for MLP.
        """

        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels > reduction_ratio) else 1

        pointwise_in = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=reduced_channels_num)

        pointwise_out = nn.Conv2d(kernel_size=1, in_channels=reduced_channels_num, out_channels=in_channels)

        # In the original paper there is a standard MLP with one hidden layer.

        # TODO: try linear layers instead of pointwise convolutions.

        self.MLP = nn.Sequential(pointwise_in, nn.ReLU(), pointwise_out,)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        h, w = input_tensor.size(2), input_tensor.size(3)



        # Get (channels, 1, 1) tensor after MaxPool

        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Get (channels, 1, 1) tensor after AvgPool

        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Throw maxpooled and avgpooled features into shared MLP

        max_feat_mlp = self.MLP(max_feat)

        avg_feat_mlp = self.MLP(avg_feat)

        # Get channel attention map of elementwise features sum.

        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)

        return channel_attention_map


class SAM(nn.Module):

    """Spatial Attention Module"""



    def __init__(self, ks=7):

        """



        :param ks: int

            kernel size for spatial conv layer.

        """



        super().__init__()

        self.ks = ks

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1)



    def _get_padding(self, dim_in, kernel_size, stride):

        """Calculates \'SAME\' padding for conv layer for specific dimension.



        :param dim_in: int

            Size of dimension (height or width).

        :param kernel_size: int

            kernel size used in conv layer.

        :param stride: int

            stride used in conv layer.

        :return: int

            padding

        """



        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2

        return padding



    def forward(self, input_tensor):

        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)


        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis

        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)

        # Get (1, h, w) tensor after MaxPool

        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)

        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)


        # Get (1, h, w) tensor after AvgPool

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)

        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)



        # Concatenate feature maps along the channel axis, so shape would be (2, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)

        # Get pad values for SAME padding for conv2d

        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)

        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)

        # Get spatial attention map over concatenated features.

        self.conv.padding = (h_pad, w_pad)

        spatial_attention_map = self.sigmoid(

            self.conv(concatenated)

        )

        return spatial_attention_map


class CBAM(nn.Module):

    """Convolutional Block Attention Module

    https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

    """

    def __init__(self, in_channels):

        """
        :param in_channels: int

            Number of input channels.

        """

        super().__init__()

        self.CAM = CAM(in_channels)

        self.SAM = SAM()


    def forward(self, input_tensor):
        # Apply channel attention module

        channel_att_map = self.CAM(input_tensor)

        # Perform elementwise multiplication with channel attention map.

        gated_tensor = torch.mul(input_tensor, channel_att_map)  # (bs, c, h, w) x (bs, c, 1, 1)

        # Apply spatial attention module

        spatial_att_map = self.SAM(gated_tensor)

        # Perform elementwise multiplication with spatial attention map.

        refined_tensor = torch.mul(gated_tensor, spatial_att_map)  # (bs, c, h, w) x (bs, 1, h, w)

        return refined_tensor


class LSTM(nn.Module):
    def __init__(self, batchsize, input_size=512, hidden_layer_size=40, output_size=1, num_layers = 2, bidirectional=True):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True )

        self.hidden_cell = (torch.zeros(num_layers*2, batchsize, self.hidden_layer_size),
                            torch.zeros(num_layers*2, batchsize, self.hidden_layer_size))

    def forward(self, input_seq):
        batchsize = input_seq.shape[0]
        seq_length = input_seq.shape[1]
        hidden_cell = (torch.zeros(self.num_layers * 2, batchsize, self.hidden_layer_size).cuda(),
                       torch.zeros(self.num_layers * 2, batchsize, self.hidden_layer_size).cuda())
        lstm_out, self.hidden_cell = self.lstm(input_seq, hidden_cell)
        return lstm_out


class SE(nn.Module):
    def __init__(self, in_features, hidden_features):
        '''
        :param in_features:
        :param hidden_features:
        '''
        super(SE, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        '''
        :param x: vector
        :return:
        '''
        se = self.relu(self.linear2(self.relu(self.linear1(x))))
        out = x * se
        return out



# Network
class ResNet_CBAM(nn.Module):

    def __init__(self, block_config=(6, 8, 16, 12)):
        super(ResNet_CBAM, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each residual block
        self.block0 = _ResidualBlock(num_layers=block_config[0], num_input_features=64, num_output_features=64, first_block=True)

        self.block1 = _ResidualBlock(num_layers=block_config[1], num_input_features=64, num_output_features=128)

        self.block2 = _ResidualBlock(num_layers=block_config[2], num_input_features=128, num_output_features=256)

        self.block3 = _ResidualBlock_CBAM(num_layers=block_config[3], num_input_features=256, num_output_features=512)

        self.bn = nn.BatchNorm2d(512)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(out_block0)
        out_block2 = self.block2(out_block1)
        out_block3 = self.bn(self.block3(out_block2))

        return out_block3


class ResNet_CBAM_cam(nn.Module):
    def __init__(self, input_size=512, num_classes=2):
        super(ResNet_CBAM_cam, self).__init__()
        # feature extractor
        self.input_size = input_size
        self.feature_extractor = ResNet_CBAM()
        self.linear = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, info):
        conv_output = self.feature_extractor(x)
        gap_output = torch.flatten(F.adaptive_avg_pool2d(conv_output, (1, 1)), 1)
        linear_output = self.linear(gap_output)
        linear_weight = self.linear.weight
        B, C, H, W = conv_output.shape
        cam = torch.matmul(linear_weight, conv_output.view(B, C, H*W)).view(B, 2, H, W)
        return linear_output, cam



class ResNet_CBAM_small(nn.Module):

    def __init__(self, block_config=(6, 8, 6, 4, 4)):
        super(ResNet_CBAM_small, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each residual block
        self.block0 = _ResidualBlock(num_layers=block_config[0], num_input_features=64, num_output_features=64, first_block=True)

        self.block1 = _ResidualBlock(num_layers=block_config[1], num_input_features=64, num_output_features=128)

        self.block2 = _ResidualBlock(num_layers=block_config[2], num_input_features=128, num_output_features=256)

        self.block3 = _ResidualBlock_CBAM(num_layers=block_config[3], num_input_features=256, num_output_features=512)

        self.block4 = _ResidualBlock_CBAM(num_layers=block_config[4], num_input_features=512, num_output_features=1024)

        self.bn = nn.BatchNorm2d(1024)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(out_block0)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.bn(self.block4(out_block3))

        return out_block4


class ResNet_LSTM(nn.Module):
    def __init__(self,  batchsize, input_size=1024, hidden_size=40, num_classes=1, num_layers=2):
        super(ResNet_LSTM, self).__init__()
        # feature extractor
        self.input_size = input_size
        self.feature_extractor = ResNet_CBAM_small()
        self.lstm = LSTM(batchsize=batchsize, input_size=input_size, hidden_layer_size=hidden_size,
                         output_size=num_classes, num_layers=num_layers)
        self.linear = nn.Linear(2*hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.feature_extractor(conv_input)
        gap_output = torch.flatten(F.adaptive_avg_pool2d(conv_output, (1, 1)), 1)
        lstm_input = gap_output.view(batch_size, timesteps, -1)
        lstm_output = self.lstm(lstm_input)
        linear_output = self.linear(lstm_output)
        output = self.sigmoid(linear_output)
        return output


class ResNet(nn.Module):

    def __init__(self, block_config=(6, 12, 24, 16), num_init_features=64, num_classes=2):
        super(ResNet, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each residual block
        num_features = num_init_features
        self.block0 = _ResidualBlock(num_layers=block_config[0], num_input_features=64, num_output_features=64, first_block=True)

        self.block1 = _ResidualBlock(num_layers=block_config[1], num_input_features=64, num_output_features=128)

        self.block2 = _ResidualBlock(num_layers=block_config[2], num_input_features=128, num_output_features=256)

        self.block3 = _ResidualBlock(num_layers=block_config[3], num_input_features=256, num_output_features=512)

        # Linear layer
        self.classifier = nn.Linear(512, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        x = x * mask
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(out_block0)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)

        out = F.adaptive_avg_pool2d(out_block3, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out, out_block0, out_block1, out_block2, out_block3


class DenseNet(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 8, 8, 6),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, bn_size=bn_size,
                                growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[0] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[1] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[2] * growth_rate
        self.trans3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                                  growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
        num_features = num_features + block_config[3] * growth_rate
        self.norm5 = nn.BatchNorm2d(num_features)

        # Linear layer
        self.linear = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.block0(out_conv)
        out_block1 = self.block1(self.trans1(out_block0))
        out_block2 = self.block2(self.trans2(out_block1))
        out_block3 = self.block3(self.trans3(out_block2))
        out_norm5 = self.norm5(out_block3)
        out = F.relu(out_norm5, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        linear_weight = self.linear.weight
        B, C, H, W = out_norm5.shape
        cam = torch.matmul(linear_weight, out_norm5.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam


class dense_cbam_info(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, dim_info, block_config=(8, 12, 12, 8), growth_rate=32,# 4 8 6 4
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(dense_cbam_info, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 3
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        # block 4
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        num_features = num_features + dim_info
        self.SE = SE(num_features, num_features// 16)
        self.linear1 = nn.Linear(num_features, num_features//16)
        self.linear2 = nn.Linear(num_features//16, num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out_gap = torch.cat([out_gap, info], 1)
        out_se = F.relu(self.linear2(F.relu(self.linear1(out_gap))))
        out_ca = out_se * out_gap
        out = self.classifier(out_ca)

        linear_weight = self.classifier.weight
        out_feat = out_norm * out_se[:, :self.num_features].view([out_se.shape[0],self.num_features, 1, 1])
        B, C, H, W = out_feat.shape
        cam = torch.matmul(linear_weight[:, :self.num_features], out_feat.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam


class softmask(nn.Module):
    def __init__(self, w, T):
        super(softmask, self).__init__()
        self.w = w
        self.T = T
    def forward(self, A):
        return 1 / (1 + torch.exp(-self.w * (A - self.T * torch.ones(A.shape).cuda())).cuda())


class dense_cbam_info_2branch(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels, dim_info, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(dense_cbam_info_2branch, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        self.CBAM1a = CBAM(num_features)
        # block 3
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        self.block2a = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.trans2a = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        self.CBAM2a = CBAM(num_features)
        # block 4
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        self.block3a = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.norma = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, num_features//16)
        self.linear2 = nn.Linear(num_features//16, num_features)
        self.upsample = nn.Upsample(scale_factor=32, mode='nearest')
        self.softmask = softmask(100, 0.2)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.classifiera = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        # main branch
        out_conv = self.conv_block(x)
        out_block0 = self.trans0(self.block0(out_conv))
        out_block1 = self.trans1(self.block1(out_block0))
        #main branch
        out_block2 = self.trans2(self.block2(out_block1))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(4, -1)
        out_main = self.classifier(out_gap)
        #output cam and mask
        linear_weight = self.classifier.weight
        B, C, H, W = out_norm.shape
        cam = torch.matmul(linear_weight, out_norm.view(B, C, H * W)).view(B, 2, H, W)
        cam_max = cam.max()
        cam_min = cam.min()
        cam_scaled = (cam - cam.min())/(cam_max - cam_min)
        mask = F.sigmoid(10*cam_scaled)
        maskup = self.upsample(mask)
        #auxiliary branch
        xa = (x * maskup).view(-1,1,512,512)
        out_conv_a = self.conv_block(xa)
        out_block0_a = self.trans0(self.block0(out_conv_a))
        out_block1_a = self.trans1(self.block1(out_block0_a))

        out_block2a = self.trans2a(self.block2a(out_block1_a))
        out_block3a = self.block3a(out_block2a)
        out_gap_a = F.adaptive_avg_pool2d(self.norma(out_block3a), (1, 1)).view(8, -1)
        out_aux = self.classifiera(out_gap_a)


        return out_main, out_aux, cam


class dense_cbam(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels = 1, growth_rate=32, block_config=(8, 12, 10, 6),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):
        super(dense_cbam, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 3
        self.block2 = _DenseCBAMBlock(num_layers=block_config[2], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 4
        self.block3 = _DenseCBAMBlock(num_layers=block_config[3], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.trans0(self.block0(out_conv))
        out_block1 = self.trans1(self.block1(out_block0))
        out_block2 = self.trans2(self.block2(out_block1))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out = self.classifier(out_gap)

        linear_weight = self.classifier.weight
        out_feat = out_norm
        B, C, H, W = out_feat.shape
        cam = torch.matmul(linear_weight, out_feat.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam



class dense_cbam_4886(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels = 1, growth_rate=32, block_config=(8, 4, 4, 4),
                 num_init_features=64, bn_size=4, drop_rate=0.5, num_classes=2, memory_efficient=False):

        super(dense_cbam_4886, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseCBAMBlock(num_layers=block_config[0], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 2
        self.block1 = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 3
        self.block2 = _DenseCBAMBlock(num_layers=block_config[2], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        # block 4
        self.block3 = _DenseCBAMBlock(num_layers=block_config[3], num_input_features=num_features, drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.trans0(self.block0(out_conv))
        out_block1 = self.trans1(self.block1(out_block0))
        out_block2 = self.trans2(self.block2(out_block1))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out = self.classifier(out_gap)

        linear_weight = self.classifier.weight
        out_feat = out_norm
        B, C, H, W = out_feat.shape
        cam = torch.matmul(linear_weight, out_feat.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam



class dense_cbam_deeper(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels, growth_rate=32, block_config=(8, 20, 20, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.4, num_classes=2, memory_efficient=False):

        super(dense_cbam_deeper, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 1
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 2
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 3
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        # block 4
        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.SE = SE(num_features, num_features// 16)
        self.linear1 = nn.Linear(num_features, num_features//16)
        self.linear2 = nn.Linear(num_features//16, num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block3 = self.block3(out_block2)
        out_norm = self.norm(out_block3)
        out_gap = F.adaptive_avg_pool2d(out_norm, (1, 1)).view(x.shape[0], -1)
        out_se = F.relu(self.linear2(F.relu(self.linear1(out_gap))))
        out_ca = out_se * out_gap
        out = self.classifier(out_ca)

        linear_weight = self.classifier.weight
        out_feat = out_norm * out_se[:, :self.num_features].view([out_se.shape[0],self.num_features, 1, 1])
        B, C, H, W = out_feat.shape
        cam = torch.matmul(linear_weight[:, :self.num_features], out_feat.view(B, C, H * W)).view(B, 2, H, W)
        return out, cam


class dense_cbam_fusion_2out(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels, dim_info, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(dense_cbam_fusion_2out, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 0
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        # block 1
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 2
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        # 2 block 3
        self.block31 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        self.block32 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features

        self.linear11 = nn.Linear(num_features + dim_info, num_features // 16)
        self.linear12 = nn.Linear(num_features//16, num_features)
        self.linear21 = nn.Linear(num_features + dim_info, num_features // 16)
        self.linear22 = nn.Linear(num_features // 16, num_features)

        # Linear layer
        self.classifier1 = nn.Linear(num_features, num_classes)
        self.classifier2 = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block31 = F.relu(self.norm(self.block31(out_block2)))
        out_block32 = F.relu(self.norm(self.block32(out_block2)))
        out_gap1 = F.adaptive_avg_pool2d(out_block31, (1, 1)).view(x.shape[0], -1)
        out_gap2 = F.adaptive_avg_pool2d(out_block32, (1, 1)).view(x.shape[0], -1)
        out_fusion1 = torch.cat([out_gap1, info], 1)
        out_fusion2 = torch.cat([out_gap2, info], 1)
        # channel attention branch 1, 2
        out_se1 = F.relu(self.linear12(F.relu(self.linear11(out_fusion1))))
        out_ca1 = out_se1 * out_gap1
        out_se2 = F.relu(self.linear22(F.relu(self.linear21(out_fusion2))))
        out_ca2 = out_se2 * out_gap2
        out1 = self.classifier1(out_ca1)
        out2 = self.classifier2(out_ca2)
        # class activation maps
        out_feat1 = out_block31 * out_se1.view([out_se1.shape[0], self.num_features, 1, 1])
        out_feat2 = out_block32 * out_se2.view([out_se2.shape[0], self.num_features, 1, 1])
        B, C, H, W = out_feat1.shape
        cam1 = torch.matmul(self.classifier1.weight, out_feat1.view(B, C, H * W)).view(B, 2, H, W)
        cam2 = torch.matmul(self.classifier2.weight, out_feat2.view(B, C, H * W)).view(B, 2, H, W)
        return out1, out2, cam1, cam2


class dense_cbam_fusion_skip_2out(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_channels, dim_info, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(dense_cbam_fusion_skip_2out, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # block 0
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        num_features0 = num_features
        # block 1
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM1 = CBAM(num_features)
        # block 2
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2 = CBAM(num_features)
        self.trans21 = _Transition(num_input_features=num_features0, num_output_features=num_features0//2)
        self.trans22 = _Transition(num_input_features=num_features0//2, num_output_features=num_features0//4)
        num_features = num_features + num_features0//4
        # 2 block 3
        self.block31 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        self.block32 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        self.block33 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.norm = nn.BatchNorm2d(num_features)
        self.num_features = num_features

        self.linear11 = nn.Linear(num_features, num_features//16)
        self.linear12 = nn.Linear(num_features//16, num_features)
        self.linear21 = nn.Linear(num_features, num_features // 16)
        self.linear22 = nn.Linear(num_features // 16, num_features)

        # Linear layer
        self.classifier1 = nn.Linear(num_features, num_classes)
        self.classifier2 = nn.Linear(num_features, num_classes)
        self.sig = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, info):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block31_in = torch.cat([out_block2, self.trans22(self.trans21((out_block0)))], 1)
        out_block31 = F.relu(self.norm(self.block31(out_block31_in)))
        out_block32 = F.relu(self.norm(self.block32(out_block31_in)))
        out_gap1 = F.adaptive_avg_pool2d(out_block31, (1, 1)).view(x.shape[0], -1)
        out_gap2 = F.adaptive_avg_pool2d(out_block32, (1, 1)).view(x.shape[0], -1)
        #out_gap = torch.cat([out_gap, info], 1)
        # channel attention branch 1, 2
        out_se1 = F.relu(self.linear12(F.relu(self.linear11(out_gap1))))
        out_ca1 = out_se1 * out_gap1
        out_se2 = F.relu(self.linear22(F.relu(self.linear21(out_gap2))))
        out_ca2 = out_se2 * out_gap2
        out1 = self.classifier1(out_ca1)
        out2 = self.classifier2(out_ca2)
        # class activation maps
        out_feat1 = out_block31 * out_se1[:, :self.num_features].view([out_se1.shape[0], self.num_features, 1, 1])
        out_feat2 = out_block32 * out_se2[:, :self.num_features].view([out_se2.shape[0], self.num_features, 1, 1])
        B, C, H, W = out_feat1.shape
        cam1 = torch.matmul(self.classifier1.weight[:, :self.num_features], out_feat1.view(B, C, H * W)).view(B, 2, H,
                                                                                                              W)
        cam2 = torch.matmul(self.classifier2.weight[:, :self.num_features], out_feat2.view(B, C, H * W)).view(B, 2, H,
                                                                                                              W)
        return out1, out2, cam1, cam2


class dense_cbam_3out(nn.Module):
    r"""Densenet-BC model_files class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        dim_info - dimensions of extra inputted information vectors including csection, region...
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(8, 12, 12, 8),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, memory_efficient=False):

        super(dense_cbam_3out, self).__init__()
        # First convolution
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each dense block
        # shared blocks
        num_features = num_init_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features)
        num_features = num_features + block_config[0] * growth_rate
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM0 = CBAM(num_features)
        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features)
        num_features = num_features + block_config[1] * growth_rate
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features2 = num_features // 2
        self.CBAM1 = CBAM(num_features2)
        # paeni branch
        num_features = num_features2
        self.block2p = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2p = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.CBAM2p = CBAM(num_features)
        self.block3p = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.normp = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.classifierp = nn.Linear(num_features, num_classes)


        # cmv and hydroce branch
        num_features = num_features2
        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features)
        num_features = num_features + block_config[2] * growth_rate
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features3 = num_features // 2
        self.CBAM2 = CBAM(num_features3)

        num_features = num_features3
        self.block3c = _DenseBlock(num_layers=block_config[3], num_input_features=num_features3)
        num_features = num_features + block_config[3] * growth_rate
        self.normc = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.classifierc = nn.Linear(num_features, num_classes)

        num_features = num_features3
        self.block3h = _DenseBlock(num_layers=block_config[3], num_input_features=num_features)
        num_features = num_features + block_config[3] * growth_rate
        self.normh = nn.BatchNorm2d(num_features)
        self.num_features = num_features
        self.classifierh = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv = self.conv_block(x)
        out_block0 = self.CBAM0(self.trans0(self.block0(out_conv)))
        out_block1 = self.CBAM1(self.trans1(self.block1(out_block0)))
        # paeni branch
        out_block2p = self.CBAM2p(self.trans2p(self.block2p(out_block1)))
        out_block3p = F.relu(self.normp(self.block3p(out_block2p)))
        out_gap_p = torch.flatten(F.adaptive_avg_pool2d(out_block3p, (1, 1)), 1)
        out_p = self.classifierp(out_gap_p)
        # cmv/pih branch
        out_block2 = self.CBAM2(self.trans2(self.block2(out_block1)))
        out_block3c = F.relu(self.normc(self.block3c(out_block2)))
        out_gap_c = torch.flatten(F.adaptive_avg_pool2d(out_block3c, (1, 1)), 1)
        out_c = self.classifierp(out_gap_c)
        # pih/npih branch
        out_block3h = F.relu(self.normh(self.block3h(out_block2)))
        out_gap_h = torch.flatten(F.adaptive_avg_pool2d(out_block3h, (1, 1)), 1)
        out_h = self.classifierh(out_gap_h)

        # class activation maps
        B, C, H, W = out_block3c.shape
        cam_cmv = torch.matmul(self.classifierc.weight, out_block3c.view(B, C, H * W)).view(B, 2, H, W)
        cam_pih = torch.matmul(self.classifierh.weight, out_block3h.view(B, C, H * W)).view(B, 2, H, W)
        cam_pae = torch.matmul(self.classifierp.weight, out_block3p.view(B, C, H * W)).view(B, 2, H, W)
        return out_h, out_p, out_c, cam_pih, cam_pae, cam_cmv


