
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


'''
loss
'''
class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        mask = (label == 1).float()
        pos_num = torch.sum((label==1).float()) + 0.0001
        neg_num = torch.sum((label==0).float()) + 0.0001

        loss_1 = torch.sum((1-mask) * torch.pow(distance, 2)) / neg_num
        loss_2 = torch.sum(mask*torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)) / pos_num
        loss = loss_1 + loss_2

        return loss/2

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )






'''
特征提取网络
'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self,  block, layers, output_stride, BatchNorm, pretrained=True, in_c=3):

        self.inplanes = 64
        self.in_c = in_c
        print('in_c: ',self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # | 4
        x = self.layer1(x)  # | 4
        low_level_feat2 = x  # | 4
        x = self.layer2(x)  # | 8
        low_level_feat3 = x
        x = self.layer3(x)  # | 16
        low_level_feat4 = x
        x = self.layer4(x)  # | 32
        return x, low_level_feat2, low_level_feat3, low_level_feat4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

'''
特征融合网络
'''
class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x
class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(64, 96)
        self.dr3 = DR(128, 96)
        self.dr4 = DR(256, 96)
        self.dr5 = DR(512, 96)
        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )

        self._init_weight()


    def forward(self, x,low_level_feat2, low_level_feat3, low_level_feat4):

        # x1 = self.dr1(low_level_feat1)
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
'''
自注意力网络
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class _PAMBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  *  H  *  (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    '''
    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)



    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)//2


        local_y = []
        local_x = []
        step_h, step_w = h//self.scale, w//self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i*step_h, j*step_w
                end_x, end_y = min(start_x+step_h, h), min(start_y+step_w, w)
                if i == (self.scale-1):
                    end_x = h
                if j == (self.scale-1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:,:,:,w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:,:,:,w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:,:,:,w:]], 4)  # B*N*H*W*2


        local_block_cnt = 2*self.scale*self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0,2,1))
            # context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local

        #  Parallel Computing to speed up
        #  reshape value_local, q, k
        v_list = [value[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list,dim=0)
        q_list = [query[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list,dim=0)
        k_list = [key[:,:,local_x[i]:local_x[i+1],local_y[i]:local_y[i+1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list,dim=0)
        # print(v_locals.shape)
        context_locals = func(v_locals,q_locals,k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size*(j+i*self.scale)
                right = batch_size*(j+i*self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:,:,:,:,0],context[:,:,:,:,1]],3)


        if self.ds !=1:
            context = F.interpolate(context, [h*self.ds, 2*w*self.ds])

        return context
class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels == None:
            key_channels = in_channels//8
        if value_channels == None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels,key_channels,value_channels,scale,ds)
class PAM(nn.Module):
    """
        PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8


        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds)
             for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0,bias=False),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels,key_channels,value_channels,size,ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]

        #  concat
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))

        return output
class CDSA(nn.Module):
    """self attention module for change detection

    """
    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        print('ds: ',self.ds)
        self.mode = mode
        if self.mode == 'BAM':
            # self.Self_Att = BAM(self.in_C, ds=self.ds)
            pass
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1,2,4,8],ds=self.ds)
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:,:,:,0:height], x[:,:,:,height:]
'''
整体网络
'''
class StaNet(nn.Module):
    def __init__(self,inc):
        super().__init__()
        self.resNet = ResNet(BasicBlock, [2, 2, 2, 2], 32, nn.BatchNorm2d, 3)
        self.fuseNet = Decoder(64, nn.BatchNorm2d)
        self.netA = CDSA(in_c=64, ds=1, mode='PAM')
        self.prehead = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=2)

    def forward(self, x1, x2):
        x, f2, f3, f4 = self.resNet(x1)
        feature1 = self.fuseNet(x, f2, f3, f4)

        x, f2, f3, f4 = self.resNet(x2)
        feature2 = self.fuseNet(x, f2, f3, f4)
        feature1, feature2 = self.netA(feature1, feature2)

        feature1 = self.prehead(feature1)
        feature2 = self.prehead(feature2)
        feature = torch.abs(feature1-feature2)

        feature = F.interpolate(feature, scale_factor=4, mode='bilinear', align_corners=True)
        result = self.classifier(feature)

        # dist = F.pairwise_distance(feature1, feature2, keepdim=True)  # 特征距离
        # result = F.interpolate(dist, size=(256, 256), mode='bilinear', align_corners=True)

        return result









