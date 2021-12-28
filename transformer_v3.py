import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Siamtrans(BaseNetwork):
    def __init__(self, init_weights=True):
        super(Siamtrans, self).__init__()
        channel = 256
        stack_num = 8 
        patchsize = [(32, 32), (16, 16),(8,8),(4,4)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)
        self.tforward = TForwardBlock(patchsize, hidden=channel)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), #  s=2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  #s=2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, frames):
        # extracting features
        #b, t, c, h, w = masked_frames.size()
        #import pdb
        #pdb.set_trace()
        b=1
        if isinstance(frames,list):
            enc_feat1 = self.encoder(frames[0])
            enc_feat2 = self.encoder(frames[1])
            _, c, h, w = enc_feat1.size()

            enc_feat = self.tforward(
                {'x': [enc_feat1,enc_feat2], 'b': b, 'c': c})['x']
        else:
            #pdb.set_trace()
            enc_feat1 = self.encoder(frames[0:1,:])
            enc_feat2 = self.encoder(frames[1:2,:])
            _, c, h, w = enc_feat1.size()
        
            enc_feat = self.tforward(
                {'x': [enc_feat1,enc_feat2], 'b': b, 'c': c})['x']
        enc_feat = self.transformer(
            {'x': enc_feat, 'b': b, 'c': c})['x']
        output = self.decoder(enc_feat)
        
        output = torch.tanh(output)
        return output

    def infer(self, feat):
        
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'b': 1, 'c': c})['x']
        return enc_feat


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        #scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, b, c):
        #pdb.set_trace()
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            #mm = m.view(b, t, 1, out_h, height, out_w, width)
            #mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            #    b,  t*out_h*out_w, height*width)
            #mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x
from random import choice

class MultiHeadedAttention_v2(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()
    def move1(self,x):
        b,c,w,h=x.shape
        x = x[:,:,2:,2:]
        x = nn.functional.interpolate(x,(w,h),mode='bilinear',align_corners=True) 
        return x

    def move2(self,x):
        b,c,w,h=x.shape
        x = x[:,:,:-2,:-2]
        x = nn.functional.interpolate(x,(w,h),mode='bilinear',align_corners=True)
        return x

    def move3(self,x):
        b,c,w,h=x.shape
        x = x[:,:,2:,:-2]
        x = nn.functional.interpolate(x,(w,h),mode='bilinear',align_corners=True)
        return x

    def move4(self,x):
        b,c,w,h=x.shape
        x = x[:,:,:-2,2:]
        x = nn.functional.interpolate(x,(w,h),mode='bilinear',align_corners=True)
        return x

    def forward(self, x , x1, b, c):
        
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        
        #random_function_selector = [self.move1, self.move2, self.move3, self.move4]
        #random_func = choice(random_function_selector)
        #if x1==None:
        #    shuffle = torch.cat([x[1:2,:],x[2:3,:],x[3:4,:],x[0:1,:]],0)
            #shuffle = random_func(shuffle).detach()
        #    _key = self.key_embedding(shuffle)
        #    _value = self.value_embedding(shuffle)
        #else:

        
        _key = self.key_embedding(x1)
        _value = self.value_embedding(x1)
        
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b*t,  out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b*t,  out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b*t,  out_h*out_w, d_k*height*width)
            y, _ = self.attention(query, key, value)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x,  b, c = x['x'], x['b'], x['c']
        x = x + self.attention(x, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}

class TForwardBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention_v2(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        #pdb.set_trace()
        if isinstance(x['x'],list):
            x0,x1,  b, c = x['x'][0],x['x'][1] ,x['b'], x['c']
            x = x0 + self.attention(x0,x1, b, c)
        else:
            x0, b, c = x['x'],x['b'], x['c']
            x = x0 + self.attention(x0,None, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}

if name=='__main__':
    model = Siamtrans()
    model = model.cuda()
    net_input = torch.randn(1,3,128,128)
    net_input2 = torch.randn(1,3,128,128)
    
    z1 = model(torch.cat([net_input.cuda(),(net_input2).cuda().detach()],0))       
        
    z2 = model(torch.cat([net_input2.cuda(),(net_input).cuda().detach()],0))
    print(z1.shape, z2.shape)

