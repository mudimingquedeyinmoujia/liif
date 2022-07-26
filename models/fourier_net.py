import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import models
from models import register

def calc_ind(sf, ef):
    """
    get start frequency and end frequency
    return: the start index, end index in orthogonal position encoding and tmp elements number : P_sf + ... + P_ef
    """
    if sf == 0:
        start_ind = 0
    else:
        start_ind = 4 * (sf - 1) * (sf - 1) + 4 * (sf - 1) + 1
    end_ind = 4 * ef * ef + 4 * ef + 1
    return start_ind, end_ind, end_ind - start_ind

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        max_freq = self.kwargs['all_freq']

        # when freqx && freqy==0
        embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))
        p_fns = [lambda x, y: torch.cos(x) * torch.cos(y),
                 lambda x, y: torch.cos(x) * torch.sin(y),
                 lambda x, y: torch.sin(x) * torch.cos(y),
                 lambda x, y: torch.sin(x) * torch.sin(y)]
        for freq in range(1, max_freq + 1):
            # when freqx==0 || freqy==0
            embed_fns.append(lambda x, freq=freq: torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.sin(x[:, :, 1] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.cos(x[:, :, 1] * freq).unsqueeze(-1))

            for freq_tmp in range(1, freq):
                for p_fn in p_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freqx=freq, freqy=freq_tmp:
                                     p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))
                    embed_fns.append(lambda x, p_fn=p_fn, freqx=freq_tmp, freqy=freq:
                                     p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))

            for p_fn in p_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freqx=freq, freqy=freq:
                                 p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))

        out_dim = 4 * max_freq * max_freq + 4 * max_freq + 1
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, sf, ef, repeat):
        if ef is None:
            ef = self.kwargs['all_freq']
        assert ef <= self.kwargs['all_freq'] and ef >= 0
        assert sf <= self.kwargs['all_freq'] and sf >= 0
        start_ind, end_ind, dim = calc_ind(sf, ef)
        tmp_embed_fns = self.embed_fns[start_ind:end_ind]
        channel = torch.cat([fn(inputs * self.kwargs['omega']).to(inputs.device) for fn in tmp_embed_fns], -1)
        if repeat:
            return channel.repeat(1, 1, 3), dim  # channel copy from 1 to 3 (grey to RGB)
        return channel, dim


def get_embedder(all_freq, omega, if_embed=True):
    """
    获取位置编码函数与位置编码后的维度,omega与周期有关
    :param all_freq: n
    :param if_embed: if use embed
    :return: example: x,y ---> 1,cosy,siny,cos2y,sin2y,... ; outdim=4*n*n+2*n*2+1(only one channel)
    """
    if not if_embed:
        return nn.Identity(), 2

    embed_kwargs = {
        'all_freq': all_freq,
        'omega': omega,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed_fn = lambda x, eo=embedder_obj, sf=0, ef=None, repeat=True: eo.embed(x, sf, ef, repeat)
    return embed_fn, embedder_obj.out_dim


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    follow the coord system as:
          +
          +
    +  +  +  +  + >
          +        y
          +
          V x
    need to use flip(-1) when use grid_sample to change to
          +
          +
    +  +  +  +  + >
          +        x
          +
          V y
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:  # H,W,2 ---> H*W,2
        ret = ret.view(-1, ret.shape[-1])
    return ret


class Fourier_render_patch(torch.nn.Module):
    """
    means every patch is independent
    not interpolation
    """

    def __init__(self, all_freq, omega=math.pi):
        super().__init__()
        self.emb_fn, self.out_dims = get_embedder(all_freq=all_freq, omega=omega,
                                                  if_embed=True)  # out_dims = C not 3C
        self.omega = omega
        self.all_freq = all_freq

    def query_F(self, feat, coord):
        # feat: N,3C,H,W / coord:  N,bsize,2
        # assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        coord_ = coord.clone()  # N,bsize,2
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        q_coord = F.grid_sample(  # 把特征图像素的坐标得到
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord - q_coord  # N,bsize,2
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1

        fourier_base, tmp_dim = self.emb_fn(rel_coord, s=self.tmp_start, e=self.tmp_end)  # N,bsize,3C / C

        fourier_series = fourier_projection * fourier_base  # N,bsize,3C
        f_R = torch.sum(fourier_series[:, :, :tmp_dim], dim=2, keepdim=True)
        f_G = torch.sum(fourier_series[:, :, tmp_dim:2 * tmp_dim], dim=2, keepdim=True)
        f_B = torch.sum(fourier_series[:, :, 2 * tmp_dim:], dim=2, keepdim=True)
        ret = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
        return ret

    def query_rgb(self, feat, coord):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord)

    def batched_predict(self, inp, coord, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :])  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, sf=0, ef=None, bsize=30000):

        self.tmp_start = sf
        if ef is None:
            self.tmp_end = self.all_freq
        else:
            self.tmp_end = ef
        _, _, dms = calc_ind(sf, ef)
        assert img_feature.shape[1] == dms * 3
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1),
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred

class Fourier_render_patch_int(torch.nn.Module):
    """
    means every patch is independent
    set omega to 0.5*pi
    use interpolation between patches
    """

    def __init__(self, all_freq, omega=0.5*math.pi):
        super().__init__()
        self.all_freq=all_freq
        self.omega=omega
        self.emb_fn,self.all_C=get_embedder(all_freq=all_freq,omega=omega,if_embed=True)

    def query_F(self, feat, coord):
        # feat: N,3C,H,W / coord:  N,bsize,2
        # assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2  # half pixel of feature map
        ry = 2 / feat.shape[-1] / 2
        # get the center point coord of feature map
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # N,bsize,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                fourier_projection_tmp = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # N,bsize,3C
                q_coord = F.grid_sample(  # 把特征图像素的坐标得到
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord  # N,bsize,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -2,2

                fourier_base_tmp, tmp_dim = self.emb_fn(rel_coord, sf=self.tmp_start_f, ef=self.tmp_end_f)  # N,bsize,3C
                fourier_series = fourier_projection_tmp * fourier_base_tmp  # N,bsize,3C
                f_R = torch.sum(fourier_series[:, :, :tmp_dim], dim=2, keepdim=True)
                f_G = torch.sum(fourier_series[:, :, tmp_dim:2 * tmp_dim], dim=2, keepdim=True)
                f_B = torch.sum(fourier_series[:, :, 2 * tmp_dim:], dim=2, keepdim=True)
                pred = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)  # area: N,30000

        tot_area = torch.stack(areas).sum(dim=0)  # areas: 4*N,bsize tot_area: 1*N,30000

        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret  # N,bsize,3

    def query_rgb(self, feat, coord):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord)

    def batched_predict(self, inp, coord, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :])  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, sf=0, ef=None, bsize=30000):

        self.tmp_start_f=sf
        self.tmp_end_f=ef
        if ef is None:
            self.tmp_end_f=self.all_freq
        _, _, dms = calc_ind(sf, ef)
        assert img_feature.shape[1] == dms*3
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1),
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred

@register('fourierSR-net')
class FourierSR_net(torch.nn.Module):
    """
    input: image with (h,w,3) and target H,W number
    output: HR image with H,W
    """

    def __init__(self, input_size, max_freq, srnet_spec=None, rend_split=None):
        super().__init__()
        self.input_size=input_size
        self.max_freq=max_freq
        self.srnet=models.make(srnet_spec)
        self.rend_split = rend_split
        if rend_split is None:
            self.rend_split = {
                0: [0, 0],
                1: [1, self.max_freq]
            }
        self.render = Fourier_render_patch_int(all_freq=self.max_freq)

    def forward(self, img_lr, h, w, retall=False, bsize=30000):
        imghr_Sn_list = []
        imghr_Pn_list = []
        feature_list = []
        feature_list.append(img_lr)
        feature_list.extend(self.srnet(img_lr))
        skip=None
        for i in range(len(feature_list)):
            imghr_rend_tmp = self.render(feature_list[i], h=h, w=w, sf=self.rend_split[i][0], ef=self.rend_split[i][1], bsize=bsize)
            imghr_Pn_list.append(imghr_rend_tmp)
            if skip is None:
                skip=imghr_rend_tmp
                imghr_Sn_list.append(imghr_rend_tmp)
            else:
                skip=skip+imghr_rend_tmp
                imghr_Sn_list.append(skip)
        if retall:
            return [imghr_Pn_list, imghr_Sn_list]
        else:
            return skip


if __name__ == "__main__":
    net = Fourier_render_patch_int(all_freq=8)
    for i in net.parameters():
        print(i)
