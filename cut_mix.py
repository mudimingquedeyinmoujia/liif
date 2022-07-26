import numpy as np
import torch
import random
## 现在的我有两张图像，分别为img256,img512,使用cutmix之后得到一张分辨率为256的图像
## 返回的这张分辨率为256的图像应当这样得到：img256随机扣下一块正方形区域（正方形区域占总体的长或宽比例随机在0.2-0.5之间），正方形区域的位置也是
## 随机的，这块正方形区域的分辨率如果是nxn的话，自然img512中相同区域的分辨率肯定为2nx2n,然后把这2nx2n的区域给下采样之后填充在img256对应的区域即可
## 注意img256中的正方形区域得和img512中的区域对应起来
def cutMix(imageLR,imageHR):
    """
    cut mix operation for img super-resolution
    :param imageLR:
    :param imageHR:
    :return: image with size 256(LR) which have a patch of img512(HR)
    """
    N, C, H, W = imageLR.shape
    cN, cC, cH, cW = imageHR.shape
    ratio = random.random() * 0.3 + 0.3 # patch range in 0.3-0.6
    rH, rW = random.randint(0, H - int(ratio * H)), random.randint(0, W - int(ratio * W))
    ds_imageHR = torch.nn.functional.interpolate(imageHR, (H, W))
    ret = imageLR.clone()
    ret[:, :, rH :rH + int(ratio * H), rW : rW + int(ratio * W)] = ds_imageHR[:, :, rH : rH + int(ratio * H), rW : rW + int(ratio * W)]
    return ret


if __name__ == '__main__':
    from torchvision.transforms import ToTensor
    from PIL import Image
    to_tensor = ToTensor()
    a = to_tensor(Image.open('evals/eval_1/eval_v3/eval_000_res2304.png').convert('RGB').resize((256, 256))).view(1, 3, 256, 256)
    b = to_tensor(Image.open('evals/eval_1/eval_v3/eval_001_res0256.png').convert("RGB").resize((512, 512))).view(1, 3, 512, 512)
    print(a.shape, b.shape)
    ret = cutMix(a, b)
    Image.fromarray((ret[0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)).save("evals/eval_1/eval_v3/eval_001_cutmix.png")
    
