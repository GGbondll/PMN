from utils import *
import torch
import cv2
from data_process.process import *

noise_path_100 = 'data/SID/Sony/short/10068_00_0.1s.ARW'
denoised_path_100 = 'images/samples-SonyA7S2_Mix_Unet/10068_[100]_denoised.png'

noise_path_250 = 'data/SID/Sony/short/10068_01_0.04s.ARW'
denoised_path_250 = 'images/samples-SonyA7S2_Mix_Unet/10068_[250]_denoised.png'

denoised_paried_path_100 = 'images/samples-SonyA7S2_Paired_Official/10068_[100]_denoised.png'
denoised_paried_path_250 = 'images/samples-SonyA7S2_Paired_Official/10068_[250]_denoised.png'

denoised_ELD_path_100 = '/home/weijie.su/PMN/images/samples-SonyA7S2_ELD_Official/10068_[100]_denoised.png'
denoised_ELD_path_100 = '/home/weijie.su/PMN/images/samples-SonyA7S2_ELD_Official/10068_[250]_denoised.png'

Img = dataload(noise_path_250)
print(Img.shape, type(Img))

point1 = [1235, 98]
point2 = [1992, 491]

coor_data = [[1235, 98, 1992, 491]]
image_path = denoised_path_100
def runsnr(image_path, coor_data):
    denoise_snr = CalSNR(image_path,coor_data)
    print(denoise_snr)
    return denoise_snr

def get_darkshading(iso):
    with open(os.path.join(f'resources/darkshading_BLE.pkl'), 'rb') as f:
        blc_mean = pkl.load(f)
    branch = '_highISO' if iso>1600 else '_lowISO'
    ds_k = np.load(os.path.join('resources', f'darkshading{branch}_k.npy'))
    ds_b = np.load(os.path.join('resources', f'darkshading{branch}_b.npy'))
    darkshading = ds_k * iso + ds_b + blc_mean[iso]

    return darkshading

def readadata():
    H, W = 2848, 4256
    iso = 150
    darkshading = get_darkshading(iso)
    
    lr_imgs_100 = []
    for i in range(10):
        lr_raw_100 = np.array(dataload(f'data/SID/Sony/short/10068_0{i}_0.1s.ARW')).reshape(H,W)
        lr_raw_100 = lr_raw_100 - darkshading
        lr_img_100 = raw2bayer(lr_raw_100, wp=16383, bl=512, norm=True, clip=False)
        lr_imgs_100.append(lr_img_100[None,:])
    lr_imgs_250 = []
    for i in range(2):
        lr_raw_250 = np.array(dataload(f'data/SID/Sony/short/10068_0{i}_0.04s.ARW')).reshape(H,W)
        lr_raw_250 = lr_raw_250 - darkshading
        lr_img_250 = raw2bayer(lr_raw_250, wp=16383, bl=512, norm=True, clip=False)
        lr_imgs_250.append(lr_img_250[None,:])
    
    return lr_imgs_100, lr_imgs_250
    
    
def train_snr(net):
    coor_data = [[1235, 98, 1992, 491]]
    lr_imgs_100, lr_imgs_250 = readadata()
    snr_100 = 0
    for img in lr_imgs_100:
        if imgs_lr.shape[-1] % 16 != 0:
            p2d = (4,4,4,4)
            imgs_lr = F.pad(img, p2d, mode='reflect')
            imgs_dn = net(img)
            imgs_lr = imgs_lr[..., 4:-4, 4:-4]
            imgs_dn = imgs_dn[..., 4:-4, 4:-4]
        else:
            imgs_dn = net(img)
        output = tensor2im(imgs_dn)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        snr_100 += runsnr(output, coor_data)
    snr_100 /= len(lr_imgs_100)
    snr_250 = 0
    for img in lr_imgs_250:
        if imgs_lr.shape[-1] % 16 != 0:
            p2d = (4,4,4,4)
            imgs_lr = F.pad(img, p2d, mode='reflect')
            imgs_dn = net(img)
            imgs_lr = imgs_lr[..., 4:-4, 4:-4]
            imgs_dn = imgs_dn[..., 4:-4, 4:-4]
        else:
            imgs_dn = net(img)
        output = tensor2im(imgs_dn)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        snr_250 += runsnr(output, coor_data)
    snr_250 /= len(lr_imgs_250)    
    return snr_100, snr_250
        
    
if __name__ == '__main__':
    for image_path in (denoised_paried_path_100, denoised_paried_path_250, denoised_ELD_path_100, denoised_ELD_path_100):
        runsnr(image_path, coor_data)
        print('------------------------------------------------------------------------------')