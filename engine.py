import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import cv2
import numpy as np
from lasinet import LASINet
from unet import UNet
from PIL import Image
import matplotlib.pyplot as plt

class Engine(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_size = 256
        self.lasinet = LASINet()
        self.unet = UNet()
        self.load_models()
    
    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters share common prefix 'module.
        If have prefix, then remove"""
        def f(x):
            return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def add_prefix(self, state_dict, prefix):
        """If not have prefix, then add"""
        def f(x):
            return prefix + x if not x.startswith(prefix) else x         
        return {f(key): value for key, value in state_dict.items()}
    
    def visulize_spect(self, image, frame, pred):
                    
        pd_E = pred[0].squeeze().detach().cpu().numpy()
        pd_A = pred[1].squeeze().detach().cpu().numpy()
        pd_E = round(pd_E*self.image_size)
        pd_A = round(pd_A*self.image_size)
        
        cv2.line(image, (pd_E, 0), (pd_E, int(self.image_size)), (0,255,0))
        cv2.line(image, (pd_A, 0), (pd_A, int(self.image_size)), (0,255,0))
  
        return image
    
    def load_models(self):
        ckpt_lasinet_dir = 'ckpt_lasinet.pth.tar'
        ckpt_unet_dir = 'ckpt_unet.pth'

        print("=> LASINET: loading checkpoint '{}'".format(ckpt_lasinet_dir))
        checkpoint = torch.load(ckpt_lasinet_dir)

        state_dict = checkpoint['state_dict']
        if not hasattr(self.lasinet, 'module'):
            state_dict = self.remove_prefix(state_dict, 'module.')
        else:
            state_dict = self.add_prefix(state_dict, 'module.')

        self.lasinet.load_state_dict(state_dict, strict=False)
        print("=> UNET: loading checkpoint '{}'".format(ckpt_lasinet_dir))
        self.unet.load_state_dict(torch.load(ckpt_unet_dir))

    def load_spetcrum(self, data_dir):       
        images = []
        image_cv = []
        sizes = []
        frames = sorted(os.listdir(data_dir))
        for frame in frames:
            image = Image.open(f'{data_dir}/{frame}').convert('RGB')
            image = np.array(image)
            h, w, _ = image.shape
            image = cv2.resize(image, (self.image_size, self.image_size))
            image_cv.append(image)
            sizes.append((h, w))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = image / 255
            images.append(image.transpose(2, 0, 1))

        image_cv = np.array(image_cv)
        images = np.array(images)
        images = torch.FloatTensor(images)
        return images, image_cv, frames
    
    def load_echogram(self, data_dir):
        images_cv = []
        images = []
        frames = sorted(os.listdir(f'{data_dir}'))
        for frame in frames:
            image = cv2.imread(f'{data_dir}/{frame}')
            image = cv2.resize(image,(224, 224))
            images_cv.append(image)
            img_trans = image.transpose((2, 0, 1))
            img_trans = img_trans / img_trans.max()
            images.append(img_trans)
        images = np.array(images)
        images = torch.from_numpy(images)
        images = images.to(dtype=torch.float32)
        return images, images_cv, frames
    
    def visulize_echo(self, images, frames, masks):
        clr = (0,255,0)
        result_images = []
        for image, mask, frame in zip(images, masks, frames):
            # mask = mask.astype(np.uint8)
            gray = mask.astype(np.uint8)
            assert gray.shape[0]==image.shape[0] and gray.shape[1] == image.shape[1],f'预测的结果形状({gray.shape[-2]},{gray.shape[-1]})和图像形状({image.shape[-2]},{image.shape[-1]})不一致'
            contours,_=cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,clr,2)
            result_images.append(image)
        gif_frames = []
        for i in range(len(result_images)):
            gif_frames.append(Image.fromarray(np.uint8(result_images[i])))
        
        return result_images, gif_frames
    
    def get_perimeter(self, mask):
        gray = mask.astype(np.uint8)
        contours,_ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            max_contour_len = 0
            for ct in contours:
                if len(ct) > max_contour_len:
                    max_contour_len = len(ct)
                    contour = ct
        else :
            contour = contours[0]
        
        perimeter = cv2.arcLength(contour, True)
        return perimeter
    
    def draw_lasi_plot(self, strains, loc_pde, loc_pda):
        plt.plot(strains)

        # 在第5个数值处添加一条垂直于 x 轴的支线
        plt.axvline(x=0, color='green', linestyle='--')  # 第5个数值对应的索引为4（从0开始索引）
        plt.axvline(x=loc_pde, color='green', linestyle='--')  # 第5个数值对应的索引为4（从0开始索引）
        plt.axvline(x=loc_pda, color='green', linestyle='--')  # 第5个数值对应的索引为4（从0开始索引）

        # 添加横坐标轴
        x_ticks = [f'{i+1}' if (i+1//10==0) else '' for i in range(len(strains))]
        x_ticks[0] = 'S_0'
        x_ticks[loc_pde] = 'S_E'
        x_ticks[loc_pda] = 'S_A'
        plt.xticks(range(len(strains)), x_ticks )
        # 添加标题和标签
       
        plt.xlabel('Frames')
        plt.ylabel('Strains')

        fig = plt.gcf()
        # 将图形转换为数组
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)

        # 重新排列颜色通道为BGA顺序
        image_array_bga = image_array[:, :, [2, 1, 0, 3]]
        return image_array_bga

    def cal_metrics(self, pred_masks, pred_locs):
        pd_E = pred_locs[0].squeeze().detach().cpu().numpy()
        pd_A = pred_locs[1].squeeze().detach().cpu().numpy()
        pd_E = int(pd_E*len(pred_masks))
        pd_A = int(pd_A*len(pred_masks))

        perimeter_0 = self.get_perimeter(pred_masks[0])
        strains = []
        for mask in pred_masks[1:]:
            perimeter_t = self.get_perimeter(mask)
            strain =(perimeter_t-perimeter_0)/perimeter_0
            strains.append(strain)
        S_0 = strains[0]
        S_E = strains[pd_E]
        S_A = strains[pd_A]

        LASr  = S_E-S_0
        LAScd = S_A-S_E
        LASct = S_0-S_A

        plot = self.draw_lasi_plot(strains, pd_E, pd_A)

        return LASr, LAScd, LASct, plot
    
    def forward(self, data_dir):
        # load_checkpoint
        spec_path = f'{data_dir}/spectrum'
        echo_path = f'{data_dir}/echogram'
        assert os.path.exists(spec_path), f'请将频谱图像置于指定目录下:{spec_path}'
        assert os.path.exists(spec_path), f'请将超声图像置于指定目录下:{echo_path}'
        input, image_cv, spec_frames = self.load_spetcrum(spec_path)
        loc_preds = self.lasinet(input.float())
        spectrum = self.visulize_spect(image_cv[0], spec_frames[0], loc_preds[0])

        input, image_cv, echo_frames = self.load_echogram(echo_path)
        seg_preds = self.unet(input)

        masks = torch.sigmoid(seg_preds)
        masks = masks.squeeze().detach().numpy()
        masks[masks > 0.5]=1

        echograms, echo_gif = self.visulize_echo(image_cv, echo_frames, masks)
        LASr, LAScd, LASct, plot = self.cal_metrics(masks, loc_preds[0])
        metrics = {'lasr':LASr, 'lascd':LAScd, 'lasct':LASct}
        return spectrum, spec_frames[0], echograms, echo_frames, echo_gif, plot, metrics
    
    