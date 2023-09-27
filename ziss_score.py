import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
# from multiprocessing import Pool
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
# import shutil
# import matplotlib.pyplot as plt
from kornia.morphology import erosion, dilation
from annoy import AnnoyIndex
# import timm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import *
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

root_dir = Path("/home/dtpthao/workspace/camo/utils")
name = "COD10K-CAM-2-Terrestrial-23-Cat-1356" # "COD10K-CAM-2-Terrestrial-23-Cat-1340"
img_path = Path(f"/home/dtpthao/data_unzip/camo/COD10K-v3/Train/Image/{name}.jpg")
gt_path = Path(f"/home/dtpthao/data_unzip/camo/COD10K-v3/Train/GT_Object/{name}.png")
cgt_path = Path(f"/home/dtpthao/workspace/camo/DexiNed/result/cgt/fused/{name}.png")
c_path = Path(f"/home/dtpthao/workspace/camo/DexiNed/result/c/fused/{name}.png")


def get_fid_image_pair(fg_path, bg_path, batch_size=1, device='cuda', dims=2048, num_workers=1):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics([fg_path], model, batch_size,
                                               dims, device, num_workers)
    m2, s2 = calculate_activation_statistics([bg_path], model, batch_size,
                                               dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_img(src, img_name, save_dir, rgb=True):
    if len(src.shape) != 3:
        src = np.stack([src, src, src], axis=2)
    elif src.shape[0] <= 3:
        src = src.transpose(1, 2, 0)
        pass

    src = cv2.normalize(np.uint8(src), None, 0, 255, cv2.NORM_MINMAX)

    if not rgb:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(save_dir / f"{img_name}.png"), src)


class ZissScore:
    def __init__(self, img_path, gt_path, c_path, cgt_path, img_size=(384, 384)):
        self.img_path = img_path
        self.gt_path = gt_path
        self.img_size = img_size
        self.img = self.read_img(self.img_path)
        self.gt = self.read_gt()
        self.c = self.read_img(c_path)
        self.cgt = self.read_img(cgt_path)
        self.gt_coordinate = self.extract_gt()[1]
        self.expanded_gt_coordinate = self.get_expanded_gt_coordinate()

    
    def get_expanded_gt_coordinate(self, expansion_factor=0.1):
        # Expand the bounding box by the expansion factor
        xmin, xmax, ymin, ymax = self.gt_coordinate
        ymin_exp = max(0, int(ymin - (ymax - ymin) * expansion_factor))
        ymax_exp = min(self.gt.shape[0], int(ymax + (ymax - ymin) * expansion_factor))
        xmin_exp = max(0, int(xmin - (xmax - xmin) * expansion_factor))
        xmax_exp = min(self.gt.shape[1], int(xmax + (xmax - xmin) * expansion_factor))
        return (xmin_exp, xmax_exp, ymin_exp, ymax_exp)
   
    def read_img(self, img_path):
        im_src = Image.open(str(img_path))
        im_src = im_src.resize(self.img_size, Image.BICUBIC)
        im_src = np.asarray(im_src, np.float32)
        return im_src

    def read_gt(self):
        gt = Image.open(str(self.gt_path))
        gt = gt.resize(self.img_size, Image.BICUBIC)
        gt = np.asarray(gt, np.float32)
        gt[gt > 0] = 1
        return gt

    def extract_gt(self):
        nonzero = np.nonzero(self.gt)
        ymin, ymax = np.min(nonzero[0]), np.max(nonzero[0])
        xmin, xmax = np.min(nonzero[1]), np.max(nonzero[1])
        
        img = self.img[:, ymin:ymax, xmin:xmax]
        return img, (xmin, xmax, ymin, ymax)
    
    def get_erosion(self, kernel_size=5):
        kernel = torch.ones((kernel_size, kernel_size))
        img_tensor = torch.tensor(self.gt).unsqueeze(0).unsqueeze(0)
        return erosion(img_tensor, kernel).squeeze(0).squeeze(0).numpy()
    
    def get_dilation(self, kernel_size=3):
        kernel = torch.ones((kernel_size, kernel_size))
        img_tensor = torch.tensor(self.gt).unsqueeze(0).unsqueeze(0)
        return dilation(img_tensor, kernel).squeeze(0).squeeze(0).numpy()
    
    def get_boundary(self, mfg, mbg):
        return (1 - mbg.astype(np.int32)) - mfg.astype(np.int32)
    
    def get_contour(self, kernel_size=5):
        erosion = self.get_erosion(kernel_size)
        dilation = self.get_dilation(kernel_size)
        return dilation - erosion
    
    def calc_ssd_patch(self, patch1, patch2):
        # Shape of the patches: (patch_size, patch_size, 3)
        return np.sum((patch1 - patch2) ** 2)
        
    
    def reconstruct_foreground(self, ifg, ibg, patch_size=7, overlap_size=3):
        
        img = self.img.copy()
        stride = patch_size - overlap_size
        fg_nonzero = np.nonzero(ifg)
        bg_non_zero = np.nonzero(ibg)
        bg_patches = []
        
        for (ii, jj) in tqdm(zip(bg_non_zero[0][::stride], bg_non_zero[1][::stride])):
            temp = np.array(img[ii:ii+patch_size, jj:jj+patch_size, :])
            if temp.shape != (patch_size, patch_size, 3):
                continue     
            bg_patches.append(temp)
        
        annoy_index = AnnoyIndex(patch_size*patch_size*3, 'euclidean')
        for i, bg_patch in enumerate(bg_patches):
            annoy_index.add_item(i, bg_patch.flatten())
        annoy_index.build(10)
        

        for (i, j) in tqdm(zip(fg_nonzero[0][::stride], fg_nonzero[1][::stride])):
            # Extract the patch from the foreground image
            y_start = max(self.gt_coordinate[2], i)
            y_end = min(i + patch_size, self.gt_coordinate[3])
            x_start = max(self.gt_coordinate[0], j)
            x_end = min(j + patch_size, self.gt_coordinate[1])
            
            fg_patch = ifg[y_start:y_end, x_start:x_end, :]
            fg_patch = np.array(fg_patch).flatten()
            
            if len(fg_patch) != patch_size*patch_size*3:
                continue
            # print(annoy_index.get_nns_by_vector(fg_patch, 1)[0])
            bg_patch = bg_patches[annoy_index.get_nns_by_vector(fg_patch, 1)[0]]   
            img[y_start:y_end, x_start:x_end, :] = bg_patch.reshape(patch_size, patch_size, 3)
            
            
        return img
        
    
    def get_trimap(self):
        """
        a trimap separation and define the foreground and background regions 
        using morphological erosion and dilation of the mask.
        """
        mfg = self.get_erosion().astype(np.bool_)
        mbg = 1 - self.get_dilation().astype(np.bool_)
        mbg = np.asarray(mbg, np.bool_)
        
        mb = self.get_boundary(mfg, mbg) # * 255.0
        ifg = self.img.copy() * mfg[..., np.newaxis]
        ibg = self.img.copy() * mbg[..., np.newaxis]
        
        # Crop the global background into local background
        ibg = ibg[
            self.expanded_gt_coordinate[2]:self.expanded_gt_coordinate[3],
            self.expanded_gt_coordinate[0]:self.expanded_gt_coordinate[1]
        ]
        
        reconstruct = self.reconstruct_foreground(
            ifg = ifg.copy(), 
            ibg = ibg.copy(),
        )
        ib = self.img.copy() * mb[..., np.newaxis]
        
        save_img(ifg, "ifg", root_dir, rgb=True)
        save_img(ibg, "ibg", root_dir, rgb=True)
        save_img(ib, "ib", root_dir, rgb=True)
        save_img(reconstruct, "reconstruct", root_dir, rgb=True)
        
        return mfg, ifg, reconstruct, mb, self.cgt, self.c


    def get_srf(self, mfg, ifg, reconstruction):
        # Compute the number of foreground pixels
        foreground_pixels = np.count_nonzero(mfg)

        # Compute the masked reconstruction
        masked_reconstruction = reconstruction * mfg[..., np.newaxis]

        # Compute the l2 norm between foreground pixels of the original image
        # and the reconstructed image
        srf = np.linalg.norm(ifg - masked_reconstruction, axis=-1)
        threshold = 0.2 * np.linalg.norm(ifg, axis=-1)

        # Compute the result
        result = np.zeros_like(srf)
        result[srf < threshold] = 1
        return np.count_nonzero(result) / foreground_pixels
        
    def get_sb(self, mb, cgt, c):
        m = MultiLabelBinarizer().fit(mb*cgt)
        score = f1_score(m.transform(mb*cgt), m.transform(mb*c), average='micro')
        return 1 - score
    
    def get_s_alpha(self, srf, sb, alpha=0.35):
        return (1 - alpha)*srf + alpha*sb
    
    def get_ziss_score(self):
        mfg, ifg, reconstruct, mb, cgt, c = self.get_trimap()
        srf = self.get_srf(mfg.copy(), ifg.copy(), reconstruct.copy())
        sb = self.get_sb(mb, cgt, c)
        s_alpha = self.get_s_alpha(srf, sb)
        return srf, sb, s_alpha
             

def main():
    ziss = ZissScore(img_path, gt_path, c_path, cgt_path)
    srf, sb, s_alpha = ziss.get_ziss_score()
    print(srf, sb, s_alpha)
    
    # ifg, ibg, ib, patch = ziss.get_trimap()


if __name__ == '__main__':
    main()
