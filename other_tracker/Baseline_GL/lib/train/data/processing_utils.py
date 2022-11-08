import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
prj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(os.path.join(prj_path, "lib/train/data/"))
from trackmix import TrackMix
'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None, im_mix=None, gt_mix=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
        if gt_mix is not None:
            mix_x, mix_y, mix_w, mix_h = gt_mix.tolist()
    else:
        x, y, w, h = target_bb
        if gt_mix is not None:
            mix_x, mix_y, mix_w, mix_h = gt_mix
    # Crop image
    if gt_mix is not None:
        track_mix = TrackMix()
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        crop_sz_mix = math.ceil(math.sqrt(mix_w * mix_h) * search_area_factor)
        if crop_sz < 1 or crop_sz_mix < 1:
            raise Exception('Too small bounding box.')

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # crop mix image
        mix_x1 = round(mix_x + 0.5 * mix_w - crop_sz_mix * 0.5)
        mix_x2 = mix_x1 + crop_sz_mix

        mix_y1 = round(mix_y + 0.5 * mix_h - crop_sz_mix * 0.5)
        mix_y2 = mix_y1 + crop_sz_mix

        mix_x1_pad = max(0, -mix_x1)
        mix_x2_pad = max(mix_x2 - im_mix.shape[1] + 1, 0)

        mix_y1_pad = max(0, -mix_y1)
        mix_y2_pad = max(mix_y2 - im_mix.shape[0] + 1, 0)
        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        im_crop_mix = im_mix[mix_y1 + mix_y1_pad:mix_y2 - mix_y2_pad, mix_x1 + mix_x1_pad:x2 - mix_x2_pad, :]
        # track mix crop
        im_crop = track_mix.transform_image(im_crop, mix_img=im_crop_mix)
    else:
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)
        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        # have extra mix image
        if len(frames) == 3:
            im_mix = frames[2]
            gt_mix = box_extract[2]
            frames = frames[:2]
            box_extract = box_extract[:2]
            crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m, im_mix=im_mix, gt_mix=gt_mix)
                                    for f, a, m in zip(frames, box_extract, masks)]
        else:
            crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                    for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out



# trackmix
def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class mixing_erasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels with different mixing operation.
    normal: original random erasing;
    soft: mixing ori with random pixel;
    self: mixing ori with other_ori_patch;
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='self',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def transform_image(self, img, mix_img=None):

        if random.uniform(0, 1) >= self.probability:
            return img
        # img = torch.tensor(img)
        # mix_img = torch.tensor(mix_img)
        # img = img.reshape(img.size()[2], img.size()[0], img.size()[1])
        # mix_img = mix_img.reshape(mix_img.size()[2], mix_img.size()[0], mix_img.size()[1])
        # cv.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/debug/ori_img.png", img)
        # cv.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/debug/ori_mix.png", mix_img)
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        mix_img = mix_img.reshape(mix_img.shape[2], mix_img.shape[0], mix_img.shape[1])
        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
                if self.type == 'self':
                    x2 = random.randint(0, img.shape[1] - h)
                    y2 = random.randint(0, img.shape[2] - w)
                    if mix_img is None:
                        img[:, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                    w] + m * img[:, x2:x2 + h,
                                                                    y2:y2 + w]
                    else:
                        img[:, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                    w] + m * mix_img[:, x2:x2 + h,
                                                                    y2:y2 + w]
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                            y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                       w] + m * self.mean[2]
                    else:
                        img[:, x1:x1 + h, y1:y1 +
                            w] = (1 - m) * img[:, x1:x1 + h,
                                               y1:y1 + w] + m * _get_pixels(
                                                   self.per_pixel,
                                                   self.rand_color,
                                                   (img.shape[0], h, w),
                                                   dtype=img.dtype,
                                                   device=self.device)
                img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
                # cv.imwrite("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/debug/mix_img.png", img)
                return img 
        img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        return img