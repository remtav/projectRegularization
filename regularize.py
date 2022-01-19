print('Importing packages and modules...')
import argparse
import multiprocessing
import logging
import logging.config

import numpy as np
import torch
from skimage import measure
from skimage.transform import rescale
from torch import nn

np.random.seed(1234)  # Set random seed for reproducibility
import rasterio
import time

from pathlib import Path
from tqdm import tqdm

logging.getLogger(__name__)

print('Done')

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=8, in_features=256):
        super(GeneratorResNet, self).__init__()

        out_features = in_features

        model = []

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # model += [nn.ReflectionPad2d(2), nn.Conv2d(out_features, 2, 7), nn.Softmax()]
        model += [nn.Conv2d(out_features, 2, 7, stride=1, padding=3), nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, feature_map):
        x = self.model(feature_map)
        return x


class Encoder(nn.Module):
    def __init__(self, channels=3 + 2):
        super(Encoder, self).__init__()

        # Initial convolution block
        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, 7, stride=1, padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
            ]
            in_features = out_features

        self.model = nn.Sequential(*model)

    def forward(self, arguments):
        x = torch.cat(arguments, dim=1)
        x = self.model(x)
        return x


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def predict_building(rgb, mask, model):
    Tensor = torch.cuda.FloatTensor

    mask = to_categorical(mask, 2)

    rgb = rgb[np.newaxis, :, :, :]
    mask = mask[np.newaxis, :, :, :]

    E, G = model

    rgb = Tensor(rgb)
    mask = Tensor(mask)
    rgb = rgb.permute(0, 3, 1, 2)
    mask = mask.permute(0, 3, 1, 2)
    # print(rgb.shape)
    # print(mask.shape)

    rgb = rgb / 255.0

    # PREDICTION
    pred = G(E([rgb, mask]))
    pred = pred.permute(0, 2, 3, 1)

    pred = pred.detach().cpu().numpy()

    pred = np.argmax(pred[0, :, :, :], axis=-1)
    return pred


def fix_limits(i_min, i_max, j_min, j_max, min_image_size=256):
    def closest_divisible_size(size, factor=4):
        while size % factor:
            size += 1
        return size

    height = i_max - i_min
    width = j_max - j_min

    # pad the rows
    if height < min_image_size:
        diff = min_image_size - height
    else:
        diff = closest_divisible_size(height) - height + 16

    i_min -= (diff // 2)
    i_max += (diff // 2 + diff % 2)

    # pad the columns
    if width < min_image_size:
        diff = min_image_size - width
    else:
        diff = closest_divisible_size(width) - width + 16

    j_min -= (diff // 2)
    j_max += (diff // 2 + diff % 2)

    return i_min, i_max, j_min, j_max


def regularization(rgb, ins_segmentation, model, in_mode="instance", out_mode="instance", min_size=10, build_val=255):
    assert in_mode == "semantic"
    assert out_mode == "instance" or out_mode == "semantic"

    border = 256

    if rgb is None:
        logging.debug(ins_segmentation.shape)
        rgb = np.zeros((ins_segmentation.shape[0], ins_segmentation.shape[1], 3), dtype=np.uint8)

    print('Padding...')
    ins_segmentation = np.pad(array=ins_segmentation, pad_width=border, mode='constant', constant_values=0)
    npad = ((border, border), (border, border), (0, 0))
    rgb = np.pad(array=rgb, pad_width=npad, mode='constant', constant_values=0)
    print('Done')

    print('Counting buildings...')
    contours_list = measure.find_contours(ins_segmentation)

    max_instance = len(contours_list)
    print(f'Found {max_instance} buildings!')

    regularization = np.zeros(ins_segmentation.shape, dtype=np.uint16)

    for ins in tqdm(range(1, max_instance + 1), desc="Regularization"):
        indices = contours_list[ins - 1]
        building_size = indices.shape[0]
        if building_size > min_size:
            i_min = int(np.amin(indices[:, 0]))
            i_max = int(np.amax(indices[:, 0]))
            j_min = int(np.amin(indices[:, 1]))
            j_max = int(np.amax(indices[:, 1]))

            i_min, i_max, j_min, j_max = fix_limits(i_min, i_max, j_min, j_max)

            if (i_max - i_min) > 10000 or (j_max - j_min) > 10000:
                continue

            mask = np.copy(ins_segmentation[i_min:i_max, j_min:j_max] == build_val)

            rgb_mask = np.copy(rgb[i_min:i_max, j_min:j_max, :])

            max_building_size = 768
            rescaled = False
            if mask.shape[0] > max_building_size and mask.shape[0] >= mask.shape[1]:
                f = max_building_size / mask.shape[0]
                mask = rescale(mask, f, anti_aliasing=False, preserve_range=True)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, multichannel=True)
                rescaled = True
            elif mask.shape[1] > max_building_size and mask.shape[1] >= mask.shape[0]:
                f = max_building_size / mask.shape[1]
                mask = rescale(mask, f, anti_aliasing=False)
                rgb_mask = rescale(rgb_mask, f, anti_aliasing=False, preserve_range=True, multichannel=True)
                rescaled = True

            pred = predict_building(rgb_mask, mask, model)

            if rescaled:
                pred = rescale(pred, 1 / f, anti_aliasing=False, preserve_range=True)

            pred_indices = np.argwhere(pred != 0)

            if pred_indices.shape[0] > 0:
                pred_indices[:, 0] = pred_indices[:, 0] + i_min
                pred_indices[:, 1] = pred_indices[:, 1] + j_min
                x, y = zip(*pred_indices)
                if out_mode == "semantic":
                    regularization[x, y] = 1
                else:
                    regularization[x, y] = ins

    return regularization[border:-border, border:-border]


def arr_threshold(arr, value=127):
    bool_M = (arr >= value)
    arr[bool_M] = 255
    arr[~bool_M] = 0
    # print(np.unique(M))
    return arr


def regularize_buildings(pred_arr, sat_img_arr=None, apply_threshold=None, build_val=255):
    if apply_threshold:
        print('Applying threshold...')
        pred_arr = arr_threshold(pred_arr, value=apply_threshold)
    logging.debug(pred_arr.shape)

    print('Done')
    model_encoder = Path("saved_models_gan") / "E140000_e1"
    model_generator = Path("saved_models_gan") / "E140000_net"
    E1 = Encoder()
    G = GeneratorResNet()
    G.load_state_dict(torch.load(model_generator))
    E1.load_state_dict(torch.load(model_encoder))
    E1 = E1.cuda()
    G = G.cuda()

    model = [E1, G]
    R = regularization(sat_img_arr, pred_arr, model, in_mode="semantic", out_mode="semantic", build_val=build_val)
    return R


def main(in_raster, out_raster, sat_img=None, build_val=255, apply_threshold=False):
    """
    -------
    :param params: (dict) Parameters found in the yaml config file.
    """
    start_time = time.time()
    in_raster = Path(in_raster)
    if not in_raster.is_file():
        raise FileNotFoundError(f"Input inference raster not a file: {in_raster}")
    out_raster = Path(out_raster)

    try:
        with rasterio.open(in_raster, 'r') as raw_pred:
            outname_reg = Path(out_raster)
            logging.debug(f'Regularizing buildings in {in_raster}...')
            meta = raw_pred.meta
            raw_pred_arr = raw_pred.read()[0, ...]
            raw_pred_arr_buildings = np.zeros(shape=raw_pred_arr.shape, dtype=np.uint8)
            raw_pred_arr_buildings[raw_pred_arr == build_val] = build_val
            raw_pred_arr[raw_pred_arr == build_val] = 0
            reg_arr = regularize_buildings(raw_pred_arr_buildings, sat_img_arr=sat_img, apply_threshold=apply_threshold,
                                           build_val=build_val)
            reg_arr[reg_arr == 1] = build_val
            raw_pred_arr[reg_arr == build_val] = build_val
            out_arr = raw_pred_arr[np.newaxis, :, :]

            meta.update({"dtype": 'uint8', "compress": 'lzw'})
            with rasterio.open(outname_reg, 'w+', **meta) as out:
                logging.info(f'Successfully regularized on {in_raster}\nWriting to file: {outname_reg}')
                out.write(out_arr.astype(np.uint8))
    except IOError as e:
        logging.error(f"Failed to regularize {in_raster}\n{e}")

    logging.info(f"End of process. Elapsed time: {int(time.time() - start_time)} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buildings inference regularization')
    parser.add_argument('--input-inf', help='Input inference as raster containing building predictions ')
    parser.add_argument('--input-rgb', default=None, help='Input RGB raster used to produce the inference')
    parser.add_argument('--output', help='Output inference as raster containing regularized building predictions ')
    input_type = parser.add_mutually_exclusive_group()
    input_type.add_argument('--build-val', default=255,
                        help='Pixel value corresponding to building prediction in input raster')
    input_type.add_argument('--threshold-val', default=None,
                            help='If input raster contains a heatmap, a threshold will be applied at this value. Above'
                                 'this value, all pixels will be considered as buildings and below, background.')
    args = parser.parse_args()
    print(f'\n\nStarting building regularization with {args.input_inf}\n\n')
    main(in_raster=args.input_inf,
         out_raster=args.output,
         sat_img=args.input_rgb,
         build_val=args.build_val,
         apply_threshold=int(args.threshold_val))
