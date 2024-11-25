from network import U2NET

import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt

import colorsys

import warnings

import matplotlib.pyplot as plt

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"




def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image, net, palette, device = 'cpu'):

    # img = Image.open(input_image).convert('RGB')
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    alpha_out_dir = os.path.join(opt.output,'alpha')
    cropped_out_dir = os.path.join(opt.output, 'cropped')
    cloth_seg_out_dir = os.path.join(opt.output,'cloth_seg')

    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cropped_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    # Check which classes are present in the image
    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # Save alpha masks
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        the_image = alpha_mask.reshape(768, 768, 1) * img * -1
        # Create an image object from the array
        the_img = Image.fromarray(the_image.astype('uint8'), 'RGB')
        # Save the image object
        the_img.save(os.path.join(cropped_out_dir, f'image{cls}.png'))
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_img.save(os.path.join(alpha_out_dir, f'{cls}.png'))

    # Save final cloth segmentations
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, 'final_seg.png'))
    return cloth_seg



def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print()


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def rgb_to_brightness(r, g, b):
  """Converts an RGB color value to a brightness value.

  Args:
    r: The red color value, in the range [0, 255].
    g: The green color value, in the range [0, 255].
    b: The blue color value, in the range [0, 255].

  Returns:
    A brightness value, in the range [0, 1].
  """

  # Convert the RGB color value to an HSV color value.
  h, s, v = colorsys.rgb_to_hsv(r, g, b)

  # The brightness value is the V component of the HSV color value.
  return v


def normalize_brightness(brightness_value):
    """
    Normalizes a pixel brightness value to a percentage and provides safety information.

    Args:
        brightness_value (int): The pixel brightness value (1 to 255).

    Returns:
        str: A message indicating the normalized percentage and safety information.
    """
    # Normalize to a percentage

    reflective_gear = input("Are you wearing reflective gear?: ")
    print()

    if reflective_gear == 'y' or reflective_gear == 'Y' or reflective_gear == 'Yes':
        brightness_value += 20

    safe_percentage = (brightness_value / 255) * 100
        
    unsafe_percentage = 100 - safe_percentage
    if safe_percentage > 100:
        unsafe_percentage = 0
    sizes = [safe_percentage, unsafe_percentage]
    labels = ['Safe', 'Unsafe']
    
    # Safety information based on brightness value
    if brightness_value > 200:
        safety_info = "This is safe."
    elif 120 <= brightness_value <= 200:
        safety_info = "This is moderately safe and should not be worn at night."
    else:
        safety_info = "This is unsafe to wear at night."

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])  
    plt.title(safety_info)  
    plt.show()

    return f"Normalized percentage: {safe_percentage:.2f}%\n{safety_info}"


def main(args):

    device = 'cuda:0' if args.cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(args.checkpoint_path, device=device)

    palette = get_palette(4)

    img = Image.open(args.image).convert('RGB')

    cloth_seg = generate_mask(img, net=model, palette=palette, device=device)
    

    image_path = 'output/cropped/image1.png'
    # Open the image
    img = Image.open(image_path)

    # Resize the image (optional, for faster processing)
    #img = img.resize((150, 150))

    # Convert the image to an array of RGB values
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.prod(shape[:2]), shape[2]).astype(float)
   
   
    # Filter out white pixels (assuming RGB value for white is [255, 255, 255])
    non_white_indices = np.any(ar > 20, axis=1)
    ar_non_white = ar[non_white_indices]
   
    dominant_color = np.round(np.average(ar_non_white, axis=0))

    r, g, b = dominant_color

    result = normalize_brightness(rgb_to_brightness(r, g, b))
    print(result)

    return dominant_color

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    args = parser.parse_args()

    main(args)

