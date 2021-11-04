import glob
import os
import shutil
import pickle
from typing import Any, List, Optional, Union, Tuple
from scipy.ndimage import affine_transform
from cfg import Opts
import numpy as np
from mlutils import Log
from PIL import Image


ESP = 1.0e-7
image_ext_list = ['jpg', 'png', 'JPG', 'PNG']


def transform_matric(scale: List[float]) -> np.ndarray:
    assert len(scale) == 2, f'len(sclae) = {len(scale)} != 2'

    resize_axis_matrix = np.array(
        [[1 / scale[0],     0.,            0.],
            [0.,          1 / scale[1],       0.],
            [0.,               0.,            1.]])

    return resize_axis_matrix

def resize_by(
    inp: np.ndarray,
    size: Optional[Union[int, Tuple[int, int], List[int]]]=None
) -> np.ndarray:
    if isinstance(size, int):
        size = [size, size]
    else:
        size = size

    height = inp.shape[1]
    width = inp.shape[2]

    scale = (size[0] / height,
             size[1] / width)

    affine_matrix = transform_matric(scale)
    inp_ = []
    for i in range(inp.shape[0]):

        c_inp_min = inp[i].min()
        c_inp_max = inp[i].max()
        c_inp = affine_transform(
            inp[i], affine_matrix, output_shape=size)
        c_inp = np.clip(c_inp, a_min=c_inp_min, a_max=c_inp_max)
        inp_.append(c_inp)

    inp = np.stack(inp_, axis=0)
    return inp


def pad_to_square(image):
    C, H, W = image.shape
    width = max(H, W)
    out = np.zeros((C, width, width))
    out[:, 0:H, 0:W] = image
    return out


def resize(img, size, sq=False):
    if sq:
        img = pad_to_square(img)
    return resize_by(img, size)


def imread(opt: Opts, path: str) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB")
    img= np.asarray(img)
    if opt.get('enable_resize', False):
        if opt.get('scale_resize', False):
            img = np.transpose(img, (2, 0, 1))
            _, H, W = img.shape
            min_width = min(H, W)
            scale = opt.input_size / min_width

            size = [int(H * scale), int(W * scale)]
            img = resize(img, size, sq=False)
            img = np.transpose(img, (1, 2, 0))
        else:
            img = np.transpose(img, (2, 0, 1))
            sq = opt.get('enable_sq', False)

            img = resize(img, opt.input_size, sq)
            img = np.transpose(img, (1, 2, 0))
    return img


def save_pickle(path: str, obj: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_npy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def save_npy_uint8(path: str, arr: np.ndarray) -> None:
    arr = arr.astype(np.uint8)
    np.save(path, arr)


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def find_files_by_ext(path: str, ext: str) -> List[str]:
    file_list = []

    for file_path in glob.glob(os.path.join(path, f'*.{ext}')):
        file_list.append(file_path)

    return file_list


def find_files_by_path(path: str) -> List[str]:
    file_list = []

    for ext in image_ext_list:
        file_list += find_files_by_ext(path, ext=ext)

    return file_list


def translate_image_to_npy(
    opt: Opts,
    image_path: str,
    npy_root: str
) -> None:
    image_name = image_path.split('/')[-1].split('.')[0]
    npy_path = os.path.join(npy_root, f'{image_name}.npy')

    image = imread(opt, image_path)
    image = np.transpose(image, (2, 0, 1))
    save_npy_uint8(npy_path, image)


def verify_images(opt: Opts):
    real_path = os.path.join(opt.data_root, 'real')
    fake_path = os.path.join(opt.data_root, 'fake')

    real_images = find_files_by_path(real_path)
    fake_images = find_files_by_path(fake_path)
    all_images = real_images + fake_images

    for image_path in all_images:
        try:
            img = Image.open(image_path)
        except IOError:
            print(image_path)
        try:
            img= np.asarray(img)
        except:
            print('corrupt img', image_path)


#######################
#  Data Processing 
####################### 
def process_data_translate(opt: Opts):
    npy_root = opt.gen_root
    if os.path.exists(npy_root):
        shutil.rmtree(npy_root)
    os.mkdir(npy_root)

    real_path = os.path.join(opt.data_root, 'real')
    fake_path = os.path.join(opt.data_root, 'fake')

    real_images = find_files_by_path(real_path)
    fake_images = find_files_by_path(fake_path)

    for i, image_path in enumerate(real_images):
        translate_image_to_npy(opt, image_path, npy_root)
        print(f'real: {i}/{len(real_images)}', end='\r')
    print('')

    for i, image_path in enumerate(fake_images):
        translate_image_to_npy(opt, image_path, npy_root)
        print(f'fake: {i}/{len(fake_images)}', end='\r')
    print('')


def process_data(opt: Opts) -> None:
    process_data_translate(opt)

    root_path = os.path.join(opt.gen_root)
    all_images = find_files_by_ext(root_path, ext='npy')

    real_images = []
    fake_images = []
    for image_path in all_images:
        image_name = image_path.split('/')[-1]
        if 'real' in image_name:
            real_images.append(image_path)
        elif 'fake' in image_name:
            fake_images.append(image_path)
        else:
            raise ValueError(f'Unrecognized file {image_path}')

    Log.info(f'All images length: {len(all_images)}')
    Log.info(f'All real images length: {len(real_images)}')
    Log.info(f'All fake images length: {len(fake_images)}')

    # save meta file
    meta = {
        'real_images': real_images,
        'fake_images': fake_images
    }

    save_pickle(opt.meta_path, meta)
    Log.info('done.')
