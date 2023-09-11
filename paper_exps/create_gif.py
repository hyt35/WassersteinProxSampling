import imageio.v3 as iio
import numpy as np
from pathlib import Path
import os
# from pygifsicle import optimize

def get_sorted_images_zoom(base_path):
    images = list()
    names = list()
    # for file in sorted(Path(base_path).rglob('*_zoom.png')):
    for file in Path(base_path).rglob('*_zoom.png'):
        if not file.is_file():
            continue
        
        images.append(iio.imread(file))
        names.append(os.path.basename(file))
    # print(names)
    return images, names


def create_gif_from_zoom(base_path, name = "comp", gif_dir = None):
    images, _ = get_sorted_images_zoom(base_path)

    frames = np.stack(images, axis=0)
    if gif_dir is None:
        gif_path = os.path.join(base_path, name+".gif")
    else:
        gif_path = os.path.join(gif_dir, name+'.gif')
    iio.imwrite(gif_path, frames)
    # optimize(gif_path)


def get_sorted_images_notzoom(base_path):
    images = list()
    names = list()
    # for file in sorted(Path(base_path).rglob('*_zoom.png')):
    for file in Path(base_path).rglob('*.png'):
        if not file.is_file():
            continue
        name = os.path.basename(file)
        if '_zoom' not in name and 'energy' not in name:
            images.append(iio.imread(file))
            names.append(name)
    # print(names)
    return images, names

def create_gif_from_notzoom(base_path, name="comp", gif_dir = None):
    images, _ = get_sorted_images_notzoom(base_path)

    frames = np.stack(images, axis=0)
    if gif_dir is None:
        gif_path = os.path.join(base_path, name+".gif")
    else:
        gif_path = os.path.join(gif_dir, name+'.gif')
    iio.imwrite(gif_path, frames)
# Scrape
# print(get_sorted_images_zoom(os.path.join('figs', 'twopole', 'fwd_diffs_1D', 'dT0.1', 'dX0.01')))
