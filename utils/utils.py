import numpy as np
import argparse
import os, shutil
import yaml
import torch


def load_config(config_path:str):
    with open(config_path, "r") as f:
        loaded_config = yaml.safe_load(f)
    config = argparse.Namespace()
    for key, value in loaded_config.items():
        setattr(config, key, value)

    return config


def print_config(config):
    print(f"{'Configurations ':-^80s}")
    for key, val in config.__dict__.items():
        print(f"{key}: {val}")
    print("-"*80)


def get_lip_verts(dataset):
    if dataset=="vocaset":
        return np.array([
            1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1590, 1591, 1593, 1593, 
            1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1686, 1687, 1691, 1693, 
            1694, 1695, 1696, 1697, 1700, 1702, 1703, 1704, 1709, 1710, 1711, 1712, 1713, 
            1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1728, 1729, 1730, 
            1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750, 
            1751, 1758, 1763, 1765, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 
            1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1801, 
            1802, 1803, 1804, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1865, 1866, 
            2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2726, 2726, 2727, 2729, 2729, 
            2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810, 
            2811, 2812, 2813, 2814, 2817, 2819, 2820, 2821, 2826, 2827, 2828, 2829, 2830, 
            2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2843, 2844, 2845, 
            2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865, 
            2866, 2869, 2871, 2873, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 
            2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2904, 
            2905, 2906, 2907, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2948, 2949, 
            3503, 3504, 3506, 3509, 3511, 3512, 3513, 3531, 3533, 3537, 3541, 3543, 3546, 
            3547, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 
            3802, 3803, 3804, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 
            3922, 3923, 3924, 3925, 3926, 3927, 3928
        ])
    elif dataset=="BIWI":
        with open("BIWI/lve.txt") as f:
            maps = f.read().split(", ")
            mouth_map = [int(i) for i in maps]
        return np.array(mouth_map)


def make_dirs(dir_path:str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def batch_orth_proj(X, camera):
    """orthgraphic projection
    
    Args:
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    Code from auto_avsr/datamodule/av_dataset.py

    Args:
        data (torch.Tensor): [length_data, 1]
        size (int): size of reference data
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data