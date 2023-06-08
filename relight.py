#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:49:04 2022

@author: hanjin
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 
from scipy import interpolate


def get_rgb(illumination_spectral, camera_SRF, reflectance, resolution=33):
    h, w = reflectance.shape[:2]
    reflectance = reflectance.reshape([-1, resolution])
    result_rgb = np.zeros([reflectance.shape[0], 3])
    for i, pixel_ref in enumerate(reflectance):
        r = np.sum(illumination_spectral*camera_SRF[0]*pixel_ref)
        g = np.sum(illumination_spectral*camera_SRF[1]*pixel_ref)
        b = np.sum(illumination_spectral*camera_SRF[2]*pixel_ref)
        result_rgb[i] = [r,g,b]
        
    return result_rgb.reshape([h,w,3])


def read_camera_srf(srf_path):
    camera_SRF = np.zeros([3,33]) 
    camera_idx = 24
    with open(srf_path) as f:
        for i, line in enumerate(f.readlines()[camera_idx*4: camera_idx*4+4]):
            if i == 0:
                print(line)
            if i == 1:
                camera_SRF[0] = np.fromstring(line, dtype=np.float32, sep=' ')
            elif i == 2:
                camera_SRF[1] = np.fromstring(line, dtype=np.float32, sep=' ')
            elif i == 3:
                camera_SRF[2] = np.fromstring(line, dtype=np.float32, sep=' ')
    return camera_SRF


def interplate_1d(x, y):
    f_interp = interpolate.interp1d(x, y)
    wv_new = np.arange(400, 730, 10)
    y_new = f_interp(wv_new)
    
    return y_new


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir', dest='input_dir', type=str, help="Path to input dir.", required=True)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dirname = args.input_dir
    
    tef_map = np.load(os.path.join(dirname, 'ev_radiance_360x640_len2.npy'))
    
    # Load the target illumination file
    df = pd.read_csv('lighting_and_SRF/lighting_files/Full.txp', engine = "python", sep = "\t")
    illumination_wv = df.values[:, 0]
    illumination_spectral = df.values[:, 1]
    illumination_new = interplate_1d(illumination_wv, illumination_spectral)
    
    # Load the illumination when capturing the event signals
    wv = np.arange(400, 730, 20)
    srf_illumination = np.load('lighting_and_SRF/lighting_files/lighting_of_capture.npy')[1:-2]
    srf_illumination_new = interplate_1d(wv, srf_illumination)
    
    # Camera SRF database from http://www.gujinwei.org/research/camspec/db.html
    rgbcam_srf_path = 'lighting_and_SRF/camera_SRF/rgb_cameras_SRF_database.txt'
    camera_SRF = read_camera_srf(rgbcam_srf_path)    
    camera_SRF = camera_SRF / camera_SRF.max()
    
    # Load the event camera spectral response function
    evcam_srf_path = 'lighting_and_SRF/camera_SRF/event_camera_SRF.npy'
    evcam_srf = np.load(evcam_srf_path)[1:-2] # 400-720
    evcam_srf_new = interplate_1d(wv, evcam_srf)
    
    img_reflectance = np.zeros([360*640, 33])
    tef_map = tef_map.reshape([-1, 17])
    num_pixels = tef_map.shape[0]
    for i in range(num_pixels):
        tef_interp = interplate_1d(wv, tef_map[i])
        img_reflectance[i] = tef_interp / (srf_illumination_new * evcam_srf_new)
    img_reflectance = img_reflectance.reshape([360, 640, 33])
    
    result_rgb = get_rgb(illumination_new, camera_SRF, img_reflectance)
    result_norm = result_rgb / result_rgb.max()
    # result_norm = np.zeros_like(result_rgb)
    # result_norm[:,:,0] = result_rgb[:,:,0] / result_rgb[:,:,0].max()
    # result_norm[:,:,1] = result_rgb[:,:,1] / result_rgb[:,:,1].max()
    # result_norm[:,:,2] = result_rgb[:,:,2] / result_rgb[:,:,2].max()
    
    plt.imshow(result_norm)



