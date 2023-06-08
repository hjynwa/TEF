#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:06:48 2022

@author: hanjin
"""

import numpy as np
import pandas as pd
from metavision_core.event_io import RawReader
import cv2
import matplotlib.pyplot as plt
import os
import json 
import scipy.io as scio


def evs_convert(evs_arr, t_start, t_end, mask_coor):
    posi = evs_arr[:,0] > t_start
    evs_arr = evs_arr[posi]
    posi = evs_arr[:,0] < t_end
    evs_arr = evs_arr[posi]
    posi = evs_arr[:,1] >= mask_coor[0][1]
    evs_arr = evs_arr[posi]
    posi = evs_arr[:,1] < mask_coor[1][1]
    evs_arr = evs_arr[posi]
    
    posi = evs_arr[:,2] >= mask_coor[0][0]
    evs_arr = evs_arr[posi]
    posi = evs_arr[:,2] < mask_coor[1][0]
    evs_arr = evs_arr[posi]
    
    evs_filter = evs_arr
    return evs_filter

    
def plot_num_ev_mean(mask_evs, mask, start_time, end_time, time_interval):
    mask_evs = np.array(mask_evs)
    mask_size = len(np.where(mask == 1)[0])
    evs_t = mask_evs[:,0]
    time_slots = np.arange(start_time, end_time, time_interval)
    num_ev_mean = []
    for time_slot in time_slots:
        evs_part = evs_t[evs_t < time_slot]
        num_ev = evs_part.shape[0]
        num_ev_mean.append(num_ev/mask_size)
        
    return num_ev_mean


def get_patch_timestamp(evs_src, mask, start_time, end_time):
    # filter events
    posi = evs_src[:,0] > start_time 
    evs = evs_src[posi]
    posi = evs[:,0] < end_time 
    evs = evs[posi]
    posi = evs[:,1] < end_time 
    evs = evs[posi]
    
    # make a list of mask positions 
    mask_h, mask_w = np.where(mask == 1)
    mask_len = mask_h[-1] - mask_h[0] + 1
    mask_posi = []
    for i in range(mask_h.shape[0]):
        tp = (mask_h[i], mask_w[i])
        mask_posi.append(tp)
    
    # make a list of timestamps w.r.t to position list 
    mask_timestamp = [[] for i in range(len(mask_h))]
    for i, ev_item in enumerate(evs):
        x = ev_item[1].astype(np.int16)
        y = ev_item[2].astype(np.int16)
        ts = ev_item[0]
        xmod = np.mod(x, mask_len)
        ymod = np.mod(y, mask_len)
        index = ymod*mask_len+xmod
        mask_timestamp[index].append(ts)

    # calculate the mean time interval of each pixel
    ev_intervals = []
    for i, pixel_evs in enumerate(mask_timestamp):
        diffs = np.diff(np.array(pixel_evs))
        diff = np.mean(diffs)
        if np.isnan(diff):
            diff = max_interval
        ev_intervals.append(diff)
    
    # re-arrange the time interval to form the timestamp_map
    timestamp_map = np.zeros([height, width]).ravel()
    for i, ev_interval in enumerate(ev_intervals):
        if ev_interval == 0:
            ev_interval = (ev_intervals[i-1] + ev_intervals[i+1]) / 2
        np.add.at(timestamp_map, mask_posi[i][1] + mask_posi[i][0] * width, ev_interval)
    timestamp_map = np.reshape(timestamp_map, (height, width))
    
    return timestamp_map


def get_mask_labelme(json_path):
    tmp = {}
    with open(json_path, "r") as f:
        tmp = f.read()
    tmp = json.loads(tmp)
    points = tmp["shapes"][0]["points"]
    points = np.array(points, np.int32) // 2 # // 2, because the original size of mask is [720, 1280]
    mask = np.zeros([height, width])
    cv2.fillPoly(mask, [points], 1)
    
    return mask.astype(np.bool_)
    

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', dest='mode', type=str, help="Choose a task from [hyperspectral, depth, iso-contour].", required=True)
    parser.add_argument('-i', '--input_dir', dest='input_dir', type=str, help="Path to input dir.", required=True)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dirname = args.input_dir
    mode = args.mode
    
    src_dir = '%s/events' % dirname 
    evs_files = sorted(os.listdir(src_dir))
    
    # You can adjust the following parameters for different event files
    lb_time = 100000 # left time bound
    rb_time = 4000000 # right time bound
    time_interval = 10000
    max_interval = 2.5e+06 # if there's no consecutive events, set as max_interval to avoid NaN 
    height, width = 360, 640
    mask_len = 2 # the size of each small patch for computing TEFs
    
    
    if mode == 'hyperspectral':
        whole_mask = np.ones([height, width]).astype(np.bool_) # the mask of the whole image plane
    else:
        mask_path = '%s/mask.json' % dirname
        whole_mask = get_mask_labelme(mask_path) # the mask of the whole object, got from labelme
    
    mask_patches_coors = np.where(whole_mask == 1) # the coordinates of each pixel in the whole mask
    mask_patches = []
    for k in range(len(mask_patches_coors[0])):
        if np.mod(mask_patches_coors[0][k], mask_len) == 0 and np.mod(mask_patches_coors[1][k], mask_len) == 0:
            mask_patches.append((mask_patches_coors[0][k], mask_patches_coors[1][k])) # append the top left pixel's coordinate of each patch
    
    num_patches = len(mask_patches)
    ev_interval_map = np.zeros([height, width, len(evs_files)])
    
    for i, evs_file in enumerate(evs_files):
        # load .raw files
        if evs_file.endswith('.raw'):
            reader = RawReader(os.path.join(src_dir, evs_file), max_events=int(1e8))
            evs_raw = reader.load_delta_t(1e8)
            num_evs = evs_raw['t'].shape[0]
            evs_arr = np.zeros([num_evs, 4])
            evs_arr[:,0] = evs_raw['t']
            evs_arr[:,1] = evs_raw['x']
            evs_arr[:,2] = evs_raw['y']
            evs_arr[:,3] = evs_raw['p']
        
        # load .csv files
        elif evs_file.endswith('.csv'):
            evs_raw = pd.read_csv(os.path.join(src_dir, evs_file), 
                                        header=None, 
                                        names=['t', 'x', 'y', 'p'],
                                        dtype={'t': np.int64, 'x': np.int16, 'y': np.int16, 'p': np.int16}, 
                                        engine='c', nrows=None)
            num_evs = evs_raw['t'].values.shape[0]
            evs_arr = np.zeros([num_evs, 4])
            evs_arr[:,0] = evs_raw['t'].values
            evs_arr[:,1] = evs_raw['x'].values
            evs_arr[:,2] = evs_raw['y'].values
            evs_arr[:,3] = evs_raw['p'].values
        
        for j, patch in enumerate(mask_patches[:]):
            mask = np.zeros([height, width], dtype=np.bool_)
            mask[patch[0]:patch[0]+mask_len, patch[1]:patch[1]+mask_len] = 1
            mask_coor = [[patch[0],patch[1]], [patch[0]+mask_len,patch[1]+mask_len]]
            
            # get all the events in the 2x2 patches
            mask_evs = evs_convert(evs_arr, lb_time, rb_time, mask_coor)
            
            time_slots = np.arange(lb_time, rb_time, time_interval)

            if len(mask_evs) <= mask_len**2:
                ev_interval_map[patch[0]:patch[0]+mask_len, patch[1]:patch[1]+mask_len, i] = max_interval
            else:
                # get the TEF curve of each patch
                num_ev_mean = plot_num_ev_mean(mask_evs, mask, lb_time, rb_time, time_interval)   
                diff = np.diff(np.array(num_ev_mean))
                diff = diff / diff.max()
                slope_threshold = np.where(diff > 0.3)
                if len(slope_threshold[0]) < 2:
                    ev_interval_map[patch[0]:patch[0]+mask_len, patch[1]:patch[1]+mask_len, i] = max_interval
                    continue
                start_posi = slope_threshold[0][0]
                start_time = lb_time + (start_posi * time_interval) # start time of constant frequency
                end_posi = slope_threshold[0][-1]
                end_time = lb_time + (end_posi * time_interval) # end time of constant frequency
                
                timestamp_map = get_patch_timestamp(mask_evs, mask, start_time, end_time)
                timestamp_map_mask = timestamp_map[patch[0]:patch[0]+mask_len, patch[1]:patch[1]+mask_len]
                timestamp_map_mask_valid = timestamp_map_mask[np.where((timestamp_map_mask != 0) & (timestamp_map_mask != max_interval))]
                ev_interval = np.mean(timestamp_map_mask_valid)
                ev_interval_map[patch[0]:patch[0]+mask_len, patch[1]:patch[1]+mask_len, i] = ev_interval
            if np.mod(j, 200) == 0:
                print("%d / %d ... (No. %d event file)" % (j, len(mask_patches), i+1))
    
    # save results
    np.save('%s/interval_map_%dx%d_len%d.npy' % (args.input_dir, height, width, mask_len), ev_interval_map)
    
    # get TEF map
    if mode == 'hyperspectral':
        tef_map = 1/ev_interval_map
    else:
        tef_map = np.zeros_like(ev_interval_map)
        one_posi = np.where(whole_mask == 1)
        for i in range(len(one_posi[0])):
            if ev_interval_map[one_posi[0][i], one_posi[1][i]] != 0:
                tef_map[one_posi[0][i], one_posi[1][i], :] = 1/ev_interval_map[one_posi[0][i], one_posi[1][i], :]
            
    np.save('%s/ev_radiance_%dx%d_len%d.npy' % (args.input_dir, height, width, mask_len), tef_map)
    scio.savemat('%s/ev_radiance_%dx%d_len%d.mat' % (args.input_dir, height, width, mask_len), {'ev_radiance': tef_map})
    
    
    
    
    
    
    
    
    