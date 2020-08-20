#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:54:12 2019

@author: mas32ea
"""
import numpy as np
import scipy as sc
from scipy import ndimage as scim
from numpy import random 
import tifffile as tif

def remove_outliers(mask,kdim,kshape):
    
#    '''this part is/was used during programming to load and generate input data'''
#    mask = np.load('/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Example_arrys/scaled_pointfield.npy')
#    threshold = 1
#    mask[mask < threshold] =0
#    mask[mask >= threshold] =1
#    
#    '''synthetic data'''
#    syn=random.randint(2, size=(10,10))
#    mask=syn
#    
#    '''defining the kernel'''
#    kdim=3 #value has to be odd and larger than 1
#    kshape = 'box' #'box' or 'cross'
    
    if kshape=='cross' and kdim==3: #cross kernel
        fkernel = np.array([[0,1,0],[1,1,1],[0,1,0]]) #cross kernel with kernel dim 3
    elif kshape=='cross' and kdim==5: #cross kernel
        fkernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]) #cross kernel with kernel dim 5
    elif kshape=='box': #box kernel
        fkernel = np.ones((kdim,kdim))
        
    
    mask_fold = scim.convolve(mask, fkernel, mode='constant', cval=0)
    mask_fold_mply = np.multiply(mask_fold, mask)
    '''This threshold deletes isolated entries/pixels/cells and binersiation of the mask'''
    out_mask=np.ones(mask_fold_mply.shape)
    out_mask[mask_fold_mply <= 1] = 0

               
    return out_mask

"""Removes isolated pixels and pixels containing entries below the threshold by using an binary mask (Filter 1 and filter 2 Hoze,Holcman 2014)
Pixels containg less entries than the threshold and are isolated will be removed by multiplying with zero"""
def remove_isolated_px(kdim,kshape,threshold,directory,scaled_pointfield,
                       scld_ten_xx,scld_ten_xy,scld_ten_yx,scld_ten_yy,
                       scld_vec_x,scld_vec_y):
    
    #binerisation
    mask=np.zeros(scaled_pointfield.shape)
    mask[scaled_pointfield < threshold] =0
    mask[scaled_pointfield >= threshold] =1
    
    #generation of the mask for the removal of isolated px           
    out_mask = remove_outliers(mask,kdim,kshape)
    
    #application of mask to data
    scld_count_mskd = np.multiply(scaled_pointfield, out_mask)
    
    scld_ten_xx_mskd = np.multiply(scld_ten_xx , out_mask)
    scld_ten_xy_mskd = np.multiply(scld_ten_xy , out_mask)
    scld_ten_yx_mskd = np.multiply(scld_ten_yx , out_mask)
    scld_ten_yy_mskd = np.multiply(scld_ten_yy , out_mask)
    
    scld_vec_x_mskd = np.multiply(scld_vec_x , out_mask)
    scld_vec_y_mskd = np.multiply(scld_vec_y , out_mask)
    
    tif.imsave(directory+'/scld_count_mskd.tif', np.int32(scld_count_mskd).T)
    
    tif.imsave(directory+'/scld_vec_x_mskd.tif',np.int32(scld_vec_x_mskd).T)
    tif.imsave(directory+'/scld_vec_y_mskd.tif',np.int32(scld_vec_y_mskd).T)
    
    tif.imsave(directory+'/scld_ten_xx_mskd.tif',np.int32(scld_ten_xx_mskd).T)
    tif.imsave(directory+'/scld_ten_xy_mskd.tif',np.int32(scld_ten_xy_mskd).T)
    tif.imsave(directory+'/scld_ten_yx_mskd.tif',np.int32(scld_ten_yx_mskd).T)
    tif.imsave(directory+'/scld_ten_yy_mskd.tif',np.int32(scld_ten_yy_mskd).T)
    
    mask=[]
    out_mask=[]
    
    return scld_count_mskd,scld_ten_xx_mskd,scld_ten_xy_mskd,scld_ten_yx_mskd,\
scld_ten_yy_mskd,scld_vec_x_mskd,scld_vec_y_mskd

if __name__ == "__main__":
    threshold = 5
    kdim=3
    kshape='box'
    scld_count_mskd,scld_ten_xx_mskd,scld_ten_xy_mskd,scld_ten_yx_mskd,\
    scld_ten_yy_mskd,scld_vec_x_mskd,scld_vec_y_mskd=\
    remove_isolated_px(kdim,kshape,threshold,directory,scaled_pointfield,
                       scld_ten_xx,scld_ten_xy,scld_ten_yx,scld_ten_yy,
                       scld_vec_x,scld_vec_y)