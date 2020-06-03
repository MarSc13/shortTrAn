#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:30:34 2019

@author: mas32ea
"""
import numpy as np
import scipy as sc
import tifffile as tif
from scipy import ndimage
import cv2


def average_box_filter(data, scaled_pointfield, kdim, outliers, mode_outliers):
    
    
    r=data.shape[0]
    c=data.shape[1]
    
    #padding
    begin=int((kdim-1)/2) #postion upper left corner of data array in padded array 
    data1=np.zeros((r+kdim-1,c+kdim-1)) #postion of the kernel center needs to be neglected in creating the padded array
    data1[begin:r+begin,begin:c+begin]=data #positioning of the data array in the padded array
    
    scaled_pointfield1=np.zeros((r+kdim-1,c+kdim-1)) #postion of the kernel center needs to be neglected in creating the padded array
    scaled_pointfield1[begin:r+begin,begin:c+begin]=scaled_pointfield #positioning of the pointfield array in the padded array
    
    smoothed_data=np.zeros((r,c))
    
    sur=int((kdim-1)/2) #number of pixels surrounding the center
    
    for a in range(begin,r+begin): #iterates through rows, excludes rows and columns generated for the padding
        for b in range(begin,c+begin): #iterates through columns, excludes rows and columns generated for the padding
#            a=10
#            b=2
            val_ar=data1[a-sur:a+sur+1,b-sur:b+sur+1] #python exludes the number in iteration -> additional +1
            kernel=scaled_pointfield1[a-sur:a+sur+1,b-sur:b+sur+1] #extraction of mask area to normalize kernel
            weig_val=kernel*val_ar #generation of the weighted val_array
            if kernel.sum() == 0:
                end_val = 0
            else:
                end_val=weig_val.sum()/kernel.sum() #calc of count weighted averaged value
            smoothed_data[a-sur,b-sur]=end_val
    
    #generation binary mask to remove added pixels at the rim
    pnt_pad=np.uint8(scaled_pointfield1.copy())
    pnt_pad[pnt_pad>0]=255
    
    out = np.zeros((r+4,c+4),np.uint8)
    
    cv2.floodFill(pnt_pad,out,(0,0),255)
    out = cv2.bitwise_not(out)
    
    if mode_outliers == 'remove': #removal of the outer pixel
        #Binerisation
        out[out < 255] = 0
        out[out == 255] = 1
        #generation of distance array to remove outliers 
        #distance array is then binerised and serves as mask  
        dist_ar = ndimage.distance_transform_cdt(out)
        dist_ar = dist_ar[2:r+2,2:c+2]
        mask = np.zeros((dist_ar.shape[0],dist_ar.shape[1]))
        mask[dist_ar > outliers] = 1
        #output data    
        smoothed_data_fin=np.multiply(mask,smoothed_data)
        
        
    elif mode_outliers == 'gauss': #outer pixel will additionally gauss smoothed
        out[out < 255] = 0
        out[out == 255] = 1
        #dist array
        dist_ar = ndimage.distance_transform_cdt(out)
        dist_ar = dist_ar[2:r+2,2:c+2]
        
        sigma_div = 1 #radius of gaussianfilter is 4 px with sigma 1
        gaus_data= ndimage.gaussian_filter(smoothed_data,sigma=sigma_div,mode='constant',cval=0)
        #generation of mask for the structure center
        rim_mask = np.ones((dist_ar.shape[0],dist_ar.shape[1]))
        rim_mask[dist_ar > outliers] = 0
        rim_mask[dist_ar == 0] = 0
        #generation of the mask for the outlier/rim
        cen_mask = np.zeros((dist_ar.shape[0],dist_ar.shape[1]))
        cen_mask[dist_ar > outliers] = 1
        #mulitplication with the smoothed data
        #will loose entries outside the cell 
        #and uses gauss filtered data at the rim
        gaus_data_final = np.multiply(gaus_data,rim_mask)
        cen_data_final = np.multiply(smoothed_data,cen_mask)
        #output data
        smoothed_data_fin=np.zeros((gaus_data_final.shape[0],gaus_data_final.shape[1]))
        smoothed_data_fin = gaus_data_final + cen_data_final

    return(smoothed_data_fin)
    


'''Uses at the moment a for the entries wheigted average filter to smooth the signal to the neighbouring pixels'''    
def smoothing(kdim_smoothing,wdth_outliers,mode_outliers,directory,scaled_pointfield,scld_vec_x_mskd, scld_vec_y_mskd,scld_ten_xx_mskd,scld_ten_xy_mskd,scld_ten_yx_mskd,scld_ten_yy_mskd):
    
    #application of the smoothing filter
    scld_vec_x_fil = average_box_filter(scld_vec_x_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)
    scld_vec_y_fil = average_box_filter(scld_vec_y_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)
    
    scld_ten_xx_fil = average_box_filter(scld_ten_xx_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)
    scld_ten_xy_fil = average_box_filter(scld_ten_xy_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)
    scld_ten_yx_fil = average_box_filter(scld_ten_yx_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)
    scld_ten_yy_fil = average_box_filter(scld_ten_yy_mskd,scaled_pointfield,kdim_smoothing,wdth_outliers,mode_outliers)    
    
    #saving
    tif.imsave(directory+'/scld_vec_x_fil.tif',scld_vec_x_fil.T) 
    tif.imsave(directory+'/scld_vec_y_fil.tif',scld_vec_y_fil.T)
    
    tif.imsave(directory+'/scld_ten_xx_fil.tif',scld_ten_xx_fil.T)
    tif.imsave(directory+'/scld_ten_xy_fil.tif',scld_ten_xy_fil.T)
    tif.imsave(directory+'/scld_ten_yx_fil.tif',scld_ten_yx_fil.T)
    tif.imsave(directory+'/scld_ten_yy_fil.tif',scld_ten_yy_fil.T)
    
    return scld_vec_x_fil,scld_vec_y_fil,scld_ten_xx_fil,scld_ten_xy_fil,scld_ten_yx_fil,scld_ten_yy_fil

if __name__ == "__main__":
    kdim_smoothing = 3 #kernel dimension
    wdth_outliers = 1
    mode_outliers = 'gauss' # or 'remove'
    #data
    path= '/Users/marieschwebs/Documents/GitHub/EvalTrc/smoothing_examp/'
    scaled_pointfield=tif.imread(path +'scaled_pointfield.tif').T # did not perfom the transposion yet
    scld_ten_xx_mskd=tif.imread(path+'scld_ten_xx_mskd.tif')
    scld_ten_xy_mskd=tif.imread(path+'scld_ten_xy_mskd.tif')
    scld_ten_yx_mskd=tif.imread(path+'scld_ten_yx_mskd.tif')
    scld_ten_yy_mskd=tif.imread(path+'scld_ten_yy_mskd.tif')
    scld_vec_x_mskd=tif.imread(path+'scld_vec_x_mskd.tif')
    scld_vec_y_mskd=tif.imread(path+'scld_vec_y_mskd.tif')
    #directory to save results
    directory= '/Users/marieschwebs/Documents/GitHub/EvalTrc/smoothing_examp/'
    
    scld_vec_x_fil,scld_vec_y_fil,scld_ten_xx_fil,scld_ten_xy_fil,scld_ten_yx_fil,scld_ten_yy_fil=\
    smoothing(kdim_smoothing,wdth_outliers,mode_outliers,directory,scaled_pointfield,scld_vec_x_mskd, 
              scld_vec_y_mskd,scld_ten_xx_mskd,scld_ten_xy_mskd,scld_ten_yx_mskd,scld_ten_yy_mskd)