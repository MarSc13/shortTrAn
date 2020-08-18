#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:53:23 2019

@author: mas32ea
"""
import numpy as np
import tifffile as tif

'''Defines the final scale and will center the new section. Prevention of signal loss at the rims 
if the former scaling is non integer multiple of the new scaling is prevented by padding process 
(addition of space to the uppper and bottom rim, as well as ,to the right and left rim)'''
def downscaling(maxx,maxy,div,field):
    
    modulox = maxx - (np.floor(maxx/div))*div #determining the overhang of the padding which will be deleted
    moduloy = maxy - (np.floor(maxy/div))*div #determining the overhang of the padding which will be deleted
    startx = np.uint16(np.floor(modulox/2))   #determine which area of the field is used for binning, considers also the padding 
    endx = maxx -(modulox-np.floor(modulox/2))#determine which area of the field is used for binning, considers also the padding
    starty = np.uint16(np.floor(moduloy/2))   #determine which area of the field is used for binning, considers also the padding
    endy = maxy -(moduloy-np.floor(moduloy/2))#determine which area of the field is used for binning, considers also the padding
    scd_len_x = np.uint16((endx - startx)/div)#defining new size if the binned field
    scd_len_y = np.uint16((endy - starty)/div)#defining new size if the binned field
    
    par_im_sec=np.zeros((2,2)) #first line shall x values, second line shall be y values
    #used for calc of conversion factor for the overlay of MORN outline and maps
    par_im_sec[0,0] = startx
    par_im_sec[0,1] = endx
    par_im_sec[1,0] = starty
    par_im_sec[1,1] = endy
      
    scaled_field = np.zeros((scd_len_x,scd_len_y),'float') #creating the shell of the binned field   
    
    for i in range(scd_len_x): # ranges trough rows of binnend field
        for j in range(scd_len_y): #ranges trough columns of binnend field 
            for g in range(i*div+startx,(i+1)*div+startx): #ranges through rows of field that shall be binnend
                for k in range(j*div+starty,(j+1)*div+starty): #ranges through columns of field that shall be binnend
                    scaled_field[i,j] = scaled_field[i,j] + field[g,k] #addition of all entries in the to be binnend 
                    #field belonging to the same px in hte binnend field

    return scaled_field, par_im_sec

'''Will be used to average over contributing entries'''
def invert_pointfield(field):
    inv_field = np.zeros((field.shape[0],field.shape[1]),'float')
    for n in range(field.shape[0]):
        for m in range(field.shape[1]):
            if field[n,m] != 0:
                inv_field[n,m] = 1 /(field[n,m])
            else:
                inv_field[n,m] = field[n,m]
    return inv_field

"""Sets the final scale. Sums all entries from underlying scale. Calculates the averaged movement direction"""
def binning(scal, directory, pointfield,vectorfield_x,vectorfield_y,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy):
    
    shape=(pointfield.shape[0],pointfield.shape[1])
    
    scaled_pointfield, par_im_sec_point = downscaling(*shape,scal,pointfield)
    scaled_pointfield_inv = invert_pointfield(scaled_pointfield) #will be used to average over the entries
    
    scaled_vectorfield_x, par_im_sec_vecx = downscaling(*shape,scal,vectorfield_x) #binning 
    scaled_vectorfield_y, par_im_sec_vecy = downscaling(*shape,scal,vectorfield_y)
                        
    scaled_tensorfield_xx, par_im_sec_tenxx = downscaling(*shape,scal,tensorfield_xx)                
    scaled_tensorfield_xy, par_im_sec_tenxy = downscaling(*shape,scal,tensorfield_xy)
    scaled_tensorfield_yx, par_im_sec_tenyx = downscaling(*shape,scal,tensorfield_yx)
    scaled_tensorfield_yy, par_im_sec_tenyy = downscaling(*shape,scal,tensorfield_yy)   
    
    scld_ten_xx = np.multiply(scaled_tensorfield_xx , scaled_pointfield_inv) #averaging over the entries for each px
    scld_ten_xy = np.multiply(scaled_tensorfield_xy , scaled_pointfield_inv)
    scld_ten_yx = np.multiply(scaled_tensorfield_yx , scaled_pointfield_inv)
    scld_ten_yy = np.multiply(scaled_tensorfield_yy , scaled_pointfield_inv)
    
    scld_vec_x = np.multiply(scaled_vectorfield_x , scaled_pointfield_inv)
    scld_vec_y = np.multiply(scaled_vectorfield_y , scaled_pointfield_inv)
    
    
    np.save(directory+'/shape',shape)
    tif.imsave(directory+'/scaled_pointfield.tif',scaled_pointfield)
    tif.imsave(directory+'/scaled_pointfield_inv.tif',scaled_pointfield_inv)
    
    tif.imsave(directory+'/scaled_vectorfield_x.tif',np.int32(scaled_vectorfield_x))
    tif.imsave(directory+'/scaled_vectorfield_y.tif',np.int32(scaled_vectorfield_y))
    
    tif.imsave(directory+'/scaled_tensorfield_xx.tif',np.int32(scaled_tensorfield_xx))
    tif.imsave(directory+'/scaled_tensorfield_xy.tif',np.int32(scaled_tensorfield_xy))
    tif.imsave(directory+'/scaled_tensorfield_yx.tif',np.int32(scaled_tensorfield_yx))
    tif.imsave(directory+'/scaled_tensorfield_yy.tif',np.int32(scaled_tensorfield_yy))
    
    np.save(directory+'/par_im_sec_cell',par_im_sec_point)
    
    tif.imsave(directory+'/scld_vec_x.tif',scld_vec_x)
    tif.imsave(directory+'/scld_vec_y.tif',scld_vec_y)
    
    tif.imsave(directory+'/scld_ten_xx.tif',scld_ten_xx)
    tif.imsave(directory+'/scld_ten_xy.tif',scld_ten_xy)
    tif.imsave(directory+'/scld_ten_yx.tif',scld_ten_yx)
    tif.imsave(directory+'/scld_ten_yy.tif',scld_ten_yy)
    
    return scaled_pointfield,scaled_pointfield_inv,scaled_vectorfield_x,scaled_vectorfield_y,scaled_tensorfield_xx,scaled_tensorfield_xy,\
           scaled_tensorfield_yx,scaled_tensorfield_yy,scld_ten_xx,scld_ten_xy,scld_ten_yx,scld_ten_yy,scld_vec_x,scld_vec_y

if __name__ == "__main__":
    scal = 160
    directory = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Results_TrcAnalysis/Trc1'
    scaled_pointfield,scaled_pointfield_inv,scaled_vectorfield_x,scaled_vectorfield_y,scaled_tensorfield_xx,scaled_tensorfield_xy,\
    scaled_tensorfield_yx,scaled_tensorfield_yy,scld_ten_xx,scld_ten_xy,scld_ten_yx,scld_ten_yy,scld_vec_x,scld_vec_y\
    =binning(scal,directory,pointfield,vectorfield_x,vectorfield_y,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy)