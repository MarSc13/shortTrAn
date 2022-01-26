#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:35:17 2019

@author: mas32ea
"""
import numpy as np
import os as os
import shutil as sh
import tifffile as tif
#import debug


'''Creates a count field from trac_array which will later used to generate the mask by going through every loc and placing it into the pointfield'''
def create_pointfield(img_array,trac_array): 
    for j in range(trac_array.shape[0]):
        img_array[trac_array[j][2]][trac_array[j][3]] =img_array[trac_array[j][2]][trac_array[j][3]] + 1
    return img_array

'''Creates a matrix(table) containg the differences of consecutive x or y pos of each trajectory'''
def create_diff_array(array):
    diff_array= np.zeros(array.shape[0]-1,'float')
    for i in range(diff_array.shape[0]):
        diff_array[i] = array[i+1]-array[i]
    return diff_array

'''Creates a matrix(table) containg the change in speed over the time for the x and y pos'''
def create_vol_array(array, diff_array, dt):
    t_array = np.multiply(diff_array,dt)
    array = np.divide(array, t_array)    
    return array

'''Fills the speed map'''
def create_vectorfield(array_x,array_y,vol_array,vectorfield):
    for i in range(vol_array.shape[0]):
        vectorfield[array_x[i]][array_y[i]]=vectorfield[array_x[i]][array_y[i]] + vol_array[i]
    return vectorfield

'''Filles the tensor map. 2D = MSD/t -> factor 2 accounted when maps are generated'''
def create_tensorfield(array_x,array_y,diff_array_x,diff_array_y,diff_array_t,tensorfield,dt):
    for i in range(diff_array_x.shape[0]):
        diff_array_dt = np.multiply(diff_array_t[i],dt)
        tensorfield[array_x[i]][array_y[i]]=tensorfield[array_x[i]][array_y[i]] + (diff_array_x[i]*diff_array_y[i]) / diff_array_dt
    return tensorfield

'''Filles the tensor map and corrects for the stat and dyn localization error. 2D = MSD/t -> factor 2 accounted when maps are generated '''
def create_tensorfield_errcorr(array_x,array_y,diff_array_x,diff_array_y, diff_array_t,tensorfield, dt, t_exp, sigma):
    for i in range(diff_array_x.shape[0]):
        
        diff_array_dt = np.multiply(diff_array_t[i],dt)       
             
        sdp = diff_array_x[i]*diff_array_y[i] # square displacement
        if sdp < 0:
            a = -1
        else:
            a = 1
        sdp = np.multiply(sdp,a)            
        sdp = sdp - (2 * sigma**2)
        sdp =  sdp / (diff_array_dt - ((1/3)*(t_exp/diff_array_dt)*t_exp))
        if sdp <= 0: # if correction yields negativ result --> no diff present
            sdp = 0
        else:
            sdp = np.multiply(sdp,a)
        tensorfield[array_x[i]][array_y[i]]=tensorfield[array_x[i]][array_y[i]] + sdp
    return tensorfield



"""Changes are calculated by firstly determining the differences of pos to pos and then trc for trc. The vectorfields 
(directed motion) are generated from the velocity arrays and tensorfields (diffusion) are generated from diffarrays in 1D"""
def calc_fields(a,time_lag,sigma,time_exp,path,resultpath,N,errcorr,pointfield,
                tracs_unint16corr,trac_numcorr,vectorfield_x,vectorfield_y,
                tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy):

    
    '''makes result folder'''
    if not os.path.exists(resultpath+'/Trc'+str(N)+'/'):
       #print('Path does not exist')
       os.mkdir(resultpath+'/Trc'+str(N))
       directory = resultpath+'/Trc'+str(N)
#       os.mkdir(directory+'/diffarray')
#       os.mkdir(directory+'/volarray')
#       os.mkdir(directory+'/trajec')
    else:
       directory = resultpath+'/Trc'+str(N)
    
    
    pointfield = create_pointfield(pointfield,tracs_unint16corr) #generation of the cound field

    x_array = np.array([],'float')
    y_array = np.array([],'float')
    t_array = np.array([],'float')
    
    k=0
    
    for i in range(tracs_unint16corr.shape[0]): 
        
        #concatenates x or y pos belonging to one specific trc until the last pos
        if tracs_unint16corr[0:a,0][i] == trac_numcorr[k] and i < tracs_unint16corr.shape[0]-1: 
            x_array = np.concatenate((x_array,np.array([tracs_unint16corr[0:a,2][i]])))
            y_array = np.concatenate((y_array,np.array([tracs_unint16corr[0:a,3][i]])))
            t_array = np.concatenate((t_array,np.array([tracs_unint16corr[0:a,1][i]])))
        
        #concatenates the last x or y pos and saves each trajectory respectively, generates for each trajectory the speed map and tensor map
        elif tracs_unint16corr[0:a,0][i] == trac_numcorr[k] and i == tracs_unint16corr.shape[0]-1: 
            x_array = np.concatenate((x_array,np.array([tracs_unint16corr[0:a,2][i]])))
            y_array = np.concatenate((y_array,np.array([tracs_unint16corr[0:a,3][i]])))
            t_array = np.concatenate((t_array,np.array([tracs_unint16corr[0:a,3][i]])))
            
            q = 'trajec_x_'+str(k)+'.csv' 
            w = 'trajec_y_'+str(k)+'.csv'
#            np.savetxt(directory + '/trajec/' + q, x_array, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/trajec/' + w, y_array, fmt='%d', delimiter=',',header='Value',comments='')
            
            x_array_uint16 = np.uint16(x_array)
            y_array_uint16 = np.uint16(y_array)
            t_array_uint16 = np.uint16(t_array)
        
        
            diff_array_x = create_diff_array(x_array)
            diff_array_y = -create_diff_array(y_array)
            diff_array_t = create_diff_array(t_array)
            
#            np.savetxt(directory + '/diffarray/diff_' + q, diff_array_x, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/diffarray/diff_' + w, diff_array_y, fmt='%d', delimiter=',',header='Value',comments='')        
            
            
            vol_array_x = create_vol_array(diff_array_x,diff_array_t,time_lag)
            vol_array_y = create_vol_array(diff_array_y,diff_array_t,time_lag)
            
#            np.savetxt(directory + '/volarray/vol_' + q, vol_array_x, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/volarray/vol_' + w, vol_array_y, fmt='%d', delimiter=',',header='Value',comments='')
    
            
            vectorfield_x = create_vectorfield(x_array_uint16,y_array_uint16,vol_array_x, vectorfield_x)
            vectorfield_y = create_vectorfield(x_array_uint16,y_array_uint16,vol_array_y, vectorfield_y)
        
            if errcorr == 'Yes' or errcorr == 'yes':
                tensorfield_xx = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_x,diff_array_x,diff_array_t,tensorfield_xx,time_lag,time_exp,sigma)
                tensorfield_xy = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_x,diff_array_y,diff_array_t,tensorfield_xy,time_lag,time_exp,sigma)
                tensorfield_yx = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_y,diff_array_x,diff_array_t,tensorfield_yx,time_lag,time_exp,sigma)
                tensorfield_yy = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_y,diff_array_y,diff_array_t,tensorfield_yy,time_lag,time_exp,sigma)
            else:                   
                tensorfield_xx = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_x,diff_array_x,diff_array_t,tensorfield_xx,time_lag)
                tensorfield_xy = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_x,diff_array_y,diff_array_t,tensorfield_xy,time_lag)
                tensorfield_yx = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_y,diff_array_x,diff_array_t,tensorfield_yx,time_lag)
                tensorfield_yy = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_y,diff_array_y,diff_array_t,tensorfield_yy,time_lag)
            
            
        else:
            q = 'trajec_x_'+str(k)+'.csv' 
            w = 'trajec_y_'+str(k)+'.csv'
#            np.savetxt(directory + '/trajec/' + q, x_array, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/trajec/' + w, y_array, fmt='%d', delimiter=',',header='Value',comments='')
            
            x_array_uint16 = np.uint16(x_array)
            y_array_uint16 = np.uint16(y_array)
            t_array_uint16 = np.uint16(t_array)
        
        
            diff_array_x = create_diff_array(x_array)
            diff_array_y = -create_diff_array(y_array)
            diff_array_t = create_diff_array(t_array)
    
#            np.savetxt(directory + '/diffarray/diff_' + q, diff_array_x, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/diffarray/diff_' + w, diff_array_y, fmt='%d', delimiter=',',header='Value',comments='')        
            
            
            vol_array_x = create_vol_array(diff_array_x,diff_array_t,time_lag)
            vol_array_y = create_vol_array(diff_array_y,diff_array_t,time_lag)
            
            
#            np.savetxt(directory + '/volarray/vol_' + q, vol_array_x, fmt='%d', delimiter=',',header='Value',comments='')
#            np.savetxt(directory + '/volarray/vol_' + w, vol_array_y, fmt='%d', delimiter=',',header='Value',comments='')
            
            
            vectorfield_x = create_vectorfield(x_array_uint16,y_array_uint16,vol_array_x, vectorfield_x)
            vectorfield_y = create_vectorfield(x_array_uint16,y_array_uint16,vol_array_y, vectorfield_y)
        
        
            if errcorr == 'Yes' or errcorr == 'yes':
                tensorfield_xx = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_x,diff_array_x,diff_array_t,tensorfield_xx,time_lag,time_exp,sigma)
                tensorfield_xy = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_x,diff_array_y,diff_array_t,tensorfield_xy,time_lag,time_exp,sigma)
                tensorfield_yx = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_y,diff_array_x,diff_array_t,tensorfield_yx,time_lag,time_exp,sigma)
                tensorfield_yy = create_tensorfield_errcorr(x_array_uint16,y_array_uint16,diff_array_y,diff_array_y,diff_array_t,tensorfield_yy,time_lag,time_exp,sigma)
            else:                   
                tensorfield_xx = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_x,diff_array_x,diff_array_t,tensorfield_xx,time_lag)
                tensorfield_xy = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_x,diff_array_y,diff_array_t,tensorfield_xy,time_lag)
                tensorfield_yx = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_y,diff_array_x,diff_array_t,tensorfield_yx,time_lag)
                tensorfield_yy = create_tensorfield(x_array_uint16,y_array_uint16,diff_array_y,diff_array_y,diff_array_t,tensorfield_yy,time_lag)
                
            k=k+1
            
            x_array = np.array([],'float')
            y_array = np.array([],'float')
            t_array = np.array([],'float')
            x_array = np.concatenate((x_array,np.array([tracs_unint16corr[0:a,2][i]])))
            y_array = np.concatenate((y_array,np.array([tracs_unint16corr[0:a,3][i]])))
            t_array = np.concatenate((t_array,np.array([tracs_unint16corr[0:a,1][i]])))
    
    
    tif.imsave(directory+'/pointfield.tif',pointfield)            
    tif.imsave(directory+'/vectorfield_x.tif',vectorfield_x)
    tif.imsave(directory+'/vectorfield_y.tif',vectorfield_y)
    
    tif.imsave(directory+'/tensorfield_xx.tif',tensorfield_xx)
    tif.imsave(directory+'/tensorfield_xy.tif',tensorfield_xy)
    tif.imsave(directory+'/tensorfield_yx.tif',tensorfield_yx)
    tif.imsave(directory+'/tensorfield_yy.tif',tensorfield_yy)
            
    return directory,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy,vectorfield_x,vectorfield_y

if __name__ == "__main__":
#    time = 10
#    """ms"""
#    N = 1
##    path = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/TrcData/trc1'
#    resultpath = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Test'
    directory,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy,vectorfield_x,vectorfield_y\
    =calc_fields(a,time_lag,sigma,time_exp,path,resultpath,N,errcorr,pointfield,
                 tracs_unint16corr,trac_numcorr,vectorfield_x,vectorfield_y,
                 tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy)
   