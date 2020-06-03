#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:50:19 2019

@author: mas32ea
"""

import numpy as np 
"""creates space for latter downscalling and defines the shape"""

def padding(tracs_unint16corr, a, resultpath, N):
    x_min = np.amin(tracs_unint16corr[0:a,2])
    x_max = np.amax(tracs_unint16corr[0:a,2])
    y_min = np.amin(tracs_unint16corr[0:a,3])
    y_max = np.amax(tracs_unint16corr[0:a,3]) #defining the latter image section --> to reduce data size
    
    a = tracs_unint16corr.shape[0] #number of entries = number of rows
    
    tracs_unint16corr[0:a,2] = tracs_unint16corr[0:a,2] - x_min + 160 #generation of additional space, generation of 160 x 1nm empty pix in x 
    tracs_unint16corr[0:a,3] = tracs_unint16corr[0:a,3] - y_min + 160 #generation of additional space, generation of 160 x 1nm empty pix in y 
    
    x_length = x_max - x_min + 320 #addtion of 160x1nm pix at the end in x direction
    y_length = y_max - y_min + 320 #addtion of 160x1nm pix at the end in y direction
    
    shape = (x_length,y_length) #used to define the padded arrays
    
    ''''commented because of checking the possibilty of start and end pix in binning.npy'''
    par_im_sec=np.zeros((2,2)) #first line shall x values, second line shall be y values
    #used for the positioning of the flagellar pocket 
    par_im_sec[0,0] = x_min
    par_im_sec[0,1] = x_max
    par_im_sec[1,0] = y_min
    par_im_sec[1,1] = y_max
    
    np.save(resultpath+'/padding_par_cell'+str(N),par_im_sec)
    
    return  x_min, x_max, y_min, y_max, a, tracs_unint16corr, x_length, y_length, shape

if __name__ == "__main__":
    resultpath = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Results_TrcAnalysis'
    N = 1
    x_min, x_max, y_min, y_max, a, tracs_unint16corr, x_length, y_length, shape = padding(tracs_unint16corr, a, resultpath, N)