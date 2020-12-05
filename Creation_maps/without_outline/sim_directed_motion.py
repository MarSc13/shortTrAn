#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:39:15 2019

@author: mas32ea
"""

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib import cm
from math import atan2,degrees
import os


def directed_motion(path,filename,resultpath,binning,N):
    
    #Generation of the resultfolder
    if not os.path.exists(resultpath+'/DirectedMotionMaps'):
            os.mkdir(resultpath+'/DirectedMotionMaps')
            resultpath = resultpath+'/DirectedMotionMaps/'
    else:
            resultpath = resultpath+'/DirectedMotionMaps/'
    
    speed_info=np.zeros((N+2,3)) #Nx2 array: first columns maximum, 
    #second column average, two additional rows to seperate the calculated 
    #mean of each column from the single cell data
    
    
    for i in range(1,N+1):
        '''Loads the bright field image'''
    #    im=tif.imread('/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/BFimages/BF2/img_000000000_Default_000.tif')
    #    plt.imshow(im, cmap='Greys')
        
        directory=path+str(i)+filename
    
        '''Loads the x and y components to calcualte the speed of the directed 
        motion within an pixel'''
        img_x = tif.imread(directory+'scld_vec_x_fil.tif')
        img_y = tif.imread(directory+'scld_vec_y_fil.tif')
        
        img_l = np.sqrt(img_x**2 + img_y**2)
        angle=np.zeros((img_x.shape[0],img_x.shape[1]))
        angle_deg=np.zeros((img_x.shape[0],img_x.shape[1]))

        
        for a in range(img_x.shape[0]): #ranges through rows
            for b in range(img_x.shape[1]): #ranges through columns
                angle_degpar = degrees(atan2(img_y[a,b],img_x[a,b]))
                angle_deg[a,b]=angle_degpar
                
                if angle_degpar <0:
                    angle_degpar=360+angle_degpar
                angle_deg[a,b]=angle_degpar
                
                if angle_degpar <=45 and angle_degpar>0 or angle_degpar>315:
                    angle[a,b]=1
                elif angle_degpar>45 and angle_degpar<=135:
                    angle[a,b]=2
                elif angle_degpar>135 and angle_degpar<=225:
                    angle[a,b]=3
                elif angle_degpar>225 and angle_degpar<=315:
                    angle[a,b]=4
                    
        np.save(resultpath+'AngleSpeed_Cell'+str(i)+'.npy', angle_deg)            
        
        '''Generation for the grid of the whole picture. 0/0 is in the upper 
        left corner'''
        Y,X = np.mgrid[0:img_x.shape[0], 0:img_y.shape[1]]
        
        
        '''Genration of the black white colormap used by the quiver plot'''
        binary = cm.get_cmap('binary', 256)
        wb = binary(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        wb[15:256, :] = black
        wbmap = ListedColormap(wb)
        
        fourcmap = colors.ListedColormap(['white', 'darkorange', 'b', 'darkcyan','m'])
        bounds=[0,1,2,3,4,5]
        norm = colors.BoundaryNorm(bounds, fourcmap.N)
        
        
        '''To display the correct orientation of incomming speed - X, Y, 
        img_x, img_y - have to  be placed in exactly this order wihtin the 
        quiver function. This was checked on 24.09.18 MS'''
        fig1 = plt.figure()
        plt.quiver(X, Y, img_x, img_y, angle, headwidth=5, linewidth=1.2, 
                   minshaft=2.0, scale_units='height', scale=250, pivot='mid',
                   alpha=.5, cmap=fourcmap) 
#        plt.quiver(X, Y, img_x, img_y, img_l, headwidth=5, linewidth=1.0, 
#                   minshaft=2.0, scale_units='height', scale=250, pivot='mid',
#                   alpha=.5, cmap=wbmap) 
#        plt.axis('equal')
        plt.xlim(0-1, img_x.shape[1]+1)
        plt.ylim(img_y.shape[0]+1, 0-1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_l, cmap='binary', alpha=.01)
        tif.imsave(resultpath+'Speed_Cell'+str(i), img_l)
        plt.savefig(resultpath+'DirectionSpeed_Cell'+str(i), dpi=300)   
        
        '''Generation of an speed heat map with the final unit of mum/s'''       
        fig2 = plt.figure()
#        plt.axis('equal')
        plt.imshow(img_l,cmap='RdPu')
        plt.xticks([])
        plt.yticks([])
#        plt.show()
        cbar=plt.colorbar()
        cbar.set_label('µm/s')
        plt.savefig(resultpath+'HeatMap_Cell'+str(i), dpi=300)
        
        
        maximum = img_l.max()
        img_l[img_l == 0] = np.nan
        average = np.nanmean(img_l)
        median = np.nanmedian(img_l)
        
        speed_info[i-1,:]=maximum, average, median
   
    if N > 1: #does not make sense to clc mean over just one value
        speed_info[-1,0] = np.mean(speed_info[0:-2,0])
        speed_info[-1,1] = np.mean(speed_info[0:-2,1])
        speed_info[-1,2] = np.mean(speed_info[0:-2,2])
    
    np.save(resultpath+'Speed_info_all_bin'+str(binning),speed_info)

if __name__ == "__main__":
    N = 20 #number to be evaluated cells
    binning = 160 #binning to x nm
    resultpath = '/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Analysis_maps/Maps_160'
    path='/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Results_Par_testing/Trc'
    filename = '/Binning_'+str(binning)+\
    '/results_filtering/'
    #directory = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Trc/Trc_thresh2/Trc'+str(N)+'/'
    directed_motion(path,filename,resultpath,binning,N)