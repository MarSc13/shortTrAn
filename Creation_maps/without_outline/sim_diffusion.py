#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:45:14 2019

@author: mas32ea
"""

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import EllipseCollection
from math import degrees
import os



def EllipsoidePlot(path,filename,resultpath,binning,N):
    
    #Generation of the resultfolder
    if not os.path.exists(resultpath+'/DiffusionMaps'):
            os.mkdir(resultpath+'/DiffusionMaps')
            resultpath = resultpath+'/DiffusionMaps/'
    else:
            resultpath = resultpath+'/DiffusionMaps/'
    
    #Generation of folder for informations used for VSG diffusion simulation        
    if not os.path.exists(resultpath+'/Info'):
            os.mkdir(resultpath+'/Info')
            os.mkdir(resultpath+'/Info/Angle')
            os.mkdir(resultpath+'/Info/Ellipticity')
            os.mkdir(resultpath+'/Info/DiffCoeffPx')
    
    diff_info=np.zeros((N+2,3)) #Nx2 array: first columns maximum, second 
    #column average, two additional rows to seperate the calculated mean of 
    #each column from the single cell data
        
    for i in range(1,N+1):
        directory = path+str(i)+filename
        
        ten_xx = tif.imread(directory +'scld_ten_xx_fil.tif')
        ten_xy = tif.imread(directory +'scld_ten_xy_fil.tif')
        ten_yx = tif.imread(directory +'scld_ten_yx_fil.tif')
        ten_yy = tif.imread(directory +'scld_ten_yy_fil.tif')
        
        #factor 1000 to convert nm2/ms into µm2/s and 
        #factor 2 to consider 2*D
        ten_xx = ten_xx/2000 
        ten_xy = ten_xy/2000 
        ten_yx = ten_yx/2000
        ten_yy = ten_yy/2000
        
        
        '''generates arrays needed for ellipsoid plot'''
        #any array could be used to determine the shape, because every array 
        #has the same shape
        matrix=np.zeros((2,2))
        eigenvalue=np.zeros((1,2))
        eigenvec=np.zeros((2,2))
        widths=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        widths_scal=np.zeros((ten_xx.shape[0],ten_xx.shape[1])) 
        heights=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        heights_scal=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        angle=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        pix_max=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        pix_min=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        ellipticity=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        eccentricity=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        
        Y,X = np.mgrid[0:ten_xx.shape[0], 0:ten_xx.shape[1]]
        XY = np.column_stack((X.ravel(), Y.ravel()))
        
        '''calculation of the eigenvalues and eigenvectors for each cell/pixel'''
        for a in range(ten_xx.shape[0]): #any array could be used to determine 
            #the shape, because every array has the same shape, ranges through 
            #rows
            for b in range(ten_xx.shape[1]): #ranges through columns
                matrix[0,0]=ten_xx[a,b] #compilation of all 4 cohesive  
                matrix[0,1]=ten_xy[a,b] #tensor directions
                matrix[1,0]=ten_yx[a,b]
                matrix[1,1]=ten_yy[a,b]
                eigenvalue,eigenvec=np.linalg.eig(matrix) #calculation of the 
                            #eigenvalue and the eigenvectors for each position
                widths[a,b]=eigenvalue[0]
                heights[a,b]=eigenvalue[1]
                '''Normalization of ellipse length to approx 1'''
                if widths[a,b] == 0 or heights[a,b] == 0:
                    widths_scal[a,b] = 0
                    heights[a,b] = 0
                elif widths[a,b] > heights[a,b]: #use the larger value to normalize
                    widths_scal[a,b]=widths[a,b]/widths[a,b] 
                    heights_scal[a,b]=heights[a,b]/widths[a,b] 
                else:
                    widths_scal[a,b]=widths[a,b]/heights[a,b]
                    heights_scal[a,b]=heights[a,b]/heights[a,b]
                    
                """ If eigv and eig or rotmat and rotmateig are not equal
                check if angles contains negative angle values 
                if this is the case evaluate rotmateig with -eigenvec"""
                angles = np.array([[np.rad2deg(np.arccos(eigenvec[0,0])),-np.rad2deg(np.arcsin(eigenvec[0,1]))],
                       [np.rad2deg(np.arcsin(eigenvec[1,0])),np.rad2deg(np.arccos(eigenvec[1,1]))]])
                
                if angles[0,1] < 0 or angles[1,0] < 0:
                    eigenvec = -eigenvec
                
                angles = np.array([[np.rad2deg(np.arccos(eigenvec[0,0])),-np.rad2deg(np.arcsin(eigenvec[0,1]))],
                       [np.rad2deg(np.arcsin(eigenvec[1,0])),np.rad2deg(np.arccos(eigenvec[1,1]))]])
                angle[a,b] = angles[0,0]
        
        
        
        
        '''Determination of the max value for colorcode and colorbar + determination ellipticity'''
        for f in range(ten_xx.shape[0]): 
            for g in range(ten_xx.shape[1]):
                if widths[f,g] < heights[f,g]:
                    pix_max[f,g] = heights[f,g]
                    pix_min[f,g] = widths[f,g]
                    ellipticity[f,g] = np.divide(widths[f,g],heights[f,g],out=np.zeros_like(widths[f,g]), where=heights[f,g]!=0)
                    eccentricity[f,g] = np.sqrt(1 - np.divide(widths[f,g],heights[f,g],out=np.zeros_like(widths[f,g]), where=heights[f,g]!=0)**2)
                else:
                    pix_max[f,g] = widths[f,g]
                    pix_min[f,g] = heights[f,g]
                    ellipticity[f,g] = np.divide(heights[f,g],widths[f,g],out=np.zeros_like(heights[f,g]), where=widths[f,g]!=0)
                    eccentricity[f,g] = np.sqrt(1 - np.divide(heights[f,g],widths[f,g],out=np.zeros_like(heights[f,g]), where=widths[f,g]!=0)**2)

        
        tif.imsave(resultpath + '/Info/Angle/angle_cell_' + str(i) + '.tif', angle)
        tif.imsave(resultpath + '/Info/DiffCoeffPx/diff_maj_px_cell_' + str(i) + '.tif', pix_max)
        tif.imsave(resultpath + '/Info/DiffCoeffPx/diff_min_px_cell_' + str(i) + '.tif', pix_min)
        tif.imsave(resultpath + '/Info/Ellipticity/ell_cell_' + str(i) + '.tif', ellipticity)
        tif.imsave(resultpath + '/Info/Ellipticity/ecc_cell_' + str(i) + '.tif', eccentricity)
        
        '''Following lines were inserted to count the batches more easily'''
#        widths_scal[pix_max >= 0.45] = 0
#        heights_scal[pix_max >= 0.45] = 0
#        angle[pix_max >= 0.45] = 0
#        pix_max[pix_max >= 0.45] = 0
        
        cmap = mpl.cm.cool
        
        fig, ax = plt.subplots()
        ax.set_xlim(0-1, ten_xx.shape[1]+1)
        ax.set_ylim(ten_xx.shape[0]+1, 0-1)
        ax.set_aspect('equal')
        ec = EllipseCollection(widths_scal, heights_scal, angle, units='x',
                               offsets=XY, transOffset=ax.transData, 
                               cmap=cmap, linewidths=0.05, edgecolor='black')
        ec.set_array(pix_max.ravel())
        ax.add_collection(ec)
        cbar = plt.colorbar(ec)
        cbar.set_label('µm$^2$/s')
        plt.xticks([])
        plt.yticks([])
#        fig.text(0.5, 0.95, 'Cell'+str(i), ha='center', va='center',fontsize=12,
#                 fontweight='bold')
#        fig.show() #cause error message: non GUI backend
        plt.savefig(resultpath+'EllipsoidePlot_Cell'+str(i), dpi=300)
        plt.close(fig)
        
        
        maximum = pix_max.max()
        pix_max[pix_max == 0] = np.nan
        average = np.nanmean(pix_max)
        median = np.nanmedian(pix_max)
        
        diff_info[i-1,:]=maximum, average, median
        
    
    if N > 1: #does not make sense to clc mean over just one value
        diff_info[-1,0]=np.mean(diff_info[0:-2,0])
        diff_info[-1,1]=np.mean(diff_info[0:-2,1])
        diff_info[-1,2]=np.mean(diff_info[0:-2,2])
    
    tif.imsave(resultpath+'/Info/Diff_info_all_bin'+str(binning)+'.tif',diff_info)
 
    
if __name__ == "__main__":
    N = 20 #number to be evaluated cells
    binning = 160 #binning to x nm
    resultpath = '/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Analysis/191113/Gauss'
    path='/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Results_Par_testing/Trc'
    filename = '/Binning_'+str(binning)+\
    '/results_filtering/'
    EllipsoidePlot(path,filename,resultpath,binning,N)

