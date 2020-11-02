#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:53:35 2020

@author: marieschwebs
"""

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import EllipseCollection
from copy import deepcopy
import os


'''Calculation of the eigenvalues and eigenvectors which are neccesary 
for the ellipsoide plots'''
def calc_eigenValVec(ten_xx, ten_xy,ten_yx, ten_yy):
    
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
    
    return widths, heights, heights_scal, widths_scal, angle



'''function to extract the fp region from the original arrays with with following rescaling of 
both axis for fp region plots'''
def extraction_fp_region(entr, outline, px_extent, widths_scal, heights_scal, angle, pix_max, ten_xx):
    
    #def of array, later on containing extremes of x and y values
    fp_ext = np.zeros((2,2))   
          
    #find ind to extract section from whole arrays
    fp_ext[0,0] = round(np.amin(entr[:,0])) - px_extent #always round down
    fp_ext[0,1] = round(np.amax(entr[:,0])) + px_extent#always round up
    fp_ext[1,0] = round(np.amin(entr[:,1])) - px_extent
    fp_ext[1,1] = round(np.amax(entr[:,1])) + px_extent
    
    #conversion to integer for later extraction
    fp_ext = fp_ext.astype(int)   
    
    outline_fp = deepcopy(outline)
    
#    #adjustment of the fp outline to the entrance section with sorting possibilty
    
#    outline_fp[outline_fp[:,0]< fp_ext[0,0]+1] = 0
#    outline_fp[outline_fp[:,0]> fp_ext[0,1]+1] = 0
#    outline_fp[outline_fp[:,1]< fp_ext[1,0]+1] = 0
#    outline_fp[outline_fp[:,1]> fp_ext[1,1]+1] = 0
#
#    #identification of zeros in the array
#    ind = np.where(~outline_fp.any(axis=1))[0]
#    start = ind[0]
#    end = ind[-1]
#    
#    #sort the coor to create logically connected line and
#    #removal of the zero entries to plot highlighted area
#    #with case distinction for the location of zeros within the array
#    if start != 0 or end != (outline_fp.shape[0]-1):
#        part1 = outline_fp[end+1:-1]
#        part2 = outline_fp[0:start-1]       
#        outline_fp = np.concatenate((part1,part2)) #default axis 0
#    else:
#        outline_fp = np.delete(outline_fp, np.where(~outline_fp.any(axis=1))[0], axis=0) #removes all zeros from array
#        
        
    #creation of deepcopies and correction of outline coor for section 
    #images. In section coor system is reseted to 0
    entr_fp = deepcopy(entr)
    entr_fp[:,0] = entr_fp[:,0] - fp_ext[0,0] 
    entr_fp[:,1] = entr_fp[:,1] - fp_ext[1,0]
    outline_fp[:,0] = outline_fp[:,0] - fp_ext[0,0] 
    outline_fp[:,1] = outline_fp[:,1] - fp_ext[1,0]
    
    
    #handling of exceptions:
    #space around fp outline smaller than to extracted px
    if fp_ext[0,0] < 0: #min x value
        entr_fp[:,0] = entr_fp[:,0] + fp_ext[0,0] # neg plus neg is pos
        outline_fp[:,0] = outline_fp[:,0] + fp_ext[0,0]
        fp_ext[0,0] = 0
    if fp_ext[1,0] < 0:#min y value
        entr_fp[:,1] = entr_fp[:,1] + fp_ext[1,0]
        outline_fp[:,1] = outline_fp[:,1] + fp_ext[1,0]
        fp_ext[1,0] = 0
    if fp_ext[0,1] > ten_xx.shape[1]-1: #max x value
        fp_ext[0,1] = ten_xx.shape[1]-1
    if fp_ext[1,1] > ten_xx.shape[0]-1: #max y value
         fp_ext[1,1] = ten_xx.shape[0]-1
               
    
    #extraction of the fp section, from a till b+1 because of python structure, excludes number b in extractionlimit b is excluded
    widths_scal_fp = widths_scal[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    heights_scal_fp = heights_scal[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    angle_fp = angle[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    pix_max_fp = pix_max[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
       
    return widths_scal_fp, heights_scal_fp, angle_fp, pix_max_fp, outline_fp, entr_fp


'''Generates elipsoid plot of the whole cell'''
def EllipsoidePlot_WholeCell(ten_xx, widths_scal, heights_scal, angle, \
                             cmap, pix_max, outline, entr, resultpath, i):
    if np.size(outline) == 2:#case distinction if outline is a tuple
        if np.size(outline[0]) == np.size(entr[0]):
            col = 'k'
        else:
            col = 'r'
    else:
        if np.size(outline) == np.size(entr):
            col = 'k'
        else:
            col = 'r'
    
    Y,X = np.mgrid[0:ten_xx.shape[0], 0:ten_xx.shape[1]]
    XY = np.column_stack((X.ravel(), Y.ravel()))
    
    #plotting
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0-1, ten_xx.shape[1])
    ax.set_ylim(ten_xx.shape[0], 0-1)
    ec = EllipseCollection(widths_scal, heights_scal, angle, units='x',
                           offsets=XY, transOffset=ax.transData, 
                           cmap=cmap, linewidths=0.05, edgecolor='black')
    ec.set_array(pix_max.ravel())
    ax.add_collection(ec)
    cbar = plt.colorbar(ec)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('µm$^2$/s', fontsize = 10)
    plt.xticks([])
    plt.yticks([])
    plt.hold
    if np.size(outline) == 2:
        plt.plot(outline[0][:,0],outline[0][:,1], color = 'k', linewidth=0.85)  
        plt.hold
        plt.plot(outline[1][:,0],outline[1][:,1], color = 'k', linewidth=0.85)  
        plt.hold
        plt.plot(entr[0][:,0],entr[0][:,1], color = col, linewidth=0.85) 
        plt.hold
        plt.plot(entr[1][:,0],entr[1][:,1], color = col, linewidth=0.85)  
    else:           
        plt.plot(outline[:,0],outline[:,1], color = 'k', linewidth=0.85)  
        plt.hold
        plt.plot(entr[:,0],entr[:,1], color = col, linewidth=0.85)  
    plt.savefig(resultpath+'EllipsoidePlot_Cell'+str(i), dpi=300)
    plt.close(fig)
    
    return



'''Generates ellipsoid plot of the area sorrounding fp'''
def EllipsoidePlot_Section(widths_scal_fp, heights_scal_fp, angle_fp, \
                             cmap, pix_max_fp, outline_fp, entr_fp, size_outline,\
                             resultpath, i, sec):
    
    #creation of a grid to place ellipsoids in plot
    Y_fp,X_fp = np.mgrid[0:widths_scal_fp.shape[0], 0:widths_scal_fp.shape[1]]
    XY_fp = np.column_stack((X_fp.ravel(), Y_fp.ravel()))
    
    #plot
    fig, ax = plt.subplots()
    ax.set_xlim(0-1, widths_scal_fp.shape[1])
    ax.set_ylim(widths_scal_fp.shape[0], 0-1)
    ax.set_aspect('equal')
    ec_fp = EllipseCollection(widths_scal_fp, heights_scal_fp, angle_fp, units='x',
                           offsets=XY_fp, transOffset=ax.transData, 
                           cmap=cmap, linewidths=0.05, edgecolor='black')
    ec_fp.set_array(pix_max_fp.ravel())
    ax.add_collection(ec_fp)
    cbar = plt.colorbar(ec_fp) 
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('µm$^2$/s', fontsize = 20)
    plt.xticks([])
    plt.yticks([])
    plt.hold
    plt.plot(outline_fp[:,0],outline_fp[:,1], color = 'k', linewidth = 4)  
    plt.hold
    plt.plot(entr_fp[:,0],entr_fp[:,1], color = 'r', linewidth = 4)
    if size_outline == 2:
        plt.savefig(resultpath+'Sec_EllipsoidePlot_Cell'+str(i)+'_Sec'+str(sec), dpi=300)
    else:
        plt.savefig(resultpath+'Sec_EllipsoidePlot_Cell'+str(i)+'_Sec', dpi=300)
    plt.close(fig)
    
    return



'''main function: generation of the ellipsoid plots of the diffusion coefficients '''
def EllipsoidePlotPlusOutline(path,filename,outline_path,resultpath,binning,px_extent,N,highlighting):
    
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
#            resultpath = resultpath+'/DiffusionMaps/'
    
    
    diff_info=np.zeros((N+2,3)) #Nx2 array: first columns maximum, second 
    #column average, two additional rows to seperate the calculated mean of 
    #each column from the single cell data
    diff_info_sec=np.zeros((1,4))
    
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
        
        pix_max=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        
        
        '''calc of the eigenvalues and eigenvectors needed for displaying 
        information as elipsoids'''        
        widths, heights, heights_scal, widths_scal, angle = \
        calc_eigenValVec(ten_xx, ten_xy,ten_yx, ten_yy)
        
        ellipticity=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        eccentricity=np.zeros((ten_xx.shape[0],ten_xx.shape[1]))
        
        '''Determination of the max value for colorcode and colorbar + ellipticity 
        + eccentricity'''
        for f in range(ten_xx.shape[0]): 
            for g in range(ten_xx.shape[1]):
                if widths[f,g] < heights[f,g]:
                    pix_max[f,g] = heights[f,g]
                    ellipticity[f,g] = np.divide(widths[f,g],heights[f,g],out=np.zeros_like(widths[f,g]), where=heights[f,g]!=0)
                    eccentricity[f,g] = np.sqrt(1 - np.divide(widths[f,g],heights[f,g],out=np.zeros_like(widths[f,g]), where=heights[f,g]!=0)**2)
                else:
                    pix_max[f,g] = widths[f,g]
                    ellipticity[f,g] = np.divide(heights[f,g],widths[f,g],out=np.zeros_like(heights[f,g]), where=widths[f,g]!=0)
                    eccentricity[f,g] = np.sqrt(1 - np.divide(heights[f,g],widths[f,g],out=np.zeros_like(heights[f,g]), where=widths[f,g]!=0)**2)

                    
                    
        
        np.save(resultpath + '/Info/Angle/angle_cell_' + str(i) + '.npy', angle)
        np.save(resultpath + '/Info/DiffCoeffPx/diffcoeff_px_cell_' + str(i) + '.npy', pix_max)
        np.save(resultpath + '/Info/Ellipticity/ellipticity_cell_' + str(i) + '.npy', ellipticity)
        

        '''Load points to draw outline in map'''
        outline = np.load(outline_path + '/coor_outline_scal_' + str(i) + '.npy')
        if highlighting == 'Yes': #Case distinction whether fp region should be highlighted
            entr = np.load(outline_path + '/entr_highl_cell' + str(i) + '.npy')
        size_outline = np.size(outline)
        
        #case distinction whether two fp are present
        if size_outline == 2:
            #determination of colormap
            cmap = mpl.cm.cool
            
            #conversion into a readable tuple
            outline = tuple(outline)
            
            #Case distinction whether fp region should be highlighted
            if highlighting == 'Yes':
                entr = tuple(entr)
                
                #extraction of the region surrounding the first fp
                widths_scal_fp1, heights_scal_fp1, angle_fp1, pix_max_fp1, outline_fp1, entr_fp1 = \
                extraction_fp_region(entr[0], outline[0], px_extent, widths_scal, heights_scal, angle, pix_max, ten_xx)
                
                #extraction of the region surrounding the second fp
                widths_scal_fp2, heights_scal_fp2, angle_fp2, pix_max_fp2, outline_fp2, entr_fp2 = \
                extraction_fp_region(entr[1], outline[1], px_extent, widths_scal, heights_scal, angle, pix_max, ten_xx)
                
                '''Generation of ellipsoid plot whole cell'''
                EllipsoidePlot_WholeCell(ten_xx, widths_scal, heights_scal, angle, \
                                         cmap, pix_max, outline, entr, resultpath, i)
                
                sec = 1 #serves as input parameter for saving the first section plot
                
                '''Generation of ellipsoid plot of section surrounding fp entrance'''
                EllipsoidePlot_Section(widths_scal_fp1, heights_scal_fp1, angle_fp1, \
                                       cmap, pix_max_fp1, outline_fp1, entr_fp1, size_outline,\
                                       resultpath, i, sec)
    
                sec = 2 #serves as input parameter for saving the second section plot
    
                '''Generation of ellipsoid plot of section surrounding fp entrance'''
                EllipsoidePlot_Section(widths_scal_fp2, heights_scal_fp2, angle_fp2, \
                                       cmap, pix_max_fp2, outline_fp2, entr_fp2, size_outline,\
                                       resultpath, i, sec)
                
                #collecting infos about max value in diff coef and average val of each cell
                sec_info = np.zeros((2,4))
                #info fp1
                sec_info[0,0] = i
                sec_info[0,1] = np.nanmax(pix_max_fp1)
                pix_max_fp1[pix_max_fp1 == 0] = np.nan
                sec_info[0,2] = np.nanmean(pix_max_fp1)
                sec_info[0,3] = np.nanmedian(pix_max_fp1)
                #info fp2
                sec_info[1,0] = i
                sec_info[1,1] = np.nanmax(pix_max_fp2)
                pix_max_fp2[pix_max_fp2 == 0] = np.nan
                sec_info[1,2] = np.nanmean(pix_max_fp2)
                sec_info[1,3] = np.nanmedian(pix_max_fp2)
                
                diff_info_sec=np.append(diff_info_sec, sec_info, axis=0)
                
            else:
                '''Generation of ellipsoid plot whole cell'''
                EllipsoidePlot_WholeCell(ten_xx, widths_scal, heights_scal, angle, \
                                         cmap, pix_max, outline, outline, resultpath, i)
            
            
        #case distinction whether only one fp is present
        else:      
            #determination of colormap
            cmap = mpl.cm.cool
            
            #Case distinction whether fp region should be highlighted
            if highlighting == 'Yes':
                #extraction of the region surrounding the fp
                widths_scal_fp, heights_scal_fp, angle_fp, pix_max_fp, outline_fp, entr_fp = \
                extraction_fp_region(entr, outline, px_extent, widths_scal, heights_scal, angle, pix_max, ten_xx)
                
                '''Generation of ellipsoid plot whole cell'''
                EllipsoidePlot_WholeCell(ten_xx, widths_scal, heights_scal, angle, \
                                         cmap, pix_max, outline, entr, resultpath, i)
                
                sec = 1 #serves as input parameter for saving the section plot
              
                '''Generation of ellipsoid plot of section surrounding fp entrance'''
                EllipsoidePlot_Section(widths_scal_fp, heights_scal_fp, angle_fp, \
                                       cmap, pix_max_fp, outline_fp, entr_fp, size_outline, \
                                       resultpath, i, sec)
                
                
                #collecting infos about max value in diff coef and average val of each cell
                sec_info = np.zeros((1,4))
                sec_info[0,0] = i
                sec_info[0,1] = np.nanmax(pix_max_fp)
                pix_max_fp[pix_max_fp == 0] = np.nan
                sec_info[0,2] = np.nanmean(pix_max_fp)
                sec_info[0,3] = np.nanmedian(pix_max_fp)        
                
                diff_info_sec=np.append(diff_info_sec, sec_info, axis=0)
                
            else:
                '''Generation of ellipsoid plot whole cell'''
                EllipsoidePlot_WholeCell(ten_xx, widths_scal, heights_scal, angle, \
                                         cmap, pix_max, outline, outline, resultpath, i)   
            
            
            
            
        #collecting infos about max value in diff coef and average val of each cell
        maximum = np.nanmax(pix_max)
        pix_max[pix_max == 0] = np.nan
        average = np.nanmean(pix_max)
        median = np.nanmedian(pix_max)
        
        diff_info[i-1,:]=maximum, average, median
        
    #averaging of the whole set of N cells
    if N > 1: #does not make sense to calc mean over just one value
        diff_info[-1,0]=np.nanmean(diff_info[0:-2,0])
        diff_info[-1,1]=np.nanmean(diff_info[0:-2,1])
        diff_info[-1,2]=np.nanmean(diff_info[0:-2,2])
    
    np.save(resultpath+'Diff_info_all_bin'+str(binning),diff_info)
    np.save(resultpath+'Diff_info_all_sec'+str(binning),diff_info_sec)
    
if __name__ == "__main__":
    N = 2 #number to be evaluated cells
    binning = 160 #binning to x nm
    px_extent = 4  #def of number of px surrounding fp outline
    resultpath = '/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Maps_plus_Outline_corr'
    path='/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Trc'
    filename = '/results_filtering/'
    outline_path = '/Volumes/Reindeer/TrackingVSGAtto/MORN/CoorOutlineCorrected'
    highlighting = 'Yes'
    EllipsoidePlotPlusOutline(path,filename,outline_path,resultpath,binning,px_extent,N,highlighting)

