#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:31:09 2020

@author: marieschwebs
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



'''def of function extraction_highlighted_area()'''
def extraction_highlighted_area(coor, fp_coords):
    
    #query the maxima for imdim to create a mask
    width = int(np.amax(coor[:,0])+1)
    height = int(np.amax(coor[:,1])+1)
    
    #creation of a binary mask, px within the polygon line get the value 1
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(fp_coords, outline=1, fill=1)
    mask = np.array(img)
    
    #definition of arrays
    pxcoor = np.zeros((1,2))      
    highl_coor = np.zeros(coor.shape)
    
    #query of the coor if they lie within the polygon
    #coor outside are replaced with the value 0
    for j in range(0,coor.shape[0]):
        pxcoor = coor[j,:]
        pxcoor = pxcoor.astype(int)
        
        if mask[pxcoor[1],pxcoor[0]] == 1:
           highl_coor[j,:] = coor[j,:]
           
    #identification of the zeros within the array
    ind = np.where(~highl_coor.any(axis=1))[0]
    start = ind[0]
    end = ind[-1]
    
    #sort the coor to create logically connected line and
    #removal of the zero entries to plot highlighted area
    #with case distinction for the location of zeros within the array
    if start != 0 or end != (highl_coor.shape[0]-1):
        part1 = highl_coor[end+1:-1]
        part2 = highl_coor[0:start-1]       
        highl_coor_sort = np.concatenate((part1,part2)) #default axis 0
    else:
        highl_coor_sort = np.delete(highl_coor, np.where(~highl_coor.any(axis=1))[0], axis=0) #removes all zeros from array
    
    return highl_coor_sort



'''def of main function to iterate over a set of data and to make a case distinction 
of cells with only one flagellar pocket or two'''
def highlighting_fp(path, resultpath, binning, N):
    
    for i in range(1, N+1):
        
        #loads outline coordinates
        coor = np.load(path + '/coor_outline_raw_' + str(i) + '.npy')
        
        #parameter used for case distinction, one fp or two fp
        size_input = np.size(coor)
        
        fig = plt.figure()
        fig.canvas.manager.window.tkraise()
        plt.close(fig)
        
        #case distinction if one or  two fp are present
        if size_input == 2: #two fp are present
            #converts into readable tuple
            coor = tuple(coor)
                       
            #selection of the first area to be marked by clicking to create a polygon
            fig1 = plt.figure()          
#            raise_window('fig1')
            fig1.canvas.manager.window.tkraise()
            plt.plot(coor[0][:,0], coor[0][:,1],'b')
            plt.axis('scaled')
            plt.xlim(np.amin(coor[0][:,0])-100, np.amax(coor[0][:,0])+100)
            plt.ylim(np.amax(coor[0][:,1])+100, np.amin(coor[0][:,1])-100)
            plt.title('Define polygonial area for highlighted fp entrance. \n left click: addition of point, right click: when finished with selection')
            plt.pause(0.1)
            fp_coords1 = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3) # extraction of coordinates
            plt.close(fig1)
            
            #selection of the second area to be marked by clicking to create a polygon
            fig2 = plt.figure()
            plt.plot(coor[1][:,0], coor[1][:,1],'b')
            plt.xlim(np.amin(coor[1][:,0])-100, np.amax(coor[1][:,0])+100)
            plt.ylim(np.amax(coor[1][:,1])+100, np.amin(coor[1][:,1])-100)
            plt.title('Define polygonial area for highlighted fp entrance. \n left click: addition of point, right click: when finished with selection')
            fig2.canvas.manager.window.tkraise()
            plt.pause(0.1)
            fp_coords2 = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3) # extraction of coordinates
            plt.close(fig2)
            
            #generation of array with coor for the highlighted area
            highl_coor_sort1 = extraction_highlighted_area(coor[0], fp_coords1)
            highl_coor_sort2 = extraction_highlighted_area(coor[1], fp_coords2)
            
            highl_coor_sort = (highl_coor_sort1, highl_coor_sort2)
            
            #plotting the highlighted area on the input data  
            fig3 = plt.figure()
            plt.plot(coor[0][:,0], coor[0][:,1],'k')
            plt.xlim(np.amin(coor[0][:,0])-1, np.amax(coor[0][:,0])+1)
            plt.ylim(np.amax(coor[0][:,1])+1, np.amin(coor[0][:,1])-1)
            plt.hold
            plt.plot(highl_coor_sort[0][:,0],highl_coor_sort[0][:,1],'red')
            fig3.canvas.manager.window.tkraise()
            #pauses for 10s until it closes the figures
            plt.pause(1)
            plt.close(fig3)
            
            fig4 = plt.figure()
            plt.plot(coor[1][:,0], coor[1][:,1],'k')
            plt.xlim(np.amin(coor[1][:,0])-1, np.amax(coor[1][:,0])+1)
            plt.ylim(np.amax(coor[1][:,1])+1, np.amin(coor[1][:,1])-1)
            plt.hold
            plt.plot(highl_coor_sort[1][:,0],highl_coor_sort[1][:,1],'red')            
            fig4.canvas.manager.window.tkraise()           
            #pauses for 10s until it closes the figures
            plt.pause(2)
            plt.close(fig4)           
            
            highl_coor_sort1 = np.divide(highl_coor_sort1, binning)
            highl_coor_sort2 = np.divide(highl_coor_sort2, binning)
            highl_coor_sort = (highl_coor_sort1, highl_coor_sort2)
            
            #saving highlighted area 
            np.save(resultpath + '/entr_highl_cell' + str(i)+ '.npy', highl_coor_sort)
            
        else: #only one fp is present       
            #selection of the area to be marked by clicking to create a polygon
            fig1 = plt.figure()
            plt.plot(coor[:,0], coor[:,1],'b')
            plt.axis('scaled')
            plt.xlim(np.amin(coor[:,0])-100, np.amax(coor[:,0])+100)
            plt.ylim(np.amax(coor[:,1])+100, np.amin(coor[:,1])-100)
            plt.title('Define polygonial area for highlighted fp entrance. \n left click: addition of point, right click: when finished with selection')
            fig1.canvas.manager.window.tkraise()
            plt.pause(0.1)
            fp_coords = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3) # extraction of coordinates
            plt.close(fig1)
            
            highl_coor_sort = extraction_highlighted_area(coor, fp_coords)
                          
            #plotting the highlighted area on the input data  
            fig2 = plt.figure()
            plt.plot(coor[:,0], coor[:,1],'k')
            plt.xlim(np.amin(coor[:,0])-1, np.amax(coor[:,0])+1)
            plt.ylim(np.amax(coor[:,1])+1, np.amin(coor[:,1])-1)
            plt.hold
            plt.plot(highl_coor_sort[:,0],highl_coor_sort[:,1],'red')
            fig2.canvas.manager.window.tkraise()
            #pauses for 10s until it closes the figures
            plt.pause(2)
            plt.close(fig2)

            highl_coor_sort = np.divide(highl_coor_sort,binning)

            #saving highlighted area 
            np.save(resultpath + '/entr_highl_cell' + str(i)+ '.npy', highl_coor_sort)
        
if __name__ == "__main__":
    #definition of input parameter
    path = '/Volumes/Reindeer/TrackingVSGAtto/MORN/CoorOutline200427'
    resultpath = '/Volumes/Reindeer/TrackingVSGAtto/MORN/CoorOutline200427'
    N = 20    
    binning = 160
    highlighting_fp(path, resultpath, binning, N)    