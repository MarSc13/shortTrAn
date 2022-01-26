#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:46:47 2019

@author: marieschwebs
"""

import numpy as np
import scipy.io as sio
import tifffile as tif
import matplotlib.pyplot as plt
from scipy import ndimage as scim
import alphashape
import tkinter
from tkinter import simpledialog
from PIL import Image, ImageDraw

'''Definitions of used functions'''
def create_countfield(img_array,input_mat): 
    for j in range(input_mat.shape[0]):
        img_array[input_mat[j][1]][input_mat[j][2]] =img_array[input_mat[j][1]][input_mat[j][2]] + 1
    return img_array

def downscaling(maxx,maxy,div,field):
    
    modulox = maxx - (np.floor(maxx/div))*div #determining how many nm px belong to the new pixelation of div
    moduloy = maxy - (np.floor(maxy/div))*div #determining how many nm px belong to the new pixelation of div
    startx = np.uint16(np.floor(modulox/2))   #determine which area of the field is used for binning, considers also the padding 
    endx = maxx -(modulox-np.floor(modulox/2))#determine which area of the field is used for binning, considers also the padding
    starty = np.uint16(np.floor(moduloy/2))   #determine which area of the field is used for binning, considers also the padding
    endy = maxy -(moduloy-np.floor(moduloy/2))#determine which area of the field is used for binning, considers also the padding
    scd_len_x = np.uint16((endx - startx)/div)#defining new size if the binned field
    scd_len_y = np.uint16((endy - starty)/div)#defining new size if the binned field
    
    scaled_field = np.zeros((scd_len_x,scd_len_y),'float') #creating the shell of the binned field   
    
    for i in range(scd_len_x): # ranges trough rows of binnend field
        for j in range(scd_len_y): #ranges trough columns of binnend field 
            for g in range(i*div+startx,(i+1)*div+startx): #ranges through rows of field that shall be binnend
                for k in range(j*div+starty,(j+1)*div+starty): #ranges through columns of field that shall be binnend
                    scaled_field[i,j] = scaled_field[i,j] + field[g,k] #addition of all entries in the to be binnend 
                    #field belonging to the same px in hte binnend field

    return scaled_field

def remove_outliers(mask,kdim,kshape):
    
    if kshape=='cross' and kdim==3: #cross kernel
        fkernel = np.array([[0,1,0],[1,1,1],[0,1,0]]) #cross kernel with kernel dim 3
    elif kshape=='cross' and kdim==5: #cross kernel
        fkernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]]) #cross kernel with kernel dim 5
    elif kshape=='box': #box kernel
        fkernel = np.ones((kdim,kdim))
        
    mask_fold = scim.convolve(mask, fkernel, mode='constant', cval=0)
    mask_fold_mply = np.multiply(mask_fold, mask)
   # '''This threshold deletes isolated entries/pixels/cells and binersiation of the mask'''
    out_mask=np.ones(mask_fold_mply.shape)
    out_mask[mask_fold_mply <= 1] =0

    return out_mask



'''Removes localization outside of the polygon'''
def removal_bg(pnts, bg):
    
    #query the maxima for imdim to create a mask
    width = int(np.amax(pnts[:,0])+1)
    height = int(np.amax(pnts[:,1])+1)
    
    #creation of a binary mask, px within the polygon line get the value 1
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(bg, outline=1, fill=1)
    mask = np.array(img)
    
    #definition of arrays
    pxcoor = np.zeros((1,2))      
    bg_rem = pnts.copy()
    
    #query of the coor if they lie within the polygon
    #coor outside are replaced with the value 0
    for j in range(0,pnts.shape[0]):
        pxcoor = pnts[j,:]
        pxcoor = pxcoor.astype(int)
        
        if mask[pxcoor[1],pxcoor[0]] == 1:
           bg_rem[j,0:2] = 0
    
    bg_rem = np.delete(bg_rem, np.where(~bg_rem.any(axis=1))[0], axis=0)
          
    return bg_rem


def extract_fp(pnts, coords):
    
    #query the maxima for imdim to create a mask
    width = int(np.amax(pnts[:,0])+1)
    height = int(np.amax(pnts[:,1])+1)
    
    #creation of a binary mask, px within the polygon line get the value 1
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(coords, outline=1, fill=1)
    mask = np.array(img)
    
    #definition of arrays
    pxcoor = np.zeros((1,2))      
    coor_fp = np.zeros(pnts.shape) 
    
    #query of the coor if they lie within the polygon
    #coor outside are replaced with the value 0
    for j in range(0,pnts.shape[0]):
        pxcoor = pnts[j,:]
        pxcoor = pxcoor.astype(int)
        
        if mask[pxcoor[1],pxcoor[0]] == 1:
           coor_fp[j,:] = pnts[j,:]
    
    coor_fp = np.delete(coor_fp, np.where(~coor_fp.any(axis=1))[0], axis=0)
    
    return coor_fp





'''Main function: evaluation of MRON signal'''
def evaluation_MORN_signal(path_morn, countfield_path, scale_input, px_padding, path_im_sec, path_start_end_points_binning, binning_factor, resultpath, N):
#    tic = time.time() # in seconds
    
    pnts_hook_cells = []
    n_hook_cells = []
    coords_cells = []
    
    for i in range(1, N+1):
              
        morn = sio.loadmat(path_morn + str(i) + '.mat')  # imread of the MORN mat file
        for key in morn: #extraction of the array name
            1+1
        morn = morn[key] #extraction of the array out of the dictionary
        morn = morn[:,1:3] #reduction of the array to only contain frame, xpos, ypos
    
#        morn = np.array(morn)
        """ nm Skalierung deswegen 160 """
        morn = morn*scale_input # scales px to nm
    
        morn_unint16 = np.int16(morn).copy() #reduction of datatype
        
        par_im_sec = np.load(path_im_sec + str(i) + '.npy',allow_pickle=True) #loads parameter from trajecory evaluation
        start_end = np.load(path_start_end_points_binning + str(i) + '/par_im_sec_cell.npy',allow_pickle=True)
        
        #addition of 160 pix used for padding and resetting both axis equal to 
        #VSG data in padding and binning
        morn_unint16[:,0] = morn_unint16[:,0] - par_im_sec[0,0] - start_end[0,0] + px_padding 
        morn_unint16[:,1] = morn_unint16[:,1] - par_im_sec[1,0] - start_end[1,0] + px_padding 


#        #has to be the same size as VSG data in and after binning process # commented 200425
#        x_length = int(start_end[0,1] - start_end[0,0]) #addtion of 160x1nm pix at the end in x direction
#        y_length = int(start_end[1,1] - start_end[1,0]) #addtion of 160x1nm pix at the end in y direction
#        
        shape = (np.amax(morn_unint16[:,0]),np.amax(morn_unint16[:,1]))        
           
        '''removal of background signal'''
        pnts = morn_unint16.copy()
        fig2 = plt.figure()
        plt.plot(pnts[:,0],pnts[:,1],'bo',markersize=1)
        plt.axis('scaled')
#        plt.gca().invert_yaxis()
        plt.xlim(-500, shape[0]+500)
        plt.ylim(shape[1]+500,-500)
        plt.xticks([])
        plt.yticks([])
        
        #creatig input window to remove background
        parent = tkinter.Tk() #Create the object
        parent.overrideredirect(1) #Avoid it appearing and then disappearing quickly
        parent.iconbitmap("PythonIcon.ico")# Set an icon (this is optional - must be in a .ico format)
        parent.withdraw() # Hide the window as we do not want to see this one
        
        removal = simpledialog.askstring('Dialog Title', 'Necessity to remove background? Yes(y) / No(n)', parent=parent)
        plt.close(fig2)        
        
        while removal == 'y':
            fig2 = plt.figure()
            plt.plot(pnts[:,0],pnts[:,1],'bo',markersize=1)
            plt.axis('scaled')
            plt.xlim(-500, shape[0]+500)
            plt.ylim(shape[1]+500,-500)
            plt.xticks([])
            plt.yticks([])
            plt.title('Click to define points for polygon outline which will be removed. \n Left click: add points, right click: when finished')
            fig2.canvas.manager.window.tkraise()
            plt.pause(0.1)
            bg_rem = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3) # extraction of coordinates
            plt.close(fig2)
            
            pnts = removal_bg(pnts, bg_rem)
            
            shape = (np.amax(pnts[:,0]),np.amax(pnts[:,1]))        

            fig = plt.figure()
            plt.plot(pnts[:,0],pnts[:,1],'bo',markersize=1) 
            plt.axis('scaled')
            plt.xlim(-500, shape[0]+500)
            plt.ylim(shape[1]+500,-500)
            plt.xticks([])
            plt.yticks([])
            plt.title('Localizations after removal')
        
            #creatig input window to stay in or leave the while loop 
            parent = tkinter.Tk() # Create the object
            parent.overrideredirect(1) # Avoid it appearing and then disappearing quickly
            parent.iconbitmap("PythonIcon.ico") # Set an icon (this is optional - must be in a .ico format)
            parent.withdraw() # Hide the window as we do not want to see this one
            
            removal = simpledialog.askstring('Dialog Title', 'Necessity to remove further background? Yes(y) / No(n)', parent=parent)
            plt.close(fig)
        
        fig3 = plt.figure()
        plt.plot(pnts[:,0],pnts[:,1],'bo',markersize=1) 
        plt.axis('scaled')
        plt.xlim(-500, shape[0]+500)
        plt.ylim(shape[1]+500,-500)
        plt.xticks([])
        plt.yticks([])
        plt.title('Localizations after removal cell ' + str(i))
        
        pnts_hook_cells.append(pnts)
        
        '''creating input window to specify if cells has one or two flagellar pockets'''
        parent = tkinter.Tk() # Create the object
        parent.overrideredirect(1) # Avoid it appearing and then disappearing quickly
        parent.iconbitmap("PythonIcon.ico") # Set an icon (this is optional - must be in a .ico format)
        parent.withdraw() # Hide the window as we do not want to see this one
        
        fenster = simpledialog.askinteger('Dialog Title', 'How many flagellar pockets are present?', minvalue=1, maxvalue=3, parent=parent)
        plt.close(fig3)   
        n_hook_cells.append(fenster)
        

        #extraction of the region containing the flagellar pocket, done to 
        #exclude some unspecific signal
        coords = []
        fig4 = plt.figure()
        plt.plot(pnts[:,0],pnts[:,1],'bo',markersize=1)
        plt.axis('scaled')
#        plt.xlim(-500, shape[0]+400)
#        plt.ylim(shape[1]+400,-500)
        plt.xticks([])
        plt.yticks([])
        plt.title('Define region of the flagellar pocket by clicking the polygon outline')
        if fenster == 2:
            coords1 = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3)
            coords2 = plt.ginput(n=-1, timeout=0, mouse_pop= [], mouse_stop=3)
            coords = (coords1,coords2)
            coords = list(coords)
        else:
            coords = pnts.copy()
        plt.close(fig4)
        
        coords_cells.append(coords)
        
        #deletes entries which prevents that entries of other cells are wrongly assigned
        coords = []
        coords1 = []
        coords2 = []
        del morn, morn_unint16, pnts
    
    np.save(resultpath + '/coords_cells.npy', coords_cells, allow_pickle=True)
    np.save(resultpath + '/n_hook_cells.npy', n_hook_cells, allow_pickle=True)
    np.save(resultpath + '/pnts_hook_cells.npy', pnts_hook_cells, allow_pickle=True)

    coords_cells = np.load(resultpath + '/coords_cells.npy', allow_pickle=True)
    n_hook_cells = np.load(resultpath + '/n_hook_cells.npy', allow_pickle=True)
    pnts_hook_cells = np.load(resultpath + '/pnts_hook_cells.npy', allow_pickle=True)
        
    for i in range(1, N+1):
        
        binned_ar = tif.imread(countfield_path + str(i) + '/scaled_pointfield.tif')
        binned_ar = binned_ar.T # correct assignment of x and y axis
        
        '''make concave hull'''
        if n_hook_cells[i-1] == 1:
            
#            coords = coords_cells[i-1]
            pnts_fp = pnts_hook_cells[i-1]
            
#            #defines area of fp, unspecfic background will be neglected # commented 200425
#            pnts_fp = extract_fp(pnts, coords) 
                  
            # to generate  the concave hull, first of all the array has to be 
            # transfomed into a list of tuples
            pnts_fp = tuple(map(tuple,pnts_fp))
            
            #calculation of the hull of pnts_fp
            alpha_fp = 0.90 * alphashape.optimizealpha(pnts_fp) # 0=konvex hull
            hull_fp = alphashape.alphashape(pnts_fp, alpha_fp)
            hull_pts_fp = hull_fp.exterior.coords.xy
            hull_pts_fp = np.array(hull_pts_fp,'float')
            #prefer working with x in first column and y in second column
            hull_pts_fp = hull_pts_fp.T 
            
            #def of matrix for scaled outline
            hull_pts_fp_scal = np.zeros(hull_pts_fp.shape)
            
#            #calculation of the scaling factor
#            x_factor = np.divide(binned_ar.shape[1],x_length)
#            y_factor = np.divide(binned_ar.shape[0],y_length)
            
            #scaling of the SM localizations to be in the same scale as data is
            #after binning
            hull_pts_fp_scal[:,0] = np.divide(hull_pts_fp[:,0],binning_factor)
            hull_pts_fp_scal[:,1] = np.divide(hull_pts_fp[:,1],binning_factor)
            
            #storing data as tuple
            coor_outline_raw = (hull_pts_fp)
            coor_outline_scal = (hull_pts_fp_scal)
            
            #save coor_outline in the result folder
            np.save(resultpath + '/coor_outline_raw_' + str(i) , coor_outline_raw, allow_pickle=True)      
            np.save(resultpath + '/coor_outline_scal_' + str(i) , coor_outline_scal, allow_pickle=True)      


        elif n_hook_cells[i-1] == 2:
            
            pnts = pnts_hook_cells[i-1] #all loc 
            
            coords = coords_cells[i-1]
            coords1 = coords[0] #outline coords of first region
            coords2 = coords[1] #outline coords of second region
            
            #defines area of one fp, removes then loc of first fp, loc left
            #are used for the second fp
            pnts_fp1 = extract_fp(pnts, coords1)
            pnts_fp2 = extract_fp(pnts, coords2)
        
            # generation of the concave hull, first of all arrays will be 
            # transfomed into a list of tuples
            pnts_fp1 = tuple(map(tuple,pnts_fp1))
            pnts_fp2 = tuple(map(tuple,pnts_fp2))
            
            #calculation of the hull of pnts_fp1, then of pnts_fp2 
            alpha_fp1 = 0.90 * alphashape.optimizealpha(pnts_fp1) # 0=konvex hull
            hull_fp1 = alphashape.alphashape(pnts_fp1, alpha_fp1)
            hull_pts_fp1 = hull_fp1.exterior.coords.xy
            hull_pts_fp1 = np.array(hull_pts_fp1,'float')
            #prefer working with x in first column and y in second column
            hull_pts_fp1 = hull_pts_fp1.T 
    
            
            alpha_fp2 = 0.90 * alphashape.optimizealpha(pnts_fp2) # 0=konvex hull
            hull_fp2 = alphashape.alphashape(pnts_fp2, alpha_fp2)
            hull_pts_fp2 = hull_fp2.exterior.coords.xy
            hull_pts_fp2 = np.array(hull_pts_fp2,'float')
            #prefer working with x in first column and y in second column
            hull_pts_fp2 = hull_pts_fp2.T 
           
            #def matrices for scaled outline
            hull_pts_fp1_scal = np.zeros(hull_pts_fp1.shape)
            hull_pts_fp2_scal = np.zeros(hull_pts_fp2.shape)
            
                        
            #scaling of the SM localizations with the binning factor to be in the same scale as data is
            #after binning
            hull_pts_fp1_scal = np.divide(hull_pts_fp1,binning_factor)
            hull_pts_fp2_scal = np.divide(hull_pts_fp2,binning_factor)
            
            #storing data in tuple               
            coor_outline_raw = (hull_pts_fp1, hull_pts_fp2)
            coor_outline_scal = (hull_pts_fp1_scal, hull_pts_fp2_scal)
            
            #save coor_outline in the result folder
            np.save(resultpath + '/coor_outline_raw_' + str(i) , coor_outline_raw, allow_pickle=True)
            np.save(resultpath + '/coor_outline_scal_' + str(i) , coor_outline_scal, allow_pickle=True)
            

    return coor_outline_raw, coor_outline_scal 
    
    
if __name__ == "__main__":
 
    #path to the folder of the input data
    path_morn = '/Volumes/Reindeer/TrackingVSGAtto/MORN/ForPython/MORNshifted'
    
    #used as template for the creation of the feature space 
    countfield_path = '/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Trc'
    
    #pixelation of camera (160 nm) used for the convertion to a nm scale
    scale_input = 160 
    
    #corresponds to the px_adding in the trajectory analysis \
    #(padding of X x1nm pix at the beginning and end of the x and y direction, respectively)
    px_padding = 160 
    
    #Adjustment of the feature space to the feature space of the trajectory analysis.
    #file generated in padding.py and saved in results/... 
    path_im_sec = '/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/par_im_sec_cell'
    
    #Adjustment of the feature space to the feature space of the trajectory analysis.
    #file generated in binning.py and saved to the subfolder results/TrcN/...
    path_start_end_points_binning = '/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Trc'
    
    #Scaling of the coordinates by the binning factor (trajectory analysis)
    binning_factor = 160
    
    #resultpath to the folder to save the final outline
    resultpath = '/Volumes/Reindeer/TrackingVSGAtto/MORN/CoorOutline200427'
    
    N = 20 #number of datasets to be evaluated
    
    evaluation_MORN_signal(path_morn, countfield_path, scale_input, px_padding, path_im_sec, path_start_end_points_binning, binning_factor, resultpath, N)