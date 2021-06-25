#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:58:28 2020

@author: marieschwebs
"""

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib import cm
from math import atan2,degrees
from copy import deepcopy
import os

plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

'''relative and absolute maximum error associated with the velocity
compared with http://davbohn.userpage.fu-berlin.de/physcalc/'''
def create_rerr_array(array_x, array_y, array_speed, sigma):
    relErr_Velocity = np.zeros(array_speed.shape) #preinit
    absErr_Velocity = np.zeros(array_speed.shape)
    for a in range(array_speed.shape[0]):#iteration over lines
        for b in range(array_speed.shape[1]): #iteration over columns
            if array_speed[a,b] !=0: #divides only when array !=0 else 0
                 # v = sqrt(v_x^2+v_y^2)
                 # first calculate the error associated with v_x^2 (i) and v_y^2 (ii)
                 # note: v = deltax/t with deltax = deltax +/- 2sigma
                 # error of square: a = b^2 -> delta a = 2* delta b/b *a
                if array_x[a,b] !=0:
                    err_1 = 2*(2*sigma/abs(array_x[a,b]))*array_x[a,b]**2
                else:
                    err_1 = 0
                if array_y[a,b] !=0:
                    err_2 = 2*(2*sigma/abs(array_y[a,b]))*array_y[a,b]**2
                else:
                    err_2 = 0
                # sum of iii = i + ii
                # c = a+b -> delta c = delta a + delta b;
                err_sum = err_1 + err_2;
                # sqrt(iii)
                # c = sqrt(a) -> delta c = 1/2* delta a/a *c;
                relErr_Velocity[a,b] = 1/2* err_sum/(array_x[a,b]**2+array_y[a,b]**2);
                absErr_Velocity[a,b] = 1/2* err_sum/(array_x[a,b]**2+array_y[a,b]**2) * np.sqrt(array_x[a,b]**2+array_y[a,b]**2);
    return relErr_Velocity, absErr_Velocity  

'''function to extract the fp region from the original arrays with with following rescaling of 
both axis for fp region plots'''
def extraction_fp_region(entr, outline, img_x, img_y, angle, img_l, px_extent):
    
    #def of arrays, later on containing extremes of x and y values
    fp_ext = np.zeros((2,2))
    
    #find ind to extract section from whole arrays      
    fp_ext[0,0] = round(np.amin(entr[:,0])) - px_extent #always round down
    fp_ext[0,1] = round(np.amax(entr[:,0])) + px_extent#always round up
    fp_ext[1,0] = round(np.amin(entr[:,1])) - px_extent
    fp_ext[1,1] = round(np.amax(entr[:,1])) + px_extent
                
    
    #conversion to integer for later extraction
    fp_ext = fp_ext.astype(int)
    
    outline_fp = deepcopy(outline)
    
    #creation of deepcopies and correction of outline coor for section 
    #images. In section coor system is reset to 0
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
    if fp_ext[0,1] > img_x.shape[1]-1: #max x value
        fp_ext[0,1] = img_x.shape[1]-1
    if fp_ext[1,1] > img_x.shape[0]-1: #max y value
         fp_ext[1,1] = img_x.shape[0]-1
    
    
    #extraction of the fp section, from a till b+1 because of python structure, limit b is excluded 
    img_x_fp = img_x[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    img_y_fp = img_y[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    angle_fp = angle[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    img_l_fp = img_l[fp_ext[1,0]:fp_ext[1,1]+1,fp_ext[0,0]:fp_ext[0,1]+1].copy()
    
    return img_x_fp, img_y_fp, angle_fp, img_l_fp, entr_fp, outline_fp



'''Generates the heat map and the quiver plots of the whole cell'''
def HeatMap_QuiverPlot_WholeCell(img_x, img_y, img_l, angle, outline, entr, colormap, resultpath, i):
    
    if np.size(outline) == 2:
        if np.size(outline[0]) == np.size(entr[0]):
            col1 = 'k'
            col2 = 'k'
        else:
            col1 = 'r'
            col2 = '#03E1F0'
    else:
        if np.size(outline) == np.size(entr):
            col1 = 'k'
            col2 = 'k'
        else:
            col1 = 'r'
            col2 = '#03E1F0'
        
        
    #Generation for the grid of the whole picture. 0/0 is in the upper 
    #left corner
    Y,X = np.mgrid[0:img_x.shape[0], 0:img_y.shape[1]]
    
    '''To display the correct orientation of incomming speed - X, Y, 
    img_x, img_y - have to  be placed in exactly this order wihtin the 
    quiver function. This was checked on 24.09.18 MS'''
    fig1 = plt.figure()    
    plt.quiver(X, Y, img_x, img_y, angle, headwidth=5, headlength=5, headaxislength=5, linewidth=.5, 
               minshaft=1, scale_units='inches', scale=60, pivot='mid',
               alpha=.5, cmap=colormap) 
    plt.axis('scaled')
    plt.xlim(0-1, img_x.shape[1])
    plt.ylim(img_y.shape[0], 0-1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_l, cmap='binary', alpha=.01)
    #plt.hold
    if np.size(outline) == 2:
        plt.plot(outline[0][:,0],outline[0][:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(outline[1][:,0],outline[1][:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(entr[0][:,0],entr[0][:,1], color = col1, linewidth=0.85)
        #plt.hold
        plt.plot(entr[1][:,0],entr[1][:,1], color = col1, linewidth=0.85)
    else:
        plt.plot(outline[:,0],outline[:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(entr[:,0],entr[:,1], color = col1)
    tif.imsave(resultpath+'/Info/Speed/Speed_Cell'+str(i)+'.tif', img_l)
    plt.tight_layout()
    plt.savefig(resultpath+'QuiverPlotSpeed_Cell'+str(i)+'.png', dpi=300)
    plt.savefig(resultpath+'QuiverPlotSpeed_Cell'+str(i)+'.pdf', dpi=300)
    plt.close(fig1)

    
    '''Generation of an speed heat map with the final unit of mum/s'''
    plt.rcParams['image.cmap'] = 'RdPu' # changes current colorbar
    #mask array to assign pixels with entry 0 to white color
    img_l_mask = np.ma.array(img_l, mask=(img_l == 0))
    #assign zero to white color in current colormap
    current_cmap = copy.copy(mpl.cm.get_cmap('RdPu'))
    current_cmap.set_bad(color='white')
       
    fig2 = plt.figure()
    plt.imshow(img_l_mask)
    plt.xticks([])
    plt.yticks([])
    cbar=plt.colorbar()    
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('µm/s')
    #plt.hold
    if np.size(outline) == 2:
        plt.plot(outline[0][:,0],outline[0][:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(outline[1][:,0],outline[1][:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(entr[0][:,0],entr[0][:,1], color = col2, linewidth=0.85)
        #plt.hold
        plt.plot(entr[1][:,0],entr[1][:,1], color = col2, linewidth=0.85)
    else:
        plt.plot(outline[:,0],outline[:,1], color = 'k', linewidth=0.85) 
        #plt.hold
        plt.plot(entr[:,0],entr[:,1], color = col2, linewidth=0.85)
    plt.tight_layout()
    plt.savefig(resultpath+'HeatMap_Cell'+str(i), dpi=300)
    plt.savefig(resultpath+'HeatMap_Cell'+str(i)+'.pdf', dpi=300)
    plt.close(fig2)
    
    return


'''Generates the heat map and the quiver plot for the section of the flagellar pocket entrance'''
def HeatMap_QuiverPlot_Section(img_x_fp, img_y_fp, img_l_fp, angle_fp, outline_fp,\
                               entr_fp, size_outline, colormap, resultpath, i, sec):
    
    #creation of a grid to place ellipsoids in plot
    Y_fp,X_fp = np.mgrid[0:img_x_fp.shape[0], 0:img_y_fp.shape[1]] 
    
    '''Plotting section of fp'''
    fig3 = plt.figure()
    plt.quiver(X_fp, Y_fp, img_x_fp, img_y_fp, angle_fp, headwidth=7, headlength=7, headaxislength=7,linewidth=.5, 
               minshaft=1, scale_units='inches', scale=30, pivot='mid',
               alpha=.5, cmap=colormap) 
#             plt.quiver(X, Y, img_x, img_y, angle, headwidth=5, linewidth=1.2, 
#                       minshaft=0.2, scale_units='height', scale=100, pivot='mid',
#                       alpha=.5, cmap=wbmap) 
    plt.xlim(0-1, img_x_fp.shape[1])
    plt.ylim(img_y_fp.shape[0], 0-1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_l_fp, cmap='binary', alpha=.01)
    #plt.hold
    plt.plot(outline_fp[:,0],outline_fp[:,1], color = 'k', linewidth=4)  
    #plt.hold
    plt.plot(entr_fp[:,0],entr_fp[:,1], color = 'r', linewidth=4)
    if size_outline == 2:
        tif.imsave(resultpath+'/Info/Speed/Sec_Speed_Cell'+str(i)+'Sec'+str(sec)+'.tif', img_l_fp)
        tif.imsave(resultpath+'/Info/Angle/Angle_Speed_Cell'+str(i)+'Sec'+str(sec)+'.tif', angle_fp)
        tif.imsave(resultpath+'/Info/Speed/Xpos_Speed_Cell'+str(i)+'Sec'+str(sec)+'.tif', img_x_fp)
        tif.imsave(resultpath+'/Info/Speed/Ypos_Speed_Cell'+str(i)+'Sec'+str(sec)+'.tif', img_y_fp)
        plt.tight_layout()
        plt.savefig(resultpath+'Sec_QuiverPlotSpeed_Cell'+str(i)+'Sec'+str(sec)+'.png', dpi=300)
        plt.savefig(resultpath+'Sec_QuiverPlotSpeed_Cell'+str(i)+'Sec'+str(sec)+'.pdf', dpi=300)
    else:
        tif.imsave(resultpath+'/Info/Speed/Sec_Speed_Cell'+str(i)+'Sec.tif', img_l_fp)
        tif.imsave(resultpath+'/Info/Angle/Angle_Speed_Cell'+str(i)+'Sec.tif', angle_fp)
        tif.imsave(resultpath+'/Info/Speed/Xpos_Speed_Cell'+str(i)+'Sec.tif', img_x_fp)
        tif.imsave(resultpath+'/Info/Speed/Ypos_Speed_Cell'+str(i)+'Sec.tif', img_y_fp)
        plt.tight_layout()
        plt.savefig(resultpath+'Sec_QuiverPlotSpeed_Cell'+str(i)+'Sec.png', dpi=300) 
        plt.savefig(resultpath+'Sec_QuiverPlotSpeed_Cell'+str(i)+'Sec.pdf', dpi=300) 
    plt.close(fig3)
    
    '''Generation of an speed heat map with the final unit of mum/s'''       
    fig4 = plt.figure()
    plt.imshow(img_l_fp,cmap='RdPu')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.5,img_l_fp.shape[1]-0.5])
    plt.ylim([img_l_fp.shape[0]-0.5,-0.5])
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    #plt.hold
    plt.plot(outline_fp[:,0],outline_fp[:,1], color = 'k', linewidth=4)
    #plt.hold
    plt.plot(entr_fp[:,0],entr_fp[:,1], color = '#03E1F0', linewidth=4)
    cbar.set_label('µm/s')
    plt.tight_layout()
    if size_outline == 2:
        plt.savefig(resultpath+'Sec_HeatMap_Cell'+str(i)+'Sec'+str(sec), dpi=300)
        plt.savefig(resultpath+'Sec_HeatMap_Cell'+str(i)+'Sec'+str(sec)+'.pdf', dpi=300)
    else:
        plt.savefig(resultpath+'Sec_HeatMap_Cell'+str(i)+'Sec', dpi=300)
        plt.savefig(resultpath+'Sec_HeatMap_Cell'+str(i)+'Sec.pdf', dpi=300)
    plt.close(fig4)
    
    return


'''Generates Relative Error map'''
def HeatMap_RelErr(relErr, outline, entr, resultpath, i):
    #color scheme for highlighting region wrapped around fp entrance
    if np.size(outline) == 2:
        if np.size(outline[0]) == np.size(entr[0]):
            col1 = 'k'
        else:
            col1 = 'r'
    else:
        if np.size(outline) == np.size(entr):
            col1 = 'k'
        else:
            col1 = 'r'
            
    #tresh for proper distr. of cbar
    relErr[relErr > 1] = 1 #apply thres for better resol in map and cbar

    plt.rcParams['image.cmap'] = 'viridis' # changes current colorbar
    #mask array to assign pixels with entry 0 to white color
    relErr_mask = np.ma.array(relErr, mask=(relErr == 0))
    #assign zero to white color in current colormap
    current_cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    current_cmap.set_bad(color='white')

    fig4 = plt.figure()
    plt.imshow(relErr_mask)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('relative standard error')
    #plt.hold
    if np.size(outline) == 2:
        plt.plot(outline[0][:,0],outline[0][:,1], color = 'k', linewidth=0.95) 
        #plt.hold
        plt.plot(outline[1][:,0],outline[1][:,1], color = 'k', linewidth=0.95) 
        #plt.hold
        plt.plot(entr[0][:,0],entr[0][:,1], color = col1, linewidth=0.95)
        #plt.hold
        plt.plot(entr[1][:,0],entr[1][:,1], color = col1, linewidth=0.95)
    else:
        plt.plot(outline[:,0],outline[:,1], color = 'k', linewidth=0.95) 
        #plt.hold
        plt.plot(entr[:,0],entr[:,1], color = col1,linewidth=0.95)
    plt.tight_layout()
    plt.savefig(resultpath+'RelErrMap_Cell'+str(i), dpi=300)   
    plt.savefig(resultpath+'RelErrMap_Cell'+str(i)+'.pdf', dpi=300) 
    plt.close(fig4)
    return



'''main function: generation of the heat map and the quiver plot of the directed motion'''
def directed_motion_plus_outline(path,filename,outline_path,resultpath,binning,\
                                 sigma,time_lag,px_extent,N,highlighting,errcorr):
    
    #Generation of the resultfolder
    if not os.path.exists(resultpath+'/DirectedMotionMaps'):
            os.mkdir(resultpath+'/DirectedMotionMaps')
            resultpath = resultpath+'/DirectedMotionMaps/'
    else:
            resultpath = resultpath+'/DirectedMotionMaps/'
            
    #Generation of folder for informations      
    if not os.path.exists(resultpath+'/Info'):
            os.mkdir(resultpath+'/Info')
            os.mkdir(resultpath+'/Info/Angle')
            os.mkdir(resultpath+'/Info/Speed')
            if errcorr == 'Yes' or errcorr == 'yes':
                os.mkdir(resultpath+'/Info/RelErr')
    
    speed_info=np.zeros((N+2,3)) #Nx2 array: first columns maximum, 
    #second column average, two additional rows to seperate the calculated 
    #mean of each column from the single cell data
    
    speed_info_sec = np.zeros((1,4))
    
    for i in range(1,N+1):
       
        directory=path+str(i)+filename
    
        '''Loads the x and y components to calcualte the speed of the directed 
        motion within an pixel'''
        img_x = tif.imread(directory+'scld_vec_x_fil.tif')
        img_y = tif.imread(directory+'scld_vec_y_fil.tif')
        count_fil = tif.imread(directory+'scld_count_fil.tif') #used in ll 359
        
        
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
                           
        tif.imsave(resultpath+'/Info/Angle/AngleSpeed_Cell'+str(i)+'.tif', angle_deg) 
        
        '''Genration of the black white colormap used by the quiver plot'''
        binary = cm.get_cmap('binary', 256)
        wb = binary(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        wb[15:256, :] = black
        wbmap = ListedColormap(wb)
        
        fourcmap = colors.ListedColormap(['white', 'darkorange', 'b', 'darkcyan','m'])
        bounds=[0,1,2,3,4,5]
        norm = colors.BoundaryNorm(bounds, fourcmap.N)
        
        '''Load points to draw outline in map'''
        outline = np.load(outline_path + '/coor_outline_scal_' + str(i) + '.npy',allow_pickle=True)
        if highlighting == 'Yes':
            entr = np.load(outline_path + '/entr_highl_cell' + str(i) + '.npy',allow_pickle=True)
        size_outline = np.size(outline)
        
        if errcorr == 'Yes' or errcorr == 'yes':
            '''calculation of the relErr absErr arrays'''
            sigma_dt = sigma/time_lag
            relErr, absErr = create_rerr_array(img_x, img_y, img_l, sigma_dt)
            count_sqrt = np.sqrt(count_fil)
            relErr = np.divide(relErr, count_sqrt, out=np.zeros_like(relErr), where=count_sqrt!=0)
            absErr = np.divide(absErr, count_sqrt, out=np.zeros_like(absErr), where=count_sqrt!=0)
            tif.imsave(resultpath + '/Info/RelErr/rel_error_cell' + str(i) + '.tif',relErr)
            tif.imsave(resultpath + '/Info/RelErr/abs_error_cell' + str(i) + '.tif',absErr)
        
        
        if size_outline == 2:
            outline = tuple(outline)
            
            if highlighting == 'Yes':
                entr = tuple(entr)
                
                '''extraction of the region surrounding the first fp'''
                img_x_fp1, img_y_fp1, angle_fp1, img_l_fp1, entr_fp1, outline_fp1 = \
                extraction_fp_region(entr[0], outline[0], img_x, img_y, angle, img_l, px_extent)
                
                '''#extraction of the region surrounding the second fp'''
                img_x_fp2, img_y_fp2, angle_fp2, img_l_fp2, entr_fp2, outline_fp2 = \
                extraction_fp_region(entr[1], outline[1], img_x, img_y, angle, img_l, px_extent)
            
            
                '''Generates quiver plot and heat map fpr the whole cell'''
                HeatMap_QuiverPlot_WholeCell(img_x, img_y, img_l, angle, outline,\
                                             entr, wbmap, resultpath, i)
                
                if errcorr == 'Yes' or errcorr == 'yes':
                    '''Generates relative error map for the whole cell'''
                    HeatMap_RelErr(relErr, outline, entr, resultpath, i)    
                    
                sec = 1 #serves as input parameter for saving the first section plot
                
                '''Generates quiver plot and heat map for the section surrounding the first fp entrance'''
                HeatMap_QuiverPlot_Section(img_x_fp1, img_y_fp1, img_l_fp1, angle_fp1, \
                                           outline_fp1, entr_fp1, size_outline, \
                                           wbmap, resultpath, i, sec)
    
                sec = 2 #serves as input parameter for saving the second section plot
    
                '''Generates quiver plot and heat map for the section surrounding the second fp entrance'''
                HeatMap_QuiverPlot_Section(img_x_fp2, img_y_fp2, img_l_fp2, angle_fp2,\
                                           outline_fp2, entr_fp2, size_outline, \
                                           wbmap, resultpath, i, sec)
                
                '''Collecting speed info of sections'''
                #collecting infos about max value in diff coef and average val of each cell
                sec_info = np.zeros((2,4))
                #info fp1
                sec_info[0,0] = i
                sec_info[0,1] = np.nanmax(img_l_fp1)
                img_l_fp1[img_l_fp1 == 0] = np.nan
                sec_info[0,2] = np.nanmean(img_l_fp1)
                sec_info[0,3] = np.nanmedian(img_l_fp1)
                #info fp2
                sec_info[1,0] = i
                sec_info[1,1] = np.nanmax(img_l_fp2)
                img_l_fp2[img_l_fp2 == 0] = np.nan
                sec_info[1,2] = np.nanmean(img_l_fp2)
                sec_info[1,3] = np.nanmedian(img_l_fp2)
                
                speed_info_sec=np.append(speed_info_sec, sec_info, axis=0)
            
            else:
                '''Generates quiver plot and heat map fpr the whole cell'''
                HeatMap_QuiverPlot_WholeCell(img_x, img_y, img_l, angle, outline,\
                                             outline, wbmap, resultpath, i)
                
                if errcorr == 'Yes' or errcorr == 'yes':
                    '''Generates relative error map for the whole cell'''
                    HeatMap_RelErr(relErr, outline, outline, resultpath, i)
            
            
        else:
            if highlighting == 'Yes':
                '''extraction of the region surrounding the fp'''
                img_x_fp, img_y_fp, angle_fp, img_l_fp, entr_fp, outline_fp = \
                extraction_fp_region(entr, outline, img_x, img_y, angle, img_l, px_extent)
                
                '''Generates quiver plot and heat map fpr the whole cell'''
                HeatMap_QuiverPlot_WholeCell(img_x, img_y, img_l, angle, outline,\
                                             entr, wbmap, resultpath, i)
                
                if errcorr == 'Yes' or errcorr == 'yes':
                    '''Generates relative error map for the whole cell'''
                    HeatMap_RelErr(relErr, outline, entr, resultpath, i)
                
                sec = 1 #serves as input parameter for saving the section plot
    
                '''Generates quiver plot and heat map for the section surrounding the fp entrance'''
                HeatMap_QuiverPlot_Section(img_x_fp, img_y_fp, img_l_fp, angle_fp,\
                                           outline_fp, entr_fp, size_outline, \
                                           wbmap, resultpath, i, sec)
                
                
                '''Collecting speed info of sections'''
                #collecting infos about max value in diff coef and average val of each cell
                sec_info = np.zeros((1,4))
                #info fp1
                sec_info[0,0] = i
                sec_info[0,1] = np.nanmax(img_l_fp)
                img_l_fp[img_l_fp == 0] = np.nan
                sec_info[0,2] = np.nanmean(img_l_fp)
                sec_info[0,3] = np.nanmedian(img_l_fp)
                
                
                speed_info_sec=np.append(speed_info_sec, sec_info, axis=0)
            
            else:
                '''Generates quiver plot and heat map fpr the whole cell'''
                HeatMap_QuiverPlot_WholeCell(img_x, img_y, img_l, angle, outline,\
                                             outline, wbmap, resultpath, i)
                 
                if errcorr == 'Yes' or errcorr == 'yes':
                    '''Generates relative error map for the whole cell'''
                    HeatMap_RelErr(relErr, outline, outline, resultpath, i)
            
        
        maximum = np.nanmax(img_l)
        img_l[img_l == 0] = np.nan
        average = np.nanmean(img_l)
        median = np.nanmedian(img_l)
        
        speed_info[i-1,:]=maximum, average, median
   
    if N > 1: #does not make sense to clc mean over just one value
        speed_info[-1,0] = np.nanmean(speed_info[0:-2,0])
        speed_info[-1,1] = np.nanmean(speed_info[0:-2,1])
        speed_info[-1,2] = np.nanmean(speed_info[0:-2,2])
    
    tif.imsave(resultpath+'/Info/Speed/Speed_info_all_bin'+str(binning)+'.tif',speed_info)
    tif.imsave(resultpath+'/Info/Speed/Speed_info_all_sec'+str(binning)+'.tif',speed_info_sec)
    
if __name__ == "__main__":
    N = 2 #number to be evaluated cells
    binning = 160 #binning to x nm
    px_extent = 4  #def of number of px surrounding fp outline
    resultpath = '/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Maps_plus_Outline_corr'
    path='/Volumes/Reindeer/TrackingVSGAtto/Trc/Eval_outgoing/Trc'
    filename = '/results_filtering/'
    outline_path = '/Volumes/Reindeer/TrackingVSGAtto/MORN/CoorOutlineCorrected'
    highlighting = 'Yes'
    #directory = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Trc/Trc_thresh2/Trc'+str(N)+'/'
    directed_motion_plus_outline(path,filename,outline_path,resultpath,binning,\
                                 sigma,time_lag,px_extent,N,highlighting,errcorr)