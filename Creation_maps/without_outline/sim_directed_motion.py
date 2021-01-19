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

'''main function'''
def directed_motion(path,filename,resultpath,binning,sigma,t_lag,N,errcorr):
    
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
    
    
    for i in range(1,N+1):
        '''Loads the bright field image'''
    #    im=tif.imread('/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/BFimages/BF2/img_000000000_Default_000.tif')
    #    plt.imshow(im, cmap='Greys')
        
        directory=path+str(i)+filename
    
        '''Loads the x and y components to calcualte the speed of the directed 
        motion within an pixel'''
        img_x = tif.imread(directory+'scld_vec_x_fil.tif')
        img_y = tif.imread(directory+'scld_vec_y_fil.tif')
        count_fil = tif.imread(directory+'scld_count_fil.tif')
        
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
        
        if errcorr == 'Yes' or errcorr == 'yes':
            '''Generation of the errormap'''
            sigma_dt = sigma/t_lag
            relErr, absErr = create_rerr_array(img_x, img_y, img_l, sigma_dt)
            count_sqrt = np.sqrt(count_fil)
            relErr = np.divide(relErr, count_sqrt, out=np.zeros_like(relErr), where=count_sqrt!=0)
            absErr = np.divide(absErr, count_sqrt, out=np.zeros_like(absErr), where=count_sqrt!=0)
            tif.imsave(resultpath + '/Info/RelErr/rel_error_cell' + str(i) + '.tif',relErr)
            tif.imsave(resultpath + '/Info/RelErr/abs_error_cell' + str(i) + '.tif',absErr)
        
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
        tif.imsave(resultpath+'/Info/Speed/Speed_Cell'+str(i)+'.tif', img_l)
        plt.savefig(resultpath+'QuiverPlotSpeed_Cell'+str(i)+'.png', dpi=300)
        plt.close(fig1)
        
        '''Generation of an speed heat map with the final unit of mum/s''' 
        plt.rcParams['image.cmap'] = 'RdPu' # changes current colorbar
        #mask array to assign pixels with entry 0 to white color
        img_l_mask = np.ma.array(img_l, mask=(img_l == 0))
        #assign zero to white color in current colormap
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        
        fig2 = plt.figure()
#        plt.axis('equal')
        plt.imshow(img_l,cmap='RdPu')
        plt.xticks([])
        plt.yticks([])
#        plt.show()
        cbar=plt.colorbar()
        cbar.set_label('µm/s')
        plt.savefig(resultpath+'HeatMap_Cell'+str(i)+'.png', dpi=300)
        plt.close(fig2)
        
        
        if errcorr == 'Yes' or errcorr == 'yes':
            '''Generation of the rel_err heat map'''
            relErr[relErr > 1] = 1 #apply thres for better resol in map
            
            plt.rcParams['image.cmap'] = 'viridis' # changes current colorbar
            #mask array to assign pixels with entry 0 to white color
            relErr_mask = np.ma.array(relErr, mask=(relErr == 0))
            #assign zero to white color in current colormap
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='white')
    
            fig3 = plt.figure()
            plt.imshow(relErr_mask)
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar()
            cbar.set_label('relative error')
            plt.savefig(resultpath + 'RelErrMap_Cell' + str(i) + '.png', dpi = 300)
            plt.close(fig3)
        
        
        maximum = img_l.max()
        img_l[img_l == 0] = np.nan
        average = np.nanmean(img_l)
        median = np.nanmedian(img_l)
        
        speed_info[i-1,:]=maximum, average, median
   
    if N > 1: #does not make sense to clc mean over just one value
        speed_info[-1,0] = np.mean(speed_info[0:-2,0])
        speed_info[-1,1] = np.mean(speed_info[0:-2,1])
        speed_info[-1,2] = np.mean(speed_info[0:-2,2])
    
    tif.imsave(resultpath+'/Info/Speed/Speed_info_all_bin'+str(binning)+'.tif',speed_info)

if __name__ == "__main__":
    N = 20 #number to be evaluated cells
    binning = 160 #binning to x nm
    sigma = 25 #nm
    t_lag = 10 #ms
    errcorr = 'Yes' # 'Yes' or 'No'
    resultpath = '/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Analysis_maps/Maps_160'
    path='/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Results_Par_testing/Trc'
    filename = '/Binning_'+str(binning)+\
    '/results_filtering/'
    #directory = '/home/mas32ea/Schreibtisch/Drift_and_Diffusion_Pad/TrackingVSGAtto/Trc/Trc_thresh2/Trc'+str(N)+'/'
    directed_motion(path,filename,resultpath,binning,sigma,t_lag,N,errcorr)