#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:20:05 2021

@author: marieschwebs
"""

import numpy as np
import tifffile as tif 
import matplotlib.pyplot as plt
from matplotlib import colors
import os as os

plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


'''boxkernel for fulling of the surroundings in the supermap'''
def FillingSurrounding(sprmp,n_sim_sqrs):
    
    supermap = sprmp.copy()
    #padding enables box kernel to move over matrix
    pad_sprmp = np.zeros((supermap.shape[0]+2,supermap.shape[1]+2))
    pad_sprmp[:,:] = 0.5 #background is assigned to the value 0.5
    
    #placing of the supermap in the predefined matrix for padding
    pad_sprmp[1:-1,1:-1] = supermap
    
    iteration = [1]
    
    while len(iteration) != 0:  
    #iteration of the box kernel with dimension 3 over the padded supermap
        iteration = []
        for a in range(1,pad_sprmp.shape[0]-1):
            for b in range(1,pad_sprmp.shape[1]-1):
                kernel = np.zeros((3,3))
                kernel = pad_sprmp[a-1:a+2,b-1:b+2]
                
                #if center position is undiceded (=0) and majority of surrounding squares are assigned to diffusion (=1)
                if kernel[1,1] == 0 and np.sum(kernel == 1) >= n_sim_sqrs:
                    supermap[a-1,b-1] = 1
                    iteration = np.append(iteration,[1])
                #if center position is undiceded (=0) and majority of surrounding squares are assigned to directed motion (=2)
                elif kernel[1,1] == 0 and np.sum(kernel == 2) >= n_sim_sqrs:
                    supermap[a-1,b-1] = 2
                    iteration = np.append(iteration,[1])
                #if kernel is placed at the rim of the outline
                    
            
        pad_sprmp[1:-1,1:-1] = supermap
        
    return supermap


'''main function supermap'''
'''descirption of the funcion will follow here soon'''
def Supermap(path_results,fldrnm_maps,path_ecc_lut,path_err_lut,resultpath,stat_thres,diff,n_sim_sqrs,strict,N):
    
    '''makes result folder'''
    if not os.path.exists(resultpath+'/Diff'+str(diff)+'/'):
       #print('Path does not exist')
       os.mkdir(resultpath+'/Diff'+str(diff))
       os.mkdir(resultpath+'/Diff'+str(diff)+'/SupermapBeforeFill')
       os.mkdir(resultpath+'/Diff'+str(diff)+'/PlotUndecided_BeforeFill')
       os.mkdir(resultpath+'/Diff'+str(diff)+'/QuotientsEccErr')
       resultpath = resultpath+'/Diff'+str(diff)
    else:
       resultpath = resultpath+'/Diff'+str(diff)
    
    
    #loading look uo tables
    ecc_lut = tif.imread(path_ecc_lut)
    err_lut = tif.imread(path_err_lut)
    
    
    for i in range(1,N+1): 
        
        #tryp outline
        count = tif.imread(path_results + '/Trc' + str(i) + '/results_filtering/scld_count_fil.tif')
        count = count.astype(int)
        
        mask = np.zeros((count.shape))
        mask[count != 0] = 1
        
        #diffusion information(eccentricity)
        ecc_diff = tif.imread(path_results + '/' + fldrnm_maps + \
                            '/DiffusionMaps/Info/Ellipticity/ecc_cell_' \
                            + str(i) + '.tif')
        ecc_diff = np.multiply(mask,ecc_diff) 
            
        #directed motion information(rel_err)
        rel_err = tif.imread(path_results + '/' + fldrnm_maps + \
                      '/DirectedMotionMaps/Info/RelErr/rel_error_cell' +  str(i) + '.tif')
        rel_err = np.multiply(mask, rel_err)           
        
        
        motion = np.zeros(count.shape)
        d_err_all = np.zeros(count.shape)
        d_ecc_all = np.zeros(count.shape)
        
        # criteria for diffusion
        for a in range(count.shape[0]): #ranges through rows
            for b in range(count.shape[1]): #ranges through columns
                if count[a,b] > stat_thres: #!= 0:
                    
                    #count value specific thresholds 
                    if count[a,b] < len(ecc_lut):
                        thres_ecc = ecc_lut[count[a,b]-1,1]
                        thres_err = err_lut[count[a,b]-1,1]
                    else:
                        thres_ecc = ecc_lut[ecc_lut.shape[0]-1,1]
                        thres_err = err_lut[err_lut.shape[0]-1,1]
                    
                    #extraction of delta ecc and relErr
                    delta_err = np.divide(rel_err[a,b],thres_err)-1
                    delta_ecc = np.divide(ecc_diff[a,b],thres_ecc)-1
                    d_err_all[a,b] = delta_err +1
                    d_ecc_all[a,b] = delta_ecc +1
                    
                    #assignment motion model
                    if ecc_diff[a,b] < thres_ecc and rel_err[a,b] > thres_err:
                        motion[a,b] = motion[a,b] + 1   
                    elif ecc_diff[a,b] > thres_ecc and rel_err[a,b] < thres_err:
                        motion[a,b] = motion[a,b] + 2
                    elif strict != 1:
                        if abs(np.subtract(delta_err,delta_ecc)) > diff and delta_err > delta_ecc:
                            if rel_err[a,b] > thres_err:
                                motion[a,b] = motion[a,b] + 1
                            elif rel_err[a,b] < thres_err:
                                motion[a,b] = motion[a,b] + 2
                        elif abs(np.subtract(delta_err,delta_ecc)) > diff and delta_err < delta_ecc:
                            if ecc_diff[a,b] < thres_ecc:
                                motion[a,b] = motion[a,b] + 1  
                            elif ecc_diff[a,b] > thres_ecc:
                                motion[a,b] = motion[a,b] + 2  
        
        
        
        
        #tif.imsave(resultpath + '/QuotientsEccErr/d_err_all.tif',d_err_all)
        #tif.imsave(resultpath + '/QuotientsEccErr/d_ecc_all.tif',d_ecc_all)
        
       
        #making the outline of the tryp visible
        motion[count == 0] = 0.5
        
        
        
        #generation of linearied matrices for plot of ecc and relErr over count value
        und_sqrs = np.zeros(motion.shape)
        und_sqrs[motion == 0] = 1 
        
        # extraction quoutienten of undiceded super-pixel
        und_d_err = np.multiply(und_sqrs,d_err_all)
        und_d_ecc = np.multiply(und_sqrs,d_ecc_all)
        tif.imsave(resultpath + '/QuotientsEccErr/d_err_undecided_cell'+str(i)+'.tif',und_d_err)
        tif.imsave(resultpath + '/QuotientsEccErr/d_ecc_undecided_cell'+str(i)+'.tif',und_d_ecc)
        
        #
        und_ecc = np.multiply(und_sqrs,ecc_diff)
        und_count = np.multiply(und_sqrs,count)
        und_err = np.multiply(und_sqrs,rel_err)
        
        
        #linearisation of count matrix for scatter plot
        und_count = np.ravel(und_count)
        #deletion of all zero entries
        und_mask = und_count.copy()
        und_count = und_count[und_count != 0]
        #cut to a count value of 242, otherwise count values would differ from countvalues of LUT
        und_count[und_count > len(ecc_lut)] = len(ecc_lut) 
        
        #linearisation of ecc matrix for scatter plot
        und_ecc = np.ravel(und_ecc)
        #deletion of all zero entries
        und_ecc = und_ecc[und_mask != 0]
        
        #linearisation of relErr matrix for scatter plot
        und_err = np.ravel(und_err)
        #deletion of all zero entries
        und_err = und_err[und_mask != 0]
        
        
        
        '''figure of undiceded squares'''
        fig1, ax1 = plt.subplots()
        ax1.plot(ecc_lut[:,0],ecc_lut[:,1],'m')
        ax1.scatter(und_count[:],und_ecc[:],s=3,color='m')
        ax1.set_xlim([0,len(ecc_lut)])
        ax1.set_ylim([0,1])
        ax1.tick_params(axis = 'y',labelcolor = 'm')
        ax1.set_xlabel('count value')
        ax1.set_ylabel('eccentricity',color = 'm')
        
        ax2 = ax1.twinx()
        
        ax2.plot(err_lut[:,0],err_lut[:,1],'b')
        ax2.scatter(und_count[:],und_err[:],s=3,color='b')
        ax2.set_ylim([0,2])
        ax2.tick_params(axis = 'y', labelcolor = 'b')
        ax2.set_ylabel('relative standard error',color = 'b')
        
        plt.title('Undecided squares diff < ' + str(diff))
        fig1.tight_layout()
        
        plt.savefig(resultpath + '/PlotUndecided_BeforeFill/Plot_undecided_Cell' +str(i)+ '.pdf',dpi=300)
        plt.savefig(resultpath + '/PlotUndecided_BeforeFill/Plot_undecided_Cell' +str(i)+ '.png',dpi=300)
        plt.close(fig1)
        
        
        tif.imsave(resultpath + '/SupermapBeforeFill/SupermapMatrix_Cell' +str(i)+ '.tif',motion)
        #box kernel for filling up the surroundings
        filled_supermap = FillingSurrounding(motion,n_sim_sqrs)
        tif.imsave(resultpath + '/SupermapMatrix_Cell' +str(i)+ '.tif',filled_supermap)
        
        
        #generation of the colorcode
        #case if no directed motion is present
        if np.max(motion) == 1: # green diffusion, magenta directed motion
            #def colors heatmap
            colormap = colors.ListedColormap(['#DCDCDC', 'white', '#009999'])
            bounds=[0,0.5,1,2]
            norm = colors.BoundaryNorm(bounds, colormap.N)
        else:#directed motion is present
            #def colors heatmap
            colormap = colors.ListedColormap(['#DCDCDC', 'white', '#009999','#990099'])
            bounds=[0,0.5,1,2]
            norm = colors.BoundaryNorm(bounds, colormap.N)
        
        
        
        fig1 = plt.figure()
        plt.imshow(motion,cmap=colormap)
        plt.xticks([])
        plt.yticks([])
        plt.title('Supermap Cell ' + str(i))
        plt.savefig(resultpath + '/SupermapBeforeFill/Supermap_Cell' +str(i)+ '.pdf',dpi=300)
        plt.savefig(resultpath + '/SupermapBeforeFill/Supermap_Cell' +str(i)+ '.png',dpi=300)
        fig1.tight_layout()
        plt.close(fig1)
        
        fig2 = plt.figure()
        plt.imshow(filled_supermap,cmap=colormap)
        plt.xticks([])
        plt.yticks([])
        plt.title('Supermap Cell ' + str(i))
        plt.savefig(resultpath + '/Supermap_Cell' +str(i)+ '.pdf',dpi=300)
        plt.savefig(resultpath + '/Supermap_Cell' +str(i)+ '.png',dpi=300)
        plt.close(fig2)
        
if __name__ == "__main__":                       
    
    path_results = '/Volumes/Vin/Python/cells/shortTrAn/results'
    fldrnm_maps = 'Maps_errCorr'
    path_ecc_lut = '/Volumes/Vin/shortTrAn/Supermaps/LUTs/ecc_thres99_fit.tif'
    path_err_lut = '/Volumes/Vin/shortTrAn/Supermaps/LUTs/relErr_thres01_fit.tif'
    resultpath = '/Volumes/Vin/Python/cells/shortTrAn/results/Supermaps_0199'
    stat_thres = 7 #count > stat_thres
    diff = 0.25
    n_sim_sqrs = 7
    strict = 0 #yes=1, no=0 
    N = 9
    
    Supermap(path_results,fldrnm_maps,path_ecc_lut,path_err_lut,resultpath,stat_thres,diff,n_sim_sqrs,strict,N)
        
        