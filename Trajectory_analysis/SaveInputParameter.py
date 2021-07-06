#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:14:28 2021

@author: marieschwebs
"""
import numpy as np

def SaveInputShortTrAn(input_path, resultpath, time_lag, time_exp, scale,sigma,
                       errcorr, scal, threshold, kdim, kshape, kdim_smoothing,
                       wdth_outliers, mode_outliers):
    
    txt = open(resultpath +'/input_parameter.txt',"w+")
    txt.write('Input parameter of shortTrAn_loop njupyter notebook: \n\n')
    txt.write('Input path: '+ input_path + '\n')
    txt.write('Resultfolder: '+ resultpath + '\n') 
    txt.write('Time_lag: '+ str(time_lag) + ' ms \n')  
    txt.write('Time_exp: '+str(time_exp)+' ms \n')
    txt.write('Scale: '+ str(scale) +' nm \n\n')
    txt.write('Sigma: '+ str(sigma) +' nm \n\n')
    txt.write('Error correction: '+ errcorr +' \n\n')
    txt.write('Scale binning: '+ str(scal) +' \n\n')
    txt.write('Filter1 - Removal of entries less than threshold: \n Threshold: '+ str(threshold) +' \n\n')
    txt.write('Filter2 - Removal of isolated pixels: \n Kernel dimension: '+ str(kdim) +'\n Kernel shape: ' + kshape + ' \n\n')
    txt.write('Filter3 - smoothing out: \n Kernel dimension: '+ str(kdim_smoothing) + \
              '\n Width outlier for projection issue: ' + str(wdth_outliers) +  \
              '\n Mode projection issue: '+ mode_outliers + ' \n\n')
    txt.close()
    
    
if __name__ == "__main__":
    SaveInputShortTrAn(input_path, resultpath, time_lag, time_exp, scale,sigma,
                       errcorr, scal, threshold, kdim, kshape, kdim_smoothing,
                       wdth_outliers, mode_outliers)