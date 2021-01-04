#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:07:22 2019

@author: mas32ea
"""
import numpy as np
import scipy.io as sio
import tifffile as tif
"""loads input file and sets the scale to 1nm"""

def loadmat_file(path,scale):  
    traces = sio.loadmat(path)
    for key in traces:
        1+1
    tracs = traces[key]
    
    tracs = np.array(tracs)
    a = tracs.shape[0] #number of rows = total entries of poistions
    b = tracs.shape[1] #number of columns, used to extract later on to extract x and y positions 
    """ Convertion to 1 nm scaling """
    tracs[0:a, 2:4] = tracs[0:a, 2:4]*scale # scales to nm

    #tracs_unint16 = np.uint16(tracs) #reduction of datatype
    tracs_unint16 = np.int32(tracs)
    
    trac_num, counts = np.unique(tracs_unint16[0:a,0],return_counts = True) #counts=trace number, #counts = length of each trajectory

    tracs_unint16corr = tracs_unint16 
    trac_numcorr = trac_num
    dt = np.amax(tracs[0:a,1]) #extraction of the total frame number
    return a, b, tracs,tracs_unint16, trac_num, counts, tracs_unint16corr, trac_numcorr, dt 

#    '''Needed to read in only one trace which was extracted by python'''
#    tracs = np.load(path)
#    
#    tracs = np.array(tracs)
#    a = tracs.shape[0]
#    b = tracs.shape[1]
#    """ nm Skalierung deswegen 160 """
#    tracs[0:a, 2:4] = tracs[0:a, 2:4]*160
#    
#    tracs_unint16 = np.uint16(tracs)
#    
#    trac_num, counts = np.unique(tracs_unint16[0:a,0],return_counts = True)
#    
#    tracs_unint16corr = tracs_unint16
#    trac_numcorr = trac_num
#    dt = np.amax(tracs[0:a,1])
#    return a, b, tracs,tracs_unint16, trac_num, counts, tracs_unint16corr, trac_numcorr, dt

if __name__ == "__main__":
    N=1
    path = ('/Users/marieschwebs/Desktop/TrackingVSGAtto/Trc/Input_data/trc'+str(N))
    a,b,tracs,tracs_unint16,trac_num,counts,tracs_unint16corr,trac_numcorr,dt = loadmat_file(path,scale)