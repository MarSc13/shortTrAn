#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:27:17 2019

@author: mas32ea
"""
import numpy as np
"""deletes traces containing only one position"""
def remove_single_part(trac_num,counts,a,tracs_unint16corr,trac_numcorr):
    for t in range(trac_num.shape[0]):
        if counts[t] == 1:
            ind = np.where(tracs_unint16corr[0:a,0] == trac_num[t])[0][0]
            ind2 = np.where(trac_numcorr == trac_num[t])[0][0]
            tracs_unint16corr = np.delete(tracs_unint16corr,(ind),axis=0)
            trac_numcorr = np.delete(trac_numcorr,(ind2),axis=0)

    threshold = 5
    
    #if some trc are deleted because they where just loc then trac_nm and counts 
    #need to be calc again
    trac_num, counts = np.unique(tracs_unint16corr[0:a,0],return_counts = True)
    
    binary_short = np.zeros(counts.shape[0]) #create array 
    binary_short[counts <= threshold] = 1 #counts shorter than thres will be 1 
    #trc num longer than thres will be set to 0
    short_tracs = np.multiply(binary_short, trac_numcorr)
    short_tracs_red = short_tracs.copy()
    #removal of zeros
    short_tracs_red = short_tracs_red[short_tracs_red !=0]
    
    binary_rem = np.ones(counts.shape[0]) #create array 
    binary_rem[counts <= threshold] = 0 #counts shorter than thres will be 0
    trac_numcorr = np.multiply(binary_rem, trac_numcorr)
    trac_numcorr = trac_numcorr[trac_numcorr !=0]
    
    for i in reversed(range(short_tracs_red.shape[0])): # reversed iteration over trc num
        #extraction of indices equal to the trc num
        ind_short = np.where(tracs_unint16corr[0:a,0] == short_tracs_red[i])
        #deletion of trc
        tracs_unint16corr = np.delete(tracs_unint16corr,(ind_short),axis=0)
        
    return  tracs_unint16corr, trac_numcorr

if __name__ == "__main__":
    tracs_unint16corr, trac_numcorr = remove_single_part(trac_num,counts,a,tracs_unint16corr,trac_numcorr) 