#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:31:19 2019

@author: mas32ea
"""
import numpy as np
"""Prefabricates fields which are defined by shape of the matrices specified by the min and max entries of x and y respectively"""

def create_fields(shape):
    pointfield = np.zeros(shape,'int8')
    vectorfield_x= np.zeros(shape,'float')
    vectorfield_y= np.zeros(shape,'float')
    tensorfield_xx = np.zeros(shape,'float')
    tensorfield_xy = np.zeros(shape,'float')
    tensorfield_yx = np.zeros(shape,'float')
    tensorfield_yy = np.zeros(shape,'float')
    
    return pointfield,vectorfield_x,vectorfield_y,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy


if __name__ == "__main__":
    pointfield,vectorfield_x,vectorfield_y,tensorfield_xx,tensorfield_xy,tensorfield_yx,tensorfield_yy=create_fields(shape)