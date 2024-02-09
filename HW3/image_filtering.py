#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:59:19 2022

@author: indra25
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import warnings

#suppress warnings
warnings.filterwarnings('ignore')


#part-2
def load_gray_image():
    try:
        gray_img = cv2.imread('./Noisy_image.png',cv2.IMREAD_GRAYSCALE)
        return gray_img
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)       


def convolute(gray_img):
    try:
        
        scaling_factor = 1/9
        conv_filter_unscaled = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]
                                    ])
        conv_filter =  scaling_factor * conv_filter_unscaled
        conv_filter = conv_filter.astype('float32')
                
        gray_img = gray_img.astype('float')
        conv_img = np.zeros((gray_img.shape))
        temp = np.pad(gray_img, (1,1), 'constant')        
        for i in range(0,gray_img.shape[0]):
            for j in range(0,gray_img.shape[1]):
                local_img = temp[i:i+3,j:j+3]
                local_img = np.fliplr(local_img)
                local_img = np.flipud(local_img)
                conv_img[i,j] = np.sum(np.multiply(local_img,conv_filter))
                conv_img[i,j] = np.clip(conv_img[i,j], 0, 255)
        conv_img = conv_img.astype('uint8')
            
        cv2.imwrite('./convolved_image.png',conv_img)
        print('convolve image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def avg_filter(gray_img):
    try:
        
        scaling_factor = 1/9
        avg_filter_unscaled = np.array([[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]
                                    ])
        avg_filter =  scaling_factor * avg_filter_unscaled
        
                
        avg_img = np.zeros(gray_img.shape)
        temp = np.pad(gray_img, (1,1), 'constant')
        for i in range(0,gray_img.shape[0]):
            for j in range(0,gray_img.shape[1]):
                local_img = temp[i:i+3,j:j+3]
                avg_img[i,j] = np.mean(local_img)
        avg_img = np.clip(avg_img, 0, 255)
        avg_img = avg_img.astype('uint8')
                
        cv2.imwrite('./average_image.png',avg_img)
        print('avg image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def gaussian_filter(gray_img):
    try:
        
        scaling_factor = 1/16
        g_filter = np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]
                            ])
        g_filter =  scaling_factor * g_filter
        g_filter = g_filter.astype('float32')
        flipped_g_filter = np.flip(g_filter)
        
        gaussian_img = np.zeros((gray_img.shape))
        padded_gray_img = np.pad(gray_img, (1,1), 'constant')    
        height,width = gray_img.shape
        for i in range(height): #height
            for j in range(width): #width
                local_img = padded_gray_img[i:i+3,j:j+3]
                #print(flipped_g_filter.shape,local_img.shape)
                gaussian_img[i,j] = np.sum(np.multiply(flipped_g_filter,local_img))
                gaussian_img[i,j] = np.clip(gaussian_img[i,j], 0, 255)
        gaussian_img = gaussian_img.astype('uint8')
                
        cv2.imwrite('./gaussian_image.png',gaussian_img)
        print('gaussian image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def median_filter(gray_img):
    try:
        
        median_img = np.zeros(gray_img.shape)
        temp = np.pad(gray_img, (1,1), 'constant')
        for i in range(0,gray_img.shape[0]):
            for j in range(0,gray_img.shape[1]):
                local_img = temp[i:i+3,j:j+3]
                arr = local_img.flatten()
                arr = np.sort(arr)
                median_img[i,j] = arr[4]#np.median(arr)
        median_img = np.clip(median_img, 0, 255)
        median_img = median_img.astype('uint8')
                    
        cv2.imwrite('./median_img.png',median_img)
        print('median image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def increase_brightness_contrast():
    try:
        
        img_unexposed = cv2.imread('./Uexposed.png')
        img_unexposed = np.mean(img_unexposed,axis=2)
             
        brightness = 100
        contrast = 100
        bc = img_unexposed * (contrast/127+1) - contrast + brightness
        bc = np.clip(bc, 0, 255)
        bc = np.uint8(bc)
        
        cv2.imwrite('./adjusted_image.png',bc)
        print('adjusted image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        

    
   
def main():
    try:
        
        #Image Filtering
        gray_img = load_gray_image()
        
        #1.Convolution
        convolute(gray_img)
        
        #2.Averaging Filter
        avg_filter(gray_img)
        
        #3.Gaussian
        gaussian_filter(gray_img)
       
        #4.Median
        median_filter(gray_img)
        
        #5. Contrast and Brightness
        increase_brightness_contrast()
        
        
        
                                 
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno),e


main()