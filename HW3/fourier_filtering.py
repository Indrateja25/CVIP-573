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


#part-3
def load_gray_image():
    try:
        gray_img = cv2.imread('./Noisy_image.png',cv2.IMREAD_GRAYSCALE)
        return gray_img
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)       

def fourier_transform(img):
    try:
        #convert to float-32
        float_img = np.float32(img)
        scaler = 20
        
        #calcualte dft
        dft = cv2.dft(float_img, flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)
        img_fourier = scaler*np.log(cv2.magnitude(dft_shifted[:,:,0],dft_shifted[:,:,1]))

        cv2.imwrite('./converted_fourier.png',img_fourier)
        print('fourier spectrum image loaded',"\n\n")
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def low_pass_smooth(img):
    try:
        #convert to float-32
        float_img = np.float32(img)

        #calcualte dft
        dft = cv2.dft(float_img, flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        #mask size
        rows, cols = img.shape
        row, col = int(rows/2) , int(cols/2)
        
        #mask
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[row-30:row+30, col-30:col+30] = 1

        #apply mask
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        
        #img_back = np.clip(img_back, 0, 255)
        #img_back = img_back.astype('uint8')
        cv2.imwrite('./gaussian_fourier.png',img_back)
        print('fourier image loaded',"\n\n")
   
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        



   
def main():
    try:

        #Fourier Transformation
        gray_img = load_gray_image()
        
        #1.Fourier filter
        fourier_transform(gray_img)
        
        #gaussian smooth
        low_pass_smooth(gray_img)
                                 
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno),e


main()