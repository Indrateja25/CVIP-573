#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:59:19 2022

@author: indra25
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

#part-1
def load_image():
    try:
        img_png = cv2.imread('./Lenna.png',)
        img = cv2.cvtColor(img_png, cv2.COLOR_BGR2RGB)
        return img
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)       

def RGB_2_HSV(img_unscaled):
    try:
        img = img_unscaled/255
        img_hsv = img.copy()
        
        H = np.zeros((img.shape[0],img.shape[1]))
        S = np.zeros((img.shape[0],img.shape[1]))
        V = np.zeros((img.shape[0],img.shape[1]))
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                #V
                V[i,j] = np.max(img[i,j])
                
                #S
                S[i,j] = (V[i,j]-np.min(img[i,j]))/(V[i,j]) if V[i,j] != 0 else 0
    
                #H
                index = np.argmax(img[i,j])
                R = img[i,j][0]
                G = img[i,j][1]
                B = img[i,j][2]
                if index == 0:
                    H[i,j] = (60*(G-B))/(V[i,j]-min(R,G,B))
                elif index == 1:
                    H[i,j] = 120+(((60*(B-R)))/(V[i,j]-min(R,G,B)))
                elif index == 2:
                    H[i,j] = 240+(((60*(R-G)))/(V[i,j]-min(R,G,B)))
                elif index == 0 & index == 1 & index == 2:
                    H[i,j] = 0
                    
                if H[i,j] < 0:
                    H[i,j] = H[i,j]+360
                    
        print("H-Range:",H.min(),H.max())
        print("S-Range:",S.min(),S.max())
        print("V-Range:",V.min(),V.max())
    
        V = V*255
        S = S*255
        H = H/2
        
        img_hsv[:,:,0] = H
        img_hsv[:,:,1] = S
        img_hsv[:,:,2] = V
        
        img_hsv = np.round(img_hsv)
        img_hsv = img_hsv.astype('uint8')
        cv2.imwrite('./hsv_image_1.png',cv2.cvtColor(img_hsv, cv2.COLOR_RGB2BGR))
        
        print('hsv-1 image loaded',"\n\n")
        
        return img_hsv
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        

 
def RGB_2_HSV_2(img):
    try:
        img_hsv = img.copy()
        
        H = np.zeros((img.shape[0],img.shape[1]))
        S = np.zeros((img.shape[0],img.shape[1]))
        I = np.zeros((img.shape[0],img.shape[1]))
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                R = img[i,j][0]
                G = img[i,j][1]
                B = img[i,j][2]
                
                #I
                I[i,j] = (R+G+B)/3
                
                #S
                S[i,j] = 1 - (min(R,G,B)/I[i,j])
                
                #H
                num = np.round(0.5*((R-G)+(R-B)),3)
                denom = np.round(np.sqrt((R-G)*(R-G) + (R-B)*(G-B)),3)
                #print(num,denom)
                
                theta = np.arccos(num/denom)
                
                if B <= G:
                    H[i,j] = theta
                else:
                    H[i,j] = 360-theta
        
        img_hsv[:,:,0] = H
        img_hsv[:,:,1] = S
        img_hsv[:,:,2] = I
        
        img_hsv = np.round(img_hsv)
        img_hsv = img_hsv.astype('uint8')
        return img_hsv
        cv2.imwrite('./hsv_image_2.png' ,cv2.cvtColor(img_hsv, cv2.COLOR_RGB2BGR))
        
        print('hsv-2 image loaded',"\n\n")
        
        return img_hsv
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        

 
def RGB_2_CMYK(img):
    try:
        img_cmyk = np.zeros((img.shape[0],img.shape[1],4))
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
        
        C = 1 - R
        M = 1 - G
        Y = 1 - B
        K = np.zeros((img.shape[0],img.shape[1]))
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                c = C[i,j]
                m = M[i,j]
                y = Y[i,j]
                
                K[i,j] = min(c,m,y)
                C[i,j] = (c - K[i,j])/(1 - K[i,j])
                M[i,j] = (m - K[i,j])/(1 - K[i,j])
                Y[i,j] = (y - K[i,j])/(1 - K[i,j])
                
        img_cmyk[:,:,0] = C
        img_cmyk[:,:,1] = M
        img_cmyk[:,:,2] = Y
        img_cmyk[:,:,3] = K
        
        img_cmyk = np.round(img_cmyk)
        img_cmyk = img_cmyk.astype('uint8')
        
        cv2.imwrite('./cmyk_image.png',cv2.cvtColor(img_cmyk, cv2.COLOR_RGB2BGR))
        print('cmyk image loaded',"\n\n")
        
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def RGB_2_LAB(img):
    try:
       
        img_LAB = np.zeros((img.shape))
        m = np.array([[0.412453, 0.35758, 0.180423],
                          [0.212671, 0.71516, 0.072169],
                          [0.019334, 0.119193, 0.950227]])

        X_n = 0.950456
        Z_n = 1.088754
        delta = 128

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                R = img[i,j,0]
                G = img[i,j,1]
                B = img[i,j,2]

                RGB = np.array([R,G,B])
                XYZ = np.matmul(m, RGB)

                X = XYZ[0]
                Y = XYZ[1]
                Z = XYZ[2]

                X = X/X_n
                Z = Z/Z_n

                L = 0
                if Y > 0.008856:
                    L = (116 * np.power(Y,1/3)) - 16
                else:
                    L = 903.3 * Y

                f_X = 0
                if X > 0.008856:
                    f_X = np.power(X,1/3)
                else:
                    f_X = 7.787*X + (16/116)

                f_Y = 0
                if Y > 0.008856:
                    f_Y = np.power(Y,1/3)
                else:
                    f_Y = 7.787*Y + (16/116)

                f_Z = 0
                if Z > 0.008856:
                    f_Z = np.power(Z,1/3)
                else:
                    f_Z = 7.787*Z + (16/116)


                a = 500 * (f_X - f_Y) + delta
                b = 200 * (f_Y - f_Z) + delta


                L = (L*255)/100
                a = a + 128
                b = b + 128

                img_LAB[i,j,0] = L
                img_LAB[i,j,1] = a
                img_LAB[i,j,2] = b

        img_LAB = np.round(img_LAB)
        img_LAB = img_LAB.astype('uint8')


        cv2.imwrite('./lab_image.png',cv2.cvtColor(img_LAB, cv2.COLOR_RGB2BGR))
        print('lab image loaded',"\n\n")

    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        


def main():
    try:
        
    #Color-Conversion
        img = load_image()
        
        #1.RGB-2-HSV
        RGB_2_HSV(img)
        
        #2.RGB-2-HSV-2
        RGB_2_HSV_2(img)
        
        #3.RGB-2-CMYK
        RGB_2_CMYK(img)
       
        #4.RGB-2-LAB
        RGB_2_LAB(img)
                             
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno),e


main()