import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os

def load_image():
    try:
        img_filtered = cv2.imread('./image.png',)        
        img = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
        print(img.shape)
    
        #plt.title('Image')
        #plt.imshow(img)
        print('image loaded',"\n\n")
        
        return img
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)       

def convert_to_gray_image(img):
    try:
        gray = np.mean(img,axis=2)
        print(gray.shape)
        
        #plt.title('gray image')
        #plt.imshow(gray,cmap='gray')
        cv2.imwrite('./gray_image.png',gray)
        print('gray image loaded',"\n\n")
        
        return gray
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)        

def scale_gray_image(gray):
    try:
        
        width = gray.shape[1]
        #print(width)
        half_width = int(width/2)
        #print(half_width)
        
        gray_half_width = gray[:,:half_width]
        #print(gray_half_width.shape)
        #plt.imshow(gray_half_width,cmap='gray')
        
        height = gray.shape[0]
        #print(height)
        half_height = int(height/2)
        #print(half_height)
        
        gray_half_height = gray[:half_height,:]
        #print(gray_half_height.shape)
        #plt.imshow(gray_half_height,cmap='gray')
        
        gray_scaled = gray[:half_height,:half_width]
        print(gray_scaled.shape)
        
        #plt.title('gray image scaled')
        #plt.imshow(gray_scaled,cmap='gray')
        cv2.imwrite('./gray_image_scaled.png',gray_scaled)
        print('gray image scaled loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)    
    
def translate_gray_image(gray):
    try:
        right_translate = 50
        gray_right_translate = gray[:,:-right_translate]
        #plt.imshow(gray_right_translate,cmap='gray')
        #print(gray_right_translate.shape)
        
        black_space_left = np.zeros((gray.shape[0],50))
        black_space_left.shape
        
        gray_right = np.concatenate((black_space_left,gray_right_translate),axis=1)
        #plt.imshow(gray_right,cmap='gray')
        #print(gray_right.shape)
        
        bottom_translate = 50
        gray_bottom_translate = gray_right[:-bottom_translate,:]
        #plt.imshow(gray_bottom_translate,cmap='gray')
        #print(gray_bottom_translate.shape)
        
        black_space_top = np.zeros((50,gray.shape[1]))
        black_space_top.shape
        
        gray_translate = np.concatenate((black_space_top,gray_bottom_translate),axis=0)
        print(gray_translate.shape)

        #plt.title('gray image translated')
        #plt.imshow(gray_translate,cmap='gray')
        cv2.imwrite('./gray_image_translated.png',gray_translate)
        print('gray image translated loaded',"\n\n")
        
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)

def flip_gray_image_horizontal(gray):
    try:
        gray_flipped_hor = np.flip(gray,axis=1)
        print(gray_flipped_hor.shape)
        
        #plt.title('gray image flipped horizontal')
        #plt.imshow(gray_flipped_hor,cmap='gray')
        cv2.imwrite('./gray_image_flip_horizontal.png',gray_flipped_hor)
        print('gray image flipped horizontal loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        
def flip_gray_image_vertical(gray):
    try:
        gray_flipped_ver = np.flip(gray,axis=0)
        print(gray_flipped_ver.shape)
        
        #plt.title('gray image flipped vertical')
        #plt.imshow(gray_flipped_ver,cmap='gray')
        cv2.imwrite('./gray_image_flip_vertical.png',gray_flipped_ver)
        print('gray image flipped vertical loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)

def invert_gray_image(gray):
    try:
        inverse_value = 255
        gray_inverse = inverse_value - gray
        print(gray_inverse.shape)     
        
        #plt.title('gray image inverted loaded')
        #plt.imshow(gray_inverse,cmap='gray')
        cv2.imwrite('./gray_image_inversion.png',gray_inverse)
        print('gray image inverted loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)

def rotate_gray_image(gray):
    try:
        angle = -45 #clock-wise rotation
        angle = np.radians(angle) #convert degrees to radians
        cos = np.cos(angle) #get cosine value
        sin = np.sin(angle) #get sine value
         
        height = gray.shape[0] #get height
        width = gray.shape[1] #get width
        origin_height = np.round(gray.shape[0]/2).astype(int) #get origin height coordinate wrt to image height
        origin_width  = np.round(gray.shape[1]/2).astype(int) #get origin width coordinate wrt to image width

        gray_rotated = np.zeros((height,width)) #generate ouput image matrix
        
        #for each pixel/cell in image:
        for i in range(height):
            for j in range(width):
                
                y = gray.shape[0]-i-origin_height #find y coordinate for this pixel wrt to origin          
                x = gray.shape[1]-j-origin_width  #find x coordinate for this pixel wrt to origin
    
                #make the rotation using rotation matrix
                rotated_y = np.round(-x*sin+y*cos).astype(int) 
                rotated_x =np.round(x*cos+y*sin).astype(int)
                
                #get matrix indices from coordinates
                rotated_y = origin_height-rotated_y
                rotated_x = origin_width-rotated_x
    
                #make sure no index is going out of range of the output natrix height and width
                if 0 <= rotated_x < width and 0 <= rotated_y < height and rotated_x>=0 and rotated_y>=0:
                    gray_rotated[rotated_y,rotated_x] = gray[i,j]  #write the original pixel value in rotated new cell
            
        print(gray_rotated.shape)
        plt.title('gray image rotated')
        plt.imshow(gray_rotated,cmap='gray')
        cv2.imwrite('./gray_image_rotated.png',gray_rotated)
        print('gray image rotated loaded',"\n\n")
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno, e)
 

#bonus

def scale_image(img):
    try:
        
        width = img.shape[1]
        #print(width)
        half_width = int(width/2)
        #print(half_width)
        
        img_half_width = img[:,:half_width,:]
        #print(img_half_width.shape)
        #plt.imshow(img_half_width,)
        
        height = img.shape[0]
        #print(height)
        half_height = int(height/2)
        #print(half_height)
        
        img_half_height = img[:half_height,:,:]
        #print(img_half_height.shape)
        #plt.imshow(img_half_height,)
        
        img_scaled = img[:half_height,:half_width,:]
        print(img_scaled.shape)
        
        #plt.title('image scaled')
        #plt.imshow(img_scaled)
        cv2.imwrite('./image_scaled.png',cv2.cvtColor(img_scaled, cv2.COLOR_RGB2BGR))
        print('image scaled loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)    
    
def translate_image(img):
    try:
        
        right_translate = 50
        img_right_translate = img[:,:-right_translate,:]
        #plt.imshow(img_right_translate)
        #print(img_right_translate.shape)
        
        
        black_space_left = np.zeros((img.shape[0],50,img.shape[2]))
        #print(black_space_left.shape)
        
        img_right = np.concatenate((black_space_left,img_right_translate),axis=1).astype(np.uint8)
        #plt.imshow(img_right,)
        #print(img_right.shape)
        
        
        bottom_translate = 50
        img_bottom_translate = img_right[:-bottom_translate,:]
        #plt.imshow(img_bottom_translate)
        #print(img_bottom_translate.shape)
        
        black_space_top = np.zeros((50,img.shape[1],img.shape[2]))
        black_space_top.shape
        
        img_translate = np.concatenate((black_space_top,img_bottom_translate),axis=0).astype(np.uint8)
        print(img_translate.shape)
                
        #plt.title('image translated')
        #plt.imshow(img_translate)
        cv2.imwrite('./image_translated.png',cv2.cvtColor(img_translate, cv2.COLOR_RGB2BGR))
        print('image translated loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)

def flip_image_horizontal(img):
    try:
        img_flipped_hor = np.flip(img,axis=1)
        print(img_flipped_hor.shape)
        
        #plt.title('image flipped horizontal')
        #plt.imshow(img_flipped_hor)
        cv2.imwrite('./image_flip_horizontal.png',cv2.cvtColor(img_flipped_hor, cv2.COLOR_RGB2BGR))
        print('image flipped horizontal loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
def flip_image_vertical(img):
    try:
        img_flipped_ver = np.flip(img,axis=0)
        print(img_flipped_ver.shape)
        
        #plt.title('image flipped vertical')
        #plt.imshow(img_flipped_ver)
        cv2.imwrite('./image_flip_vertical.png',cv2.cvtColor(img_flipped_ver, cv2.COLOR_RGB2BGR))
        print('image flipped vertical loaded',"\n\n")
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    
def main():
    try:
        img = load_image()
        
        #gray-scale & it's transformations
        gray = convert_to_gray_image(img)
        scale_gray_image(gray)
        translate_gray_image(gray)
        flip_gray_image_horizontal(gray)
        flip_gray_image_vertical(gray)
        invert_gray_image(gray)
        rotate_gray_image(gray)
        
        #bonus
        scale_image(img)
        translate_image(img)
        flip_image_horizontal(img)
        flip_image_vertical(img)
                                 
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno),e


main()