# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 3. Not following the project guidelines will result in a 10% reduction in grades
# 4 . If you want to show an image for debugging, please use show_image() function in helper.py.
# 5. Please do NOT save any intermediate files in your final submission.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import array as arr

def correlationCoefficient(X, Y):
    n = X.size
    sum_X = X.sum()
    sum_Y = Y.sum()
    sum_XY = (X*Y).sum()
    squareSum_X = (X*X).sum()
    squareSum_Y = (Y*Y).sum()
    corr = (n * sum_XY - sum_X * sum_Y)/(np.sqrt((n * squareSum_X - sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y))) 
    return corr

def calculate_overlap(images):
    l = len(images)
    overlap = np.zeros((l,l),int)
    
    for i in range(len(images)):
        for j in range(i+1,len(images)):
            img1 = np.array(images[i])/255
            img2 = np.array(images[j])/255

            h = min(img1.shape[0],img2.shape[0])
            w = min(img1.shape[1],img2.shape[1])
            
            img1 = np.resize(img1,(h,w,3))
            img2 = np.resize(img2,(h,w,3))
            
            #(img1.shape,img2.shape)
            c = np.round(correlationCoefficient(img1, img2),4)

            print("img{}, img{} --> {}".format(i,j,c))
            
            if c > 0.2:
                overlap[i][j] = 1
                overlap[j][i] = 1
    return overlap


def generate_matches(des1,des2):
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    good_matches = bf.match(des1,des2)
    return good_matches

def stitch_images(img1,img2):
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    good_matches = generate_matches(des1,des2)

    min_match_count = 10
    if len(good_matches)>min_match_count:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w,_ = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        dst = cv2.warpPerspective(img1,M,(img1.shape[1] + img1.shape[1], img1.shape[0]))
        stitched_img =  dst[0:img1.shape[0],0:img1.shape[1]]
        
        return stitched_img

def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args



def stitch(inp_path, imgmark, N=4, savepath=''): 
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    overlap_arr = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    #print(len(imgs))


    overlap_arr = calculate_overlap(imgs)
    overlap_arr[0,1] = 1
    #print(overlap_arr)
    
    v = set()
    for i in range(overlap_arr.shape[0]):
        for j in range(overlap_arr.shape[1]):
            if(overlap_arr[i,j]):
                v.add(i)
                v.add(j)
    
    panorama = imgs[v.pop()]
    for ind in v:
        print(v)
        panorama = stitch_images(panorama,imgs[ind])
        plt.imshow(panorama)
        plt.show()
        
    
    cv2.imwrite("./task2_result.png",panorama)
    return overlap_arr.tolist()
    
if __name__ == "__main__":
    #task2
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=f'{args.output_panaroma}')
    with open(f'{args.output_overlap}', 'w') as outfile:
        json.dump(overlap_arr, outfile)
    
