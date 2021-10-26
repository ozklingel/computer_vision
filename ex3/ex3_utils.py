
import time
import numpy as np
from matplotlib import pyplot as plt

import cv2


# This sets the program to ignore a divide error which does not affect the outcome of the program




# -------------------- Generate Gaussian Pyramids ----------------------- #

def gauss_pyramid(image,lev):  # function (input image)

    G_temp = image.copy()  # temp image = original image

                           # G_ temp is used to temporary record gaussian pyramid

    gauss_pymid = [G_temp] # gauss_pymid[0] = original image



    ## generate gaussian pyramid by using opencv function "cv2.pyrDown"

    for i in range(lev):

        G_temp = cv2.pyrDown(gauss_pymid[i]) # generate gaussian pyramid

        gauss_pymid.append(G_temp)           # append G_temp into gauss_pymid



    return gauss_pymid # return gaussian pyramid set



# -------------------- Generate Laplacian Pyramids ----------------------- #

def laplacian_pyramid(gauss_back,lev): # function (input image)

    lap_pymid = [gauss_back[lev-1]]    # lap_pymid[0] = gauss_back[5]



    ## generate laplacian pyramid by use gaussin pyramid and function "cv2.pyrUp"

    for i in range(lev-1,0,-1):

        size = (gauss_back[i-1].shape[1], gauss_back[i-1].shape[0]) # get the next image size for the pyramid

        L_temp = cv2.pyrUp(gauss_back[i], dstsize = size)           # generate the first step for laplacian pyramid

                                                                    # (first step: expand gaussian pyramid)

        L_subtract = cv2.subtract(gauss_back[i-1],L_temp)           # do subtraction to generate laplacian pyramid

        lap_pymid.append(L_subtract)                                # append L_temp into lap_pymid



    return lap_pymid # return laplacian pyramid set



# -------------------- Generate reconstruct image ----------------------- #

# combine background laplacian pyramid and target laplacian pyramid with a mask



def combine_lp_mask(gauss_m, lap_back, lap_tar,lev): # function (mask gaussian pyramid,

                                                 # background laplacian pyramid, target laplacian pyramid)

    combined_result = [] # create an empty object for record the combined pyramid image



    ## combined laplacian image with the mask

    for i in range(lev):

        x, y = gauss_m[lev-1-i].shape  # get the image size, raw and column value

        combined_img = lap_back[i] # let combined image  = background laplacin image



        # if mask pixel value < 125 remain the background image, else fill in the target image

        for count_x in range (0, x) :

            for count_y in range (0, y):

                if gauss_m[lev-1-i][count_x, count_y] > 125: # if mask pixel value > 125,

                                                         # combined image pixel value = target pixel value

                    combined_img[count_x, count_y] = lap_tar[i][count_x, count_y]



        combined_result.append(combined_img) # append combined image into combined_result set



    ## reconstruct the image

    recon_img = combined_result[0] # recon_img = laplacin combined image [0]

    # start reconstruct the image

    for i in range(1,lev):

        size = (combined_result[i].shape[1], combined_result[i].shape[0]) # get image size

        recon_img = cv2.pyrUp(recon_img, dstsize = size)                  # using opencv function "pyrUp" to reconstruct the image

        recon_img = cv2.add(recon_img, combined_result[i])                # recon_img image + expand image



    return recon_img # return result image




# -------------------------- main -------------------------- #
from ex3_utils import *
import time


def laplaceianReduce(img: np.ndarray, levels):
# generate Laplacian Pyramid for apple
    apple_copy = img
    gp_apple=gaussianPyr(img, levels)
    lp_apple = laplaceianReduce2(gp_apple[levels - 1], gp_apple, levels)
    return lp_apple
def laplaceianReduce2(img: np.ndarray,gp_apple:np.ndarray, levels):
# generate Laplacian Pyramid for apple
    apple_copy = img
    lp_apple = [apple_copy]
    for i in range(levels-1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_apple[i])
        laplacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
        lp_apple.append(laplacian)

    return lp_apple
def laplaceianExpand(apple_orange_pyramid) -> np.ndarray:
    apple_orange_reconstruct = apple_orange_pyramid[0]
    for i in range(1, len(apple_orange_pyramid)):
        apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
        apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)
    return apple_orange_reconstruct





def gaussianPyr(apple: np.ndarray, levels: int = 4):

    apple_copy = apple.copy()

    gp_apple = [apple_copy]

    for i in range(levels):

        apple_copy = cv2.pyrDown(apple_copy)

        gp_apple.append(apple_copy)
    return gp_apple



def gaussExpand(apple_orange_reconstruct: np.ndarray, gs_k: np.ndarray) -> np.ndarray:

    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)
    return apple_orange_reconstruct


def pyrBlend(target,  background, gray_mask, level):
    gray_mask = gray_mask.astype('float32')
    gray_mask = gray_mask * 255
    gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_RGB2GRAY)  # convert mask image into grayscale

    gauss_background1 = gauss_pyramid(background, 1)  # generate gaussin pyramid for background image

    gauss_target1 = gauss_pyramid(target, 1)  # generate gaussin pyramid for target image

    gauss_mask1 = gauss_pyramid(gray_mask, 1)  # generate gaussin pyramid for mask image

    # generate laplacian pyramid

    lap_background1 = laplacian_pyramid(gauss_background1, 1)  # generate laplacian pyramid for background image

    lap_target1 = laplacian_pyramid(gauss_target1, 1)  # generate laplacian pyramid for target image

    lap_mask1 = laplacian_pyramid(gauss_mask1, 1)  # generate laplacian pyramid for mask image


    gauss_background = gauss_pyramid(background,level)  # generate gaussin pyramid for background image

    gauss_target = gauss_pyramid(target,level)  # generate gaussin pyramid for target image

    gauss_mask = gauss_pyramid(gray_mask,level)  # generate gaussin pyramid for mask image

# generate laplacian pyramid

    lap_background = laplacian_pyramid(gauss_background,level)  # generate laplacian pyramid for background image

    lap_target = laplacian_pyramid(gauss_target,level)  # generate laplacian pyramid for target image

    lap_mask = laplacian_pyramid(gauss_mask,level)  # generate laplacian pyramid for mask image

# combine two laplacian pyramids with a mask and reconstruct the image
    apple_orange = combine_lp_mask(gauss_mask1, lap_background1, lap_target1,1)

    result_oitput = combine_lp_mask(gauss_mask, lap_background, lap_target,level)


    return [apple_orange, result_oitput]


'''

TODO: Add comments for this method

'''

def reduce(image, level=1):

    result = np.copy(image)



    for _ in range(level - 1):

        result = cv2.pyrDown(result)



    return result





'''

TODO: Add comments for this method

'''

def expand(image, level=1):

    return cv2.pyrUp(np.copy(image))





'''

TODO: Add comments for this method

'''

def compute_flow_map(u, v, gran=8):

    flow_map = np.zeros(u.shape)



    for y in range(flow_map.shape[0]):

        for x in range(flow_map.shape[1]):



            if y % gran == 0 and x % gran == 0:

                dx = 10 * int(u[y, x])

                dy = 10 * int(v[y, x])



                if dx > 0 or dy > 0:

                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)



    return flow_map





'''

TODO: Add comments for this method

'''
def opticalFlow(im1, im2,step_size=10, win=7):


    Ix = np.zeros(im1.shape)

    Iy = np.zeros(im1.shape)

    It = np.zeros(im1.shape)



    Ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2

    Iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2

    It[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]



    params = np.zeros(im1.shape + (5,)).astype(int)

    params[..., 0] = cv2.GaussianBlur(Ix * Ix, (5, 5), 3)

    params[..., 1] = cv2.GaussianBlur(Iy * Iy, (5, 5), 3)

    params[..., 2] = cv2.GaussianBlur(Ix * Iy, (5, 5), 3)

    params[..., 3] = cv2.GaussianBlur(Ix * It, (5, 5), 3)

    params[..., 4] = cv2.GaussianBlur(Iy * It, (5, 5), 3)



    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)

    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -

                  cum_params[2 * win + 1:, :-1 - 2 * win] -

                  cum_params[:-1 - 2 * win, 2 * win + 1:] +

                  cum_params[:-1 - 2 * win, :-1 - 2 * win])



    u = np.zeros(im1.shape)

    v = np.zeros(im1.shape)



    Ixx = win_params[..., 0].astype(int)

    Iyy = win_params[..., 1].astype(int)

    Ixy = win_params[..., 2].astype(int)

    Ixt = -win_params[..., 3].astype(int)

    Iyt = -win_params[..., 4].astype(int)



    M_det = Ixx * Iyy - Ixy ** 2

    temp_u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)

    temp_v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)

    op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)

    op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)



    u[win + 1: -1 - win, win + 1: -1 - win] = op_flow_x[:-1, :-1]

    v[win + 1: -1 - win, win + 1: -1 - win] = op_flow_y[:-1, :-1]
    flow_map1 = compute_flow_map(u, v, 8)
    plt.imshow(flow_map1.astype(int), cmap='gray')

    plt.show()
    i1= im1.reshape(1,im1.shape[0]*im1.shape[1] )
    i2=im2.reshape(1,im2.shape[0]*im2.shape[1] )
    u= u.reshape(1,u.shape[0]*u.shape[1] )
    v=v.reshape(1,v.shape[0]*v.shape[1] )
    u=u[0]
    v=v[0]
    i1=i1[0]
    i2=i2[0]
    a=np.array((u, v)).T
    b=np.array((i1, i2)).T


    return b,a

