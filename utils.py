import cv2
import numpy as np

def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    h = mask.shape[0]
    w = mask.shape[1]
    gt_poly = np.zeros((poly.shape[0],poly.shape[1]),np.int32)
    gt_poly[:,0] = np.floor(poly[:,0]*w)
    gt_poly[:,1] = np.floor(poly[:,1]*h)    
    cv2.polylines(mask, np.int32([gt_poly]), True, [1])

    return mask

def get_fp_mask(poly,mask):
    h = mask.shape[0]
    w = mask.shape[1]
    x = np.int32(np.floor(poly[0,0]*w))
    y = np.int32(np.floor(poly[0,1]*h))
    mask[y,x] = 1
    return mask