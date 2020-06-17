import os
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, PSM

class ImageEnhancement:
    
    def __init__(self, pdf_path):
        self.images = convert_from_path(pdf_path)
        
    def ROIcropper(self, img):
        bit = cv2.bitwise_not(img)
        nonzero = np.nonzero(bit)
        xmin,xmax,ymin,ymax = min(nonzero[1]),max(nonzero[1]),min(nonzero[0]),max(nonzero[0])
        return img[ymin:ymax,xmin:xmax]
        
    def img_sharpening(self, img):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(img, -1, kernel)
        return sharp
    
    def thresholding(self, img):
        _, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th
    
    def deskew_angle(self,img):
        with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
            image = Image.fromarray(img)
            api.SetImage(image)
            api.Recognize()
            it = api.AnalyseLayout()
            orientation, direction, order, deskew_angle = it.Orientation()
        return deskew_angle
    
    def deskew(self, img):
        angle = self.deskew_angle(img)*180/np.pi
        cv2.bitwise_not(img, img)
    
        #compute the minimum bounding box:
        non_zero_pixels = cv2.findNonZero(img)
        center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    
        root_mat = cv2.getRotationMatrix2D(center, angle, 1)
        rows, cols = img.shape
        rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)


        #Border removing:
        sizex = np.int0(wh[0])
        sizey = np.int0(wh[1])
        if theta > -45 :
            temp = sizex
            sizex= sizey
            sizey= temp
        return cv2.getRectSubPix(rotated, (sizey,sizex), center)
    
    def culminator(self):
        img_list = []
        for img in self.images:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.thresholding(img)
            img = self.ROIcropper(img)
            img = self.img_sharpening(img)
            img = self.deskew(img)
            cv2.bitwise_not(img, img)
            img_list.append(img)
        return img_list
            
        
obj = ImageEnhancement("pdfs/2019 W-2.pdf")
Image.fromarray(obj.culminator()[1])
        


