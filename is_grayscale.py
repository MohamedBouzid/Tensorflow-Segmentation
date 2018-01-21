from PIL import Image
from IPython.display import display, Image as Im
from scipy import ndimage
import numpy as np
import cv2
import os

def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True

def size(path):
	l = os.listdir(path)
	for im in l:
		print(im)	
		try:
			print(ndimage.imread(os.path.join(path,im)).shape)
		except:
			print("not an image -- skipped --")

def greyscaletest(path):
	l = os.listdir(path)
	b=True
	for im in l:
		if(im[-3:-1]=="jp"):
			b=b and is_grey_scale(os.path.join(path,im))
	return b


def resize(path, height, width):
	l = os.listdir(path)
	i=0
	for im in l:
		#print(im[-3:-1])
		if(im[-3:]=="png"):
			print(im)	
			i+=1
			print("DEBUG -- before")
			imag = cv2.imread(os.path.join(path,im))
			mod = cv2.resize(imag, (height, width))
			cv2.imwrite(os.path.join(path,im), mod[:,:,2])
			print(ndimage.imread(os.path.join(path,im)).shape)
			print("DEBUG -- after")	

def resizeone(path, height, width):
		imag = cv2.imread(path)
		mod = cv2.resize(imag, (height, width))
		cv2.imshow("im",mod[:,:,2])
		cv2.waitKey(0)
		cv2.imwrite(path, mod)
		print(ndimage.imread(path).shape)
		print("DEBUG -- after")	


def modify(path):

	img = Image.open(path) 
	pix = img.load()
	width, height = img.size
	for x in range(width):
		for y in range(height):
			if pix[x,y] > 128:
				pix[x,y] = 255
			else:
				pix[x,y] = 0
	img.save(path) 
	img.close()

def clean_image(path):
	l = os.listdir(path)
	for im in l:
		if(im[-3:]=="png"):
			modify(os.path.join(path,im))

def print_imageset(path):
	l = os.listdir(path)
	for im in l:
		ima = ndimage.imread(os.path.join(path,im))
		print(ima)

def print_image(path):
	ima = ndimage.imread(path)
	print(ima)
#modify("imageset1300.png")
#resize(os.path.join("data200_200","inputs"), 128, 128)
#resize(os.path.join("data200_200","inputs1"), 128, 128)
#resize(os.path.join("data200_200","inputs"), 128, 128)
#resize(os.path.join("data200_200","targets"), 128, 128)
#resize(os.path.join("data200_200","targets1"), 128, 128)
#clean_image(os.path.join("data200_200","targets1"))
#clean_image(os.path.join("data200_200","targets"))

np.set_printoptions(threshold=np.nan)

s=192
#resize(os.path.join("data128_128","inputs"), s, s)
#resize(os.path.join("data128_128","targets"), s, s)
#print(size(os.path.join("data200_200","inputs")))
#print(size(os.path.join("data200_200","targets")))

#print(ndimage.imread(os.path.join("/","tmp","result.jpg")).shape)
#print_image(os.path.join("/","tmp","result.jpg"))
#print(print_imageset(os.path.join("data200_200","targets")))
#print_imageset(os.path.join("data128_128","targets"))
#print_imageset(os.path.join("data200_200","inputs"))
#print(greyscaletest(os.path.join("data200_200","inputs")))
#print_imageset(os.path.join("data200_200","targets"))
#clean_image(os.path.join("data","targets"))
#print("cleaned")
#resize(os.path.join("data","inputs"), 128, 128)
#resize(os.path.join("data","targets"), 128, 128)
#print(size(os.path.join("data","inputs")))
#print(size(os.path.join("data","targets")))
#print_imageset(os.path.join("data200_200","targets"))
#print(size(os.path.join("data200_200","inputs")))
#print(size(os.path.join("data200_200","targets")))
#resizeone(os.path.join("data200_200","targets","labels0000.png"),128,128)
#print("size him = "+str(ndimage.imread(os.path.join("data128_128","targets","Aaron_Peirsol_0001.jpg")).shape))
#print("size me = "+str(ndimage.imread(os.path.join("data200_200","targets","labels0000.jpg")).shape))

