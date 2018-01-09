from PIL import Image
import os
	

def is_bin(path):
	l = os.listdir(path)
	s=set()
	for im in l:
		print(im)
		img = Image.open(os.path.join(path,im)) 
		pix = img.load()
		width, height = img.size
		for x in range(width):
			for y in range(height):
				s.add(pix[x,y])
	for i in s:
		print(i)
	return len(s)==2
def adapt(path):
	img = Image.open(path) 
	pix = img.load()
	width, height = img.size
	for x in range(width):
		for y in range(height):
			if pix[x,y] >= 200:
				pix[x,y] = 255
			else:
				pix[x,y] = 0
	img.save(path) 
	img.close()



adapt(os.path.join("/","tmp","result.jpg"))
#print(is_bin(os.path.join("data200_200","targets")))
