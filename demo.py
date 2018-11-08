from __init__ import image_data
import numpy as np
from getJmap_EM import getJmap_EM
from getJmapNA_EM import getJmapNA_EM
from main_files import smooth_unshift
from PIL import Image
from mat_fn import filt
import time
t=time.time()


ncomp=0
c2=6
#input image directory
image = image_data('florence2_tamp95_3_7.jpg')


########For NonAlign Image####################
LLRmap ,LLRmap_s, q1table, k1e, k2e, alphatable = getJmapNA_EM(image,ncomp,c2)
map_final = smooth_unshift(np.sum(LLRmap,2),k1e,k2e)
print 'Estimated Q1 Table' ,'\n',q1table.astype(int)
print 'Estimated alphatable' ,'\n',np.round(alphatable,4)
#eng.eval('imagesc('+n2m(map_final)+')')
im = Image.fromarray(np.uint8(map_final))
im.show()





'''
########For Align Image####################


LLRmap, LLRmap_s, q1table, alphatable = getJmap_EM(image,ncomp,c2)
#filter an array
map_final = filt(np.sum(LLRmap,2))
print 'Estimated Q1 Table' ,'\n',q1table.astype(int)
print 'Estimated alphatable' ,'\n',np.round(alphatable,4)
#create an array of result
im = Image.fromarray(np.uint8(map_final))
#show result as an image 
im.show()
#print time taken by program
print time.time()-t

'''


