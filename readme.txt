
This code is the pyhton implementation of "T. Bianchi and A. Piva, “Image forgery localization via blockgrained
analysis of JPEG artifacts,” IEEE Trans. Information Forensics and Security, vol. 7, no. 3, 2012"

This code is python conversion of matlab code which is available on github (https://github.com/caomw/matlab-forensics)

There are two main function getJmapNA_EM and getJmap_EM which are in getJmapNA_EM.py and getJmap_EM.py
files respectively. For align folger we use getJmap_EM and for non-align forgery we use getJmapNA_EM.
Uses of these function can be shown in demo file.

-----------Prerequisite------------
PIL, numpy, scipy library should be alredy installed



-----------dependencies------------
These files has been taken from pysteg which is available at:
http://www.ifs.schaathun.net/pysteg/starting.html#download-and-installation

__init__.py
base.py
compress.py
dct.py
jpegobject.c
jpegobject.o
jpegObject.so




---------matlab functions----------

In mat_fn.py file all function are python conversion of matlab original functions




