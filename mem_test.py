#! /usr/bin/python
import ctypes
import sys
size = int(sys.argv[1])
class MemoryTest(ctypes.Structure):
    _fields_ = [  ('chars' , ctypes.c_char*size * 1024*1024 ) ]
try:
    test = MemoryTest()
    print('success => {0:>4}MB was allocated'.format(size) )
except:
    print('failure => {0:>4}MB can not be allocated'.format(size) )