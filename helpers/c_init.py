import ctypes
import os
try:
	path = os.path.dirname(os.path.abspath(__file__))
	parent = os.path.dirname(path)
	_c_extension = ctypes.CDLL(parent + '/so_files/libsom.so')
	_c_extension.train_from_c.argtypes = (ctypes.c_int , ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.train_from_c.restype = ctypes.POINTER(ctypes.c_double)

	_c_extension.train_from_c_periodic.argtypes = (ctypes.c_int , ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),ctypes.POINTER(ctypes.c_double),ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.train_from_c_periodic.restype = ctypes.POINTER(ctypes.c_double)

	_c_extension.map_from_c.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.map_from_c.restype = ctypes.POINTER(ctypes.c_int)
	
	_c_extension.activation_from_c.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.activation_from_c.restype = ctypes.POINTER(ctypes.c_int)


	C_INIT_SUCESS = True
except Exception as e:
	print("WARNING C-Library could not be imported.\nC-accelerated functions can not be used.\nConsider train_async instead of train for speed boost, if using large batches.\nError as follows %s"%e)
	_c_extension = None
	C_INIT_SUCESS = False