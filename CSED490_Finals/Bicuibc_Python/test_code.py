import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule

import os

if (os.system("cl.exe")):
    os.environ[
        'PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

'''mod = SourceModule("""
__global__ void multiply_them(int *dest, int N)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (i < N * N)
  {
    dest[i] = i;
  }
}
""")

multiply_them = mod.get_function("multiply_them")

x = 128
dim_block = 1024
dim_grid = int((x ** 2) / dim_block)

a = numpy.zeros((x, x)).astype(numpy.int32)
a_N = numpy.int32(a.shape[0])  # no of rows in a
a_stride = numpy.int32(a.strides[0])  # no of byte per row
a_bytes = a.size * a.dtype.itemsize  # total no of bytes req

a_gpu = drv.mem_alloc(a_bytes)  # bytes allocated in GPU for array a
drv.memcpy_htod(a_gpu, a)

multiply_them(
    a_gpu, a_N,
    block=(dim_block, 1, 1), grid=(dim_grid, 1))
drv.memcpy_dtoh(a, a_gpu)

print(a)'''

'''def matrix(size):
    m = []
    i = 0
    for y in range(size):
        p_m = []
        for x in range(size):
            p_m.append(i)
            i += 1
        m.append(p_m)
    return np.array(m)


a = 2
for i in range(1, 5, 1):
    ma = matrix(a * i)
    print(ma)
a = 3
for i in range(1, 5, 1):
    ma = matrix(a * i)
    print(ma)
a = 4
for i in range(1, 5, 1):
    ma = matrix(a * i)
    print(ma)'''

a = np.array([1, 2, 3, 4]).astype(np.int32)
a_byte = a.size * a.dtype.itemsize
a_gpu = drv.mem_alloc(a_byte)
drv.memcpy_htod(a_gpu, a)
b = drv.Module.get_global(a_gpu)
print(b)
