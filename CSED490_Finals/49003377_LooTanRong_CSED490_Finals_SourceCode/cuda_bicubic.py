import numpy as np
import random
import time
import matplotlib
import os
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import DynamicSourceModule

start_time = time.time() * 1000
print("Start!")

if (os.system("cl.exe")):
    os.environ[
        'PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

""" !!! define CUDA kernel here !!! """
cuda_kernel = DynamicSourceModule("""
__device__ __constant__ int constant_a[4][4];
__device__ __constant__ int constant_b[4][4];

__global__ void image_resize (int *n_img, int *o_img, int N, int scale)
{
    // Constant variables here!!!  
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int xi = threadIdx.x + blockDim.x * blockIdx.x;
    const int yi = threadIdx.y + blockDim.y * blockIdx.y;
    
    const int tile_w = blockDim.x - 3;
    const int val_o = blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x + threadIdx.y * N + threadIdx.x;
    
    // Shared memmory
    __shared__ int sub_img[8 + 3][8 + 3];

    sub_img[y + 1][x + 1] = o_img[val_o];
    
    if (x == 0)
    {  
        if (xi == 0) {sub_img[y + 1][x] = 0;}
        
        else {sub_img[y + 1][x] = o_img[val_o - 1];}
    }
    
    if (y == 0)
    {
        if (yi == 0) {sub_img[y][x + 1] = 0;}
        
        else {sub_img[y][x + 1] = o_img[val_o - N];}
    }
    
    if (x == 0 && y == 0)
    {
        if (xi == 0 && yi == 0) {sub_img[y][x] = 0;}
        
        else {sub_img[y][x] = o_img[val_o - N - 1];}
    }
    
    if (x >= blockDim.x - 1)
    {
        if (xi >= N - 1) {sub_img[y + 1][x + 2] = 0; sub_img[y + 1][x + 3] = 0;}
        
        else {sub_img[y + 1][x + 2] = o_img[val_o + 1]; sub_img[y + 1][x + 3] = o_img[val_o + 2];}
    }
    
    if (y >= blockDim.y - 1)
    {
        if (yi >= N - 1) {sub_img[y + 2][x + 1] = 0; sub_img[y + 3][x + 1] = 0;}
        
        else {sub_img[y + 2][x + 1] = o_img[val_o + N]; sub_img[y + 3][x + 1] = o_img[val_o + 2 * N];}
    }
    
    if (x >= blockDim.x - 1 && y >= blockDim.y - 1)
    {
        if (xi >= N - 1 && yi >= N - 1) 
        {
            sub_img[y + 2][x + 2] = 0;
            sub_img[y + 2][x + 3] = 0;
            sub_img[y + 3][x + 2] = 0;
            sub_img[y + 3][x + 3] = 0;
        }
        
        else
        {
            sub_img[y + 2][x + 2] = o_img[val_o + N + 1];
            sub_img[y + 2][x + 3] = o_img[val_o + N + 2];
            sub_img[y + 3][x + 2] = o_img[val_o + 2 * N + 1];
            sub_img[y + 3][x + 3] = o_img[val_o + 2 * N + 2];
        }
    }
    
    __syncthreads();
    
    // CUDA Operations
    n_img[val_o] = sub_img[y][x];
    
}
                        """

# Bicubic interpolation with shared memory
"""
__device__ __constant__ int constant_a[4][4];
__device__ __constant__ int constant_b[4][4];

__global__ void image_resize (int *n_img, int *o_img, int N, int scale)
{
    // Constant variables here!!!  
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int xi = threadIdx.x + blockDim.x * blockIdx.x;
    const int yi = threadIdx.y + blockDim.y * blockIdx.y;
    
    const int val_o =   blockIdx.y * blockDim.y * N
                      + blockIdx.x * blockDim.x
                      + threadIdx.y * N
                      + threadIdx.x;
                      
    const int val_pn = blockIdx.y * blockDim.y * N * scale * scale +
                       blockIdx.x * blockDim.x * scale +
                       threadIdx.y * N * scale * scale +
                       threadIdx.x * scale;
    
    // Shared memmory
    __shared__ int sub_img[32 + 3][32 + 3];

    sub_img[y + 1][x + 1] = o_img[val_o];
    
    if (x == 0)
    {  
        if (xi == 0) {sub_img[y + 1][x] = 0;}
        
        else {sub_img[y + 1][x] = o_img[val_o - 1];}
    }
    
    if (y == 0)
    {
        if (yi == 0) {sub_img[y][x + 1] = 0;}
        
        else {sub_img[y][x + 1] = o_img[val_o - N];}
    }
    
    if (x == 0 && y == 0)
    {
        if (xi == 0 && yi == 0) {sub_img[y][x] = 0;}
        
        else {sub_img[y][x] = o_img[val_o - N - 1];}
    }
    
    if (x >= blockDim.x - 1)
    {
        if (xi >= N - 1) {sub_img[y + 1][x + 2] = 0; sub_img[y + 1][x + 3] = 0;}
        
        else {sub_img[y + 1][x + 2] = o_img[val_o + 1]; sub_img[y + 1][x + 3] = o_img[val_o + 2];}
    }
    
    if (y >= blockDim.y - 1)
    {
        if (yi >= N - 1) {sub_img[y + 2][x + 1] = 0; sub_img[y + 3][x + 1] = 0;}
        
        else {sub_img[y + 2][x + 1] = o_img[val_o + N]; sub_img[y + 3][x + 1] = o_img[val_o + 2 * N];}
    }
    
    if (x >= blockDim.x - 1 && y >= blockDim.y - 1)
    {
        if (xi >= N - 1 && yi >= N - 1) 
        {
            sub_img[y + 2][x + 2] = 0;
            sub_img[y + 2][x + 3] = 0;
            sub_img[y + 3][x + 2] = 0;
            sub_img[y + 3][x + 3] = 0;
        }
        
        else
        {
            sub_img[y + 2][x + 2] = o_img[val_o + N + 1];
            sub_img[y + 2][x + 3] = o_img[val_o + N + 2];
            sub_img[y + 3][x + 2] = o_img[val_o + 2 * N + 1];
            sub_img[y + 3][x + 3] = o_img[val_o + 2 * N + 2];
        }
    }
    
    __syncthreads();
    
    // Extract input matrix
    int in_m[4][4];
    
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            in_m[i][j] = sub_img[y + i][x + j];
        }
    }
    
    // Generate partial gradient matrix
    // Multiply weight a & input matrix
    
    int a_m[4][4];
    
    for (int i = 0; i < 4; i ++)
    {
        for (int j = 0; j < 4; j++)
        {
            a_m[i][j] = constant_a[i][0] * in_m[0][j] +
                        constant_a[i][1] * in_m[1][j] +
                        constant_a[i][2] * in_m[2][j] +
                        constant_a[i][3] * in_m[3][j];
        }
    }
    
    // Generate full gradient matrix
    // Multiply partial gradient matrix & weight b
    
    int grad_m[4][4];
    
    for (int i = 0; i < 4; i ++)
    {
        for (int j = 0; j < 4; j++)
        {
            grad_m[i][j] = a_m[i][0] * constant_b[0][j] +
                           a_m[i][1] * constant_b[1][j] +
                           a_m[i][2] * constant_b[2][j] +
                           a_m[i][3] * constant_b[3][j];
        }
    }
    
    // generate new pixel
    int p_pixel[4];
    int xs;
    int ys;
    int val_n;
    
    for (int cnt_y = 0; cnt_y < scale; cnt_y++)
    {
        ys = cnt_y * (1 / scale);
        
        for (int cnt_x = 0; cnt_x < scale; cnt_x++)
        {
            val_n = val_pn + cnt_y * N * scale + cnt_x;
            xs = cnt_x * (1 / scale);
            
            for (int i = 0; i < 4; i ++)
            {
                p_pixel[i] = grad_m[0][i] +
                             xs * grad_m[1][i]+
                             xs * xs * grad_m[2][i] +
                             xs * xs * xs *grad_m[3][i];
            }
            
            n_img[val_n] = p_pixel[0] +
                           p_pixel[1] * ys +
                           p_pixel[2] * ys * ys +
                           p_pixel[3] * ys * ys * ys;
            
        }
    }   
}
""")

kernel_time = time.time() * 1000
delta_time = kernel_time - start_time
print("\nCUDA kernel took {}ms to initialise.".format(delta_time))


def random_square_image(width):
    random.seed()
    img = np.array([])

    for y in range(width):
        img_row = np.array([])

        for x in range(width):
            random_pixel = random.randint(0, 255)
            img_row = np.append(img_row, random_pixel)

        if y == 0:
            img = np.array(img_row)

        else:
            img = np.vstack((img, img_row))

    return img.astype(np.int32)


def generate_weights():
    a = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [-3, 3, -2, -1],
                  [2, -2, 1, 1]]).astype(np.int32)

    b = np.array([[1, 0, -3, 2],
                  [0, 0, 3, -2],
                  [0, 1, -2, 1],
                  [0, 0, -1, 1]]).astype(np.int32)

    return a, b


" CUDA parameters "
img_size = 256
img_scale = np.int32(2)
dim_block = 32
dim_grid = int(img_size / dim_block + 1)

resize_img = cuda_kernel.get_function("image_resize")

print("\nGenerating image ... ...\n")

" Generate image & extract details "
o_img = random_square_image(img_size)  # generate random square image
o_img_N = np.int32(o_img.shape[0])  # no of pixels in o_img
o_img_stride = np.int32(o_img.strides[0])  # no of bytes per row
o_img_byte = o_img.size * o_img.dtype.itemsize  # total no of byte in o_img
print(o_img)

img_time = time.time() * 1000
delta_time = img_time - kernel_time
print("\nImage took {}ms to create.".format(delta_time))

" define NEW IMAGE parameters "
new_img = np.zeros((img_size * img_scale, img_size * img_scale)).astype(np.int32)  # generate size for new image
new_img_N = np.int32(new_img.shape[0])  # no of pixels in new_img
new_img_stride = np.int32(new_img.strides[0])  # no of bytes per row
new_img_byte = new_img.size * new_img.dtype.itemsize  # total no of byte in new_img
print(new_img_stride)

" Generate weights "
w_a, w_b = generate_weights()

print("\nCopying to GPU memory ... ... \n")

pre_time = time.time() * 1000

" Transfer data from CPU memory to GPU DRAM "
o_img_gpu = drv.mem_alloc(o_img_byte)
new_img_gpu = drv.mem_alloc(new_img_byte)

c_a, _ = cuda_kernel.get_global("constant_a")
c_b, _ = cuda_kernel.get_global("constant_b")

drv.memcpy_htod(o_img_gpu, o_img)
drv.memcpy_htod(new_img_gpu, new_img)
drv.memcpy_htod(c_a, w_a)
drv.memcpy_htod(c_b, w_b)

print("Finish memory copy!\n\nStarting image resizing with CUDA... ...\n")

" image resizing with bicubic interpolation in GPU "
resize_img(new_img_gpu, o_img_gpu, o_img_N, img_scale,
           block=(dim_block, dim_block, 1), grid=(dim_grid, dim_grid))

print("Resize completed!\n\nCopying new image to system memory ...\n")

" transfer new image back from GPU DRAM to CPU memory "
drv.memcpy_dtoh(new_img, new_img_gpu)

resize_time = time.time() * 1000
delta_time = resize_time - pre_time
print("Image took {}ms to resize.\n".format(delta_time))

print("Copy done!\n")
print(new_img)

final_time = time.time() * 1000
t_time = final_time - start_time
print("Total compute time = {}ms.".format(t_time))
print("Image size: {}".format(img_size))
print("Block size: {}".format(dim_block))

#pycuda.autoinit.context.detach()
