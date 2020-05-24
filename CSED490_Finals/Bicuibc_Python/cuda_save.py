# KNN 1D
"""
__global__ void image_resize (int *n_img, int *o_img, int o_N, int scale)
{
    // Constant variables here!!!
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int n_N = o_N * scale;
    const int o_size = o_N * o_N;

    // Processes data here!!!
    // Data: mem location + type


    //  CUDA Memory access


    // CUDA Operations
    if (x < o_size)
    {
        int div = int(x / o_N);
        int par_val = x * scale + div * (o_N * (scale * scale - scale));

        for (int row_cnt = 0; row_cnt < scale; row_cnt++)
        {
            for (int col_cnt = 0; col_cnt < scale; col_cnt++)
            {
                int p_val = par_val + row_cnt * n_N + col_cnt;
                n_img[p_val] = o_img[x];
            }
        }
    }
}
"""


# KNN 2D
"""
__global__ void image_resize (int *n_img, int *o_img, int o_N, int scale)
{
    // Constant variables here!!!  
    const int o_val = threadIdx.x + blockDim.x * blockIdx.x + o_N * threadIdx.y + 2 * o_N * blockIdx.y;
    const int p_val = o_val * scale + (o_val / o_N) * (o_N * (scale * scale - scale));

    // Processes data here!!!
    // Data: mem location + type


    //  CUDA Memory access


    // CUDA Operations
    for (int col_cnt = 0; col_cnt < scale; col_cnt++)
    {
        for (int row_cnt = 0; row_cnt < scale; row_cnt++)
        {
            int n_val = p_val + row_cnt + col_cnt * scale * o_N;
            n_img[n_val] = o_img[o_val];
        }
    }
}
                        """

# KNN 2D with shared mem
"""
__device__ __constant__ int constant_a[4][4];
__device__ __constant__ int constant_b[4][4];

__global__ void image_resize (int *n_img, int *o_img, int o_N, int scale)
{
    // Constant variables here!!!  
    const int o_val = threadIdx.x + blockDim.x * blockIdx.x + o_N * threadIdx.y + 2 * o_N * blockIdx.y;
    const int p_val = o_val * scale + (o_val / o_N) * (o_N * (scale * scale - scale));
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    // Shared memmory
    __shared__ int sub_img[32][32];
    sub_img[y][x] = o_img[o_val];
    __syncthreads();        

    // Extract input matrix

    // CUDA Operations
    for (int col_cnt = 0; col_cnt < scale; col_cnt++)
    {
        for (int row_cnt = 0; row_cnt < scale; row_cnt++)
        {
            int n_val = p_val + row_cnt + col_cnt * scale * o_N;
            n_img[n_val] = sub_img[y][x];
        }
    }
}
                        """

# shared mem with many control divergence
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
"""

# Bicubic resize w/ shared memory & cuda stream
"""
__device__ __constant__ int constant_a[4][4];
__device__ __constant__ int constant_b[4][4];

__device__ int in_m[4][4];
__device__ int a_m[4][4];
__device__ int grad_m[4][4];

__device__ int tx;
__device__ int ty;
__device__ int bx;
__device__ int by;

__global__ void grad_multiply (void)
{
    a_m[ty][tx] = constant_a[ty][0] * in_m[0][tx] +
                  constant_a[ty][1] * in_m[1][tx] +
                  constant_a[ty][2] * in_m[2][tx] +
                  constant_a[ty][3] * in_m[3][tx];


    grad_m[ty][tx] = a_m[ty][0] * constant_b[0][tx] +
                     a_m[ty][1] * constant_b[1][tx] +
                     a_m[ty][2] * constant_b[2][tx] +
                     a_m[ty][3] * constant_b[3][tx];
}    

__global__ void image_resize (int *n_img, int *o_img, int N, int scale)
{
    // Constant variables here!!!  
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int xi = threadIdx.x + blockDim.x * blockIdx.x;
    const int yi = threadIdx.y + blockDim.y * blockIdx.y;

    tx = threadIdx.x;
    ty = threadIdx.y;
    bx = blockIdx.x;
    by = blockIdx.y;

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

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            in_m[i][j] = sub_img[y + i][x + j];
        }
    }

    // Generate full gradient matrix
    // Multiply partial gradient matrix & weight b

    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    grad_multiply <<<1, (4, 4)>>> ();
    cudaStreamDestroy(s);

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
"""