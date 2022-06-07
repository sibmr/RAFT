#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#define BLOCK_H 4
#define BLOCK_W 8
#define BLOCK_HW BLOCK_H * BLOCK_W
#define CHANNEL_STRIDE 32


__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

/**
 * @brief Perform grid indexing using the image features
 * 
 * @tparam scalar_t   datatype of the tensor
 * @param fmap1   Tensor containing image 1 features (full resolution) of shape (batch, ht, wd, fdim)
 * @param fmap2   Tensor containing image 2 features (pooled) of shape (batch, ht/2**i, wd/2**i, fdim)
 * @param coords  Current correspondence estimation of shape (batch, 1, ht, wd, 2)
 * @param corr    indexed correlation volume of shape (batch, 1, (2*r+1)**2, ht, wd)
 * @param r       Radius for correlation lookup
 * @return __global__ 
 */
template <typename scalar_t>
__global__ void corr_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,
    int r)
{

  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);

  const int b = blockIdx.x; // current batch

  // global starting x/y value for this block
  const int h0 = blockIdx.y * blockDim.x;   // block_i * BLOCK_H
  const int w0 = blockIdx.z * blockDim.y;   // block_j * BLOCK_W
  
  //                (0 to 3)  * 8          + (0 to 7)
  const int tid = threadIdx.x * blockDim.y + threadIdx.y; // threadId that is unique within its block

  const int H1 = fmap1.size(1); // ht of img1
  const int W1 = fmap1.size(2); // wd of img1
  const int H2 = fmap2.size(1); // ht of img2: ht/2**i
  const int W2 = fmap2.size(2); // wd of img2: wd/2**i
  const int N = coords.size(1); // 1
  const int C = fmap1.size(3); // fdim of img features

  // memory that is shared between all threads in one block
  // this memory has much faster access times
  __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t x2s[BLOCK_HW];
  __shared__ scalar_t y2s[BLOCK_HW];

  // stride over all channels
  for (int c=0; c<C; c+=CHANNEL_STRIDE) {
    // go over all channels in the current stride (if BLOCK_HW==CHANNEL_STRIDE)
    // load feature channels this thread is responsible for to shared memory
    for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
      
      //  k1 = (0 to 31) + (0 to 31) / 32
      //  k1 = k if (CHANNEL_STRIDE = BLOCK_HW = 32)
      int k1 = k + tid / CHANNEL_STRIDE;
      
      // global pixel indices covererd by this block
      int h1 = h0 + k1 / BLOCK_W; // h0 + (0 to 3)
      int w1 = w0 + k1 % BLOCK_W; // w0 + (0 to 7)
      int c1 = tid % CHANNEL_STRIDE;  // channel offset for this thread

      // iterates over all global pixels covered by this block for each thread
      auto fptr = fmap1[b][h1][w1]; // pointer to feature dimension at h1, w1

      // assign the first 32 feature channels of image1 for this thread
      if (within_bounds(h1, w1, H1, W1))
        f1[c1][k1] = fptr[c+c1];
      else
        f1[c1][k1] = 0.0;
    }

    // barrier: wait for other threads in this block to arrive
    __syncthreads();

    // N=1 thus this only runs once
    for (int n=0; n<N; n++) {
      // pixel coordinate for this thread
      int h1 = h0 + threadIdx.x;
      int w1 = w0 + threadIdx.y;
      // if the current pixel is inside the bounds of the image
      if (within_bounds(h1, w1, H1, W1)) {
        x2s[tid] = coords[b][n][h1][w1][0]; // threadId to correspondenceX
        y2s[tid] = coords[b][n][h1][w1][1]; // threadId to correspondenceY
      }

      // between-pixel correspondence coordinates
      scalar_t dx = x2s[tid] - floor(x2s[tid]);
      scalar_t dy = y2s[tid] - floor(y2s[tid]);

      // grid width/height
      int rd = 2*r + 1;

      // for each point on the grid (around the current/cor)
      for (int iy=0; iy<rd+1; iy++) {
        for (int ix=0; ix<rd+1; ix++) {
          // for each channel this thread is responsible for
          // load feature channels for this pixel to shared memory
          for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
            // (0 to 31) + (0 to 31) / 31 = k
            int k1 = k + tid / CHANNEL_STRIDE;
            // coordinates within this threads grid (around correspondence)
            int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
            int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
            // the channel offset for this thread
            int c2 = tid % CHANNEL_STRIDE;

            // pointer to features of current pixel
            auto fptr = fmap2[b][h2][w2];

            // if current coordinates are within bounds
            if (within_bounds(h2, w2, H2, W2))
              // assign feature of current channel to buffers for all threads
              f2[c2][k1] = fptr[c+c2];
            else
              f2[c2][k1] = 0.0;
          }

          // barrier waits for all threads in block to arrive
          __syncthreads();

          // calculate partial feature dot product for current pixel and correspondence
          scalar_t s = 0.0;
          for (int k=0; k<CHANNEL_STRIDE; k++)
            s += f1[k][tid] * f2[k][tid];

          // top-left: nw, top-right:ne, bottom-left:sw, bottom-right:se

          // global indices for current neighbouring pixel indices (ix, iy)
          int ix_nw = H1*W1*((iy-1) + rd*(ix-1));
          int ix_ne = H1*W1*((iy-1) + rd*ix);
          int ix_sw = H1*W1*(iy + rd*(ix-1));
          int ix_se = H1*W1*(iy + rd*ix);

          // contributions of current pixel to neighbouring grid-point bilinear interpolations
          scalar_t nw = s * (dy) * (dx);
          scalar_t ne = s * (dy) * (1-dx);
          scalar_t sw = s * (1-dy) * (dx);
          scalar_t se = s * (1-dy) * (1-dx);

          // pointer to this threads pixel in the resulting indexed corr volume
          scalar_t* corr_ptr = &corr[b][n][0][h1][w1];

          // add bilinear interpolation contribution to neighboring pixels
          if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_nw) += nw;

          if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_ne) += ne;

          if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_sw) += sw;

          if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
            *(corr_ptr + ix_se) += se;
        }
      } 
    }
  }
}


template <typename scalar_t>
__global__ void corr_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap1_grad,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> fmap2_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> coords_grad,
    int r)
{

  const int b = blockIdx.x;
  const int h0 = blockIdx.y * blockDim.x;
  const int w0 = blockIdx.z * blockDim.y;
  const int tid = threadIdx.x * blockDim.y + threadIdx.y;

  const int H1 = fmap1.size(1);
  const int W1 = fmap1.size(2);
  const int H2 = fmap2.size(1);
  const int W2 = fmap2.size(2);
  const int N = coords.size(1);
  const int C = fmap1.size(3);

  __shared__ scalar_t f1[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t f2[CHANNEL_STRIDE][BLOCK_HW+1];

  __shared__ scalar_t f1_grad[CHANNEL_STRIDE][BLOCK_HW+1];
  __shared__ scalar_t f2_grad[CHANNEL_STRIDE][BLOCK_HW+1];

  __shared__ scalar_t x2s[BLOCK_HW];
  __shared__ scalar_t y2s[BLOCK_HW];

  for (int c=0; c<C; c+=CHANNEL_STRIDE) {

    for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
      int k1 = k + tid / CHANNEL_STRIDE;
      int h1 = h0 + k1 / BLOCK_W;
      int w1 = w0 + k1 % BLOCK_W;
      int c1 = tid % CHANNEL_STRIDE;

      auto fptr = fmap1[b][h1][w1];
      if (within_bounds(h1, w1, H1, W1))
        f1[c1][k1] = fptr[c+c1];
      else
        f1[c1][k1] = 0.0;

      f1_grad[c1][k1] = 0.0;
    }

    __syncthreads();

    int h1 = h0 + threadIdx.x;
    int w1 = w0 + threadIdx.y;

    for (int n=0; n<N; n++) {  
      x2s[tid] = coords[b][n][h1][w1][0];
      y2s[tid] = coords[b][n][h1][w1][1];

      scalar_t dx = x2s[tid] - floor(x2s[tid]);
      scalar_t dy = y2s[tid] - floor(y2s[tid]);

      int rd = 2*r + 1;
      for (int iy=0; iy<rd+1; iy++) {
        for (int ix=0; ix<rd+1; ix++) {
          for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
            int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
            int c2 = tid % CHANNEL_STRIDE;

            auto fptr = fmap2[b][h2][w2];
            if (within_bounds(h2, w2, H2, W2))
              f2[c2][k1] = fptr[c+c2];
            else
              f2[c2][k1] = 0.0;

            f2_grad[c2][k1] = 0.0;
          }

          __syncthreads();
      
          const scalar_t* grad_ptr = &corr_grad[b][n][0][h1][w1];
          scalar_t g = 0.0;

          int ix_nw = H1*W1*((iy-1) + rd*(ix-1));
          int ix_ne = H1*W1*((iy-1) + rd*ix);
          int ix_sw = H1*W1*(iy + rd*(ix-1));
          int ix_se = H1*W1*(iy + rd*ix);

          if (iy > 0 && ix > 0 && within_bounds(h1, w1, H1, W1))
            g +=  *(grad_ptr + ix_nw) * dy * dx;

          if (iy > 0 && ix < rd && within_bounds(h1, w1, H1, W1))
            g += *(grad_ptr + ix_ne) * dy * (1-dx);

          if (iy < rd && ix > 0 && within_bounds(h1, w1, H1, W1))
            g += *(grad_ptr + ix_sw) * (1-dy) * dx;

          if (iy < rd && ix < rd && within_bounds(h1, w1, H1, W1))
            g += *(grad_ptr + ix_se) * (1-dy) * (1-dx);
            
          for (int k=0; k<CHANNEL_STRIDE; k++) {
            f1_grad[k][tid] += g * f2[k][tid];
            f2_grad[k][tid] += g * f1[k][tid];
          }

          for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
            int k1 = k + tid / CHANNEL_STRIDE;
            int h2 = static_cast<int>(floor(y2s[k1]))-r+iy;
            int w2 = static_cast<int>(floor(x2s[k1]))-r+ix;
            int c2 = tid % CHANNEL_STRIDE;

            scalar_t* fptr = &fmap2_grad[b][h2][w2][0];
            if (within_bounds(h2, w2, H2, W2))
              atomicAdd(fptr+c+c2, f2_grad[c2][k1]);
          }
        }
      } 
    }
    __syncthreads();


    for (int k=0; k<BLOCK_HW; k+=BLOCK_HW/CHANNEL_STRIDE) {
      int k1 = k + tid / CHANNEL_STRIDE;
      int h1 = h0 + k1 / BLOCK_W;
      int w1 = w0 + k1 % BLOCK_W;
      int c1 = tid % CHANNEL_STRIDE;

      scalar_t* fptr = &fmap1_grad[b][h1][w1][0];
      if (within_bounds(h1, w1, H1, W1))
        fptr[c+c1] += f1_grad[c1][k1];
    }
  }
}


/**
 * @brief start point for the correlation lookup operation
 *        calculates blocks and threads shape and launches kernel
 * 
 * @param fmap1   Tensor containing image 1 features (full resolution) of shape (batch, ht, wd, fdim)
 * @param fmap2   Tensor containing image 2 features (pooled) of shape (batch, ht/2**i, wd/2**i, fdim)
 * @param coords  Current correspondence estimation of shape (batch, 1, ht, wd, 2)
 * @param radius  Radius for correlation lookup
 * @return std::vector<torch::Tensor>   indexed correlation volume of shape (batch, 1, (2*r+1)**2, ht, wd)
 */
std::vector<torch::Tensor> corr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius)
{
  const auto B = coords.size(0);  // batch
  const auto N = coords.size(1);  // 1
  const auto H = coords.size(2);  // ht
  const auto W = coords.size(3);  // wd

  const auto rd = 2 * radius + 1; // lookup grid height and width
  auto opts = fmap1.options();

  // allocate storage for result
  auto corr = torch::zeros({B, N, rd*rd, H, W}, opts);
  
  // number of blocks
  // shape: (batch, (ht+4-1)/4, (wd+8-1)/8) = (batch, ceil(ht/4), ceil(ht/8))
  const dim3 blocks(B, (H+BLOCK_H-1)/BLOCK_H, (W+BLOCK_W-1)/BLOCK_W);
  
  // number of threads per block
  // shape: (4,8)
  const dim3 threads(BLOCK_H, BLOCK_W);

  // for each pixel in each image in the batch, one thread will be executed
  // launch kernel with packed accessors to allow for fast 32bit integer indexing
  corr_forward_kernel<float><<<blocks, threads>>>(
    fmap1.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    corr.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {corr};
}

std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius)
{
  const auto B = coords.size(0);
  const auto N = coords.size(1);

  const auto H1 = fmap1.size(1);
  const auto W1 = fmap1.size(2);
  const auto H2 = fmap2.size(1);
  const auto W2 = fmap2.size(2);
  const auto C = fmap1.size(3);

  auto opts = fmap1.options();
  auto fmap1_grad = torch::zeros({B, H1, W1, C}, opts);
  auto fmap2_grad = torch::zeros({B, H2, W2, C}, opts);
  auto coords_grad = torch::zeros({B, N, H1, W1, 2}, opts);
    
  const dim3 blocks(B, (H1+BLOCK_H-1)/BLOCK_H, (W1+BLOCK_W-1)/BLOCK_W);
  const dim3 threads(BLOCK_H, BLOCK_W);


  corr_backward_kernel<float><<<blocks, threads>>>(
    fmap1.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    corr_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    fmap1_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    fmap2_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    coords_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
    radius);

  return {fmap1_grad, fmap2_grad, coords_grad};
}