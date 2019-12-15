// kernel fusion

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

# define TILE_WIDTH 16

namespace mxnet
{
namespace op
{
__constant__ float filter[24*12*5*5];

__global__ void kernelfusion(int C, int H, int W, int K, int M, float* x, float* y){
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define k4d(i3, i2, i1, i0) filter[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    __shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];    
    __shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int column = blockIdx.x * TILE_WIDTH + tx;
    int numMatAColumn = C*K*K;
    float acc = 0.0;
    
    int num_iterations = (numMatAColumn - 1) / TILE_WIDTH + 1;
    for(int i = 0; i < num_iterations; i++){
        int temp_col = i*TILE_WIDTH + tx;
        int temp_row = i*TILE_WIDTH + ty;
        tileMatA[ty][tx] = 0;
        tileMatB[ty][tx] = 0;

        int W_m = row;
        int W_c = temp_col / (K*K);
        int W_h = (temp_col % (K*K)) / K;
        int W_w = (temp_col % (K*K)) % K;

        if(temp_col < numMatAColumn && row < M)
            tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
        else
            tileMatA[ty][tx] = 0;

        int X_b = b;
        int X_c = temp_row / (K*K);
        int X_p = (temp_row % (K*K)) / K;
        int X_q = (temp_row % (K*K)) % K;
        int X_h = column / W_out;
        int X_w = column % W_out;

        if(temp_row < numMatAColumn && column < H_out*W_out)
            tileMatB[ty][tx] = x4d(X_b, X_c, X_h+X_p, X_w+X_q);
        else
            tileMatB[ty][tx] = 0;
        
        __syncthreads();

        for(int q = 0; q < TILE_WIDTH; q++){
            acc += tileMatA[ty][q] * tileMatB[q][tx];
        __syncthreads();
        }

        int Y_b = b;
        int Y_m = row;
        int Y_h = column / W_out;
        int Y_w = column % W_out;
        if(row < M && column < W_out*H_out)
            y4d(Y_b, Y_m, Y_h, Y_w) = acc;
    }
#undef x4d
#undef y4d
#undef k4d
}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
 
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; //batch size
    const int M = y.shape_[1]; //output channel
    const int C = x.shape_[1]; //input channel
    const int H = x.shape_[2]; //input height
    const int W = x.shape_[3]; //input width
    const int K = w.shape_[3]; //filter size
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    const int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    const int Z = H_grid * W_grid;
    

    // Weight matrix (kernel values) in constant memory
    cudaMemcpyToSymbol(filter,w.dptr_,M*C*K*K*sizeof(float));
    
    dim3 gridDim(ceil(H_out*W_out / (1.0*TILE_WIDTH)),ceil(M / (1.0*TILE_WIDTH)), 1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH, 1);
    
    for(int b = 0; b < B; ++b) {
        kernelfusion<<<gridDim, blockDim, 0>>>(C, H, W, K, M, x.dptr_,y.dptr_);
        cudaDeviceSynchronize();
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif