// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 32


//@@ insert code here

//*** Kernel 1
__global__ void convertFloatToChar(float *input, unsigned char* output, int width, int height) {  
  
  int tw = blockIdx.x * blockDim.x + threadIdx.x;
  int th = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(tw < width && th < height) {
    int tid = blockIdx.z * width * height + th * width + tw;   //Here blockIdx.z is the channel index (r/g/b)
    output[tid] = (unsigned char)((HISTOGRAM_LENGTH - 1) * input[tid]);
    
  }
  
}

//*** Kernel 2
__global__ void convertRGBToGray(unsigned char *input, unsigned char* output, int width, int height) {
  
  int tw = blockIdx.x * blockDim.x + threadIdx.x;
  int th = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(tw < width && th < height) {
    int tid = th * width + tw;
    auto r = input[3 * tid];
    auto g = input[3 * tid + 1];
    auto b = input[3 * tid + 2];
    output[tid] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
        //printf("gray value is %d!!\n", output[tid]);
  }
  
}

//*** Kernel 3
__global__ void computeHistogram(unsigned char* input, unsigned int* output, int width, int height) {
  
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  
  //Initialize histo_private
  int tid = blockDim.x * threadIdx.y + threadIdx.x;
  if(tid < HISTOGRAM_LENGTH)
    histo_private[tid] = 0;
  
  __syncthreads();
  
  int tw = blockIdx.x * blockDim.x + threadIdx.x;
  int th = blockIdx.y * blockDim.y + threadIdx.y;
  if(tw < width && th < height) {
    int i = th * width + tw;
    auto cur_intensity = input[i];
    atomicAdd(&(histo_private[cur_intensity]), 1);
  }
  
  __syncthreads();
  
  if(tid < HISTOGRAM_LENGTH) {
    atomicAdd(&output[tid], histo_private[tid]);
    //if(tid == 128)
      //printf("Histogram value at %d is: %d!!\n", tid, output[tid]);  
  }
}

//*** Kernel 4
__global__ void computeCDF(unsigned int* input, float* output, int width, int height) {
  
  //Here, only have one block with blockDim = HISTOGRAM_LENGTH*1*1
  
  __shared__ unsigned int cdf_private[HISTOGRAM_LENGTH];
  
  int tid = threadIdx.x;
  
  cdf_private[tid] = input[tid];
  
  for(unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if(index < HISTOGRAM_LENGTH) {
      cdf_private[index] += cdf_private[index - stride];
    }
  }

  // Second scan half
  for(int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * 2 * stride - 1;

    if (index + stride < HISTOGRAM_LENGTH) {
      cdf_private[index + stride] += cdf_private[index];
    }
  }

  __syncthreads();

  float img_size = (float)(width * height);
  
  if (tid < HISTOGRAM_LENGTH) {
    output[tid] = cdf_private[tid] / img_size;
    //printf("CRF is %f!!\n", output[tid]);    
  }
   
}

//*** Kernel 5
__global__ void histogramEqualization(float* input, unsigned char* output, int width, int height) {
  
  //input is deviceImageCDF, output is deviceGreyImageData
  
  int tw = blockIdx.x * blockDim.x + threadIdx.x;
  int th = blockIdx.y * blockDim.y + threadIdx.y;
  
  /*
    Correct intensity
    cdf[val] : val should be grey intensity in original input image
  */
  if(tw < width && th < height) {

    int tid = blockIdx.z * height * width + th * width + tw;   //Here blockIdx.z is the channel index (r/g/b)
    unsigned char temp = output[tid];
    
    //Compute the minimum value of the CDF: should be deviceImageCDF[0]
    float correct_val = (HISTOGRAM_LENGTH - 1) * (input[temp] - input[0]) / (1.0 - input[0]);
    
    output[tid] = (unsigned char)(min(max(correct_val, 0.0), 255.0));
    
    if(tid == 0)
      printf("CRF Min is: %f!!\n", output[0]);
      //printf("Output image value is: %d at %d!!\n", output[tid], tid);
  }
   
}

//*** Kernel 6
__global__ void convertCharToFloat(unsigned char* input, float* output, int width, int height) {  
  
  int tw = blockIdx.x * blockDim.x + threadIdx.x;
  int th = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(tw < width && th < height) {
    int tid = blockIdx.z * width * height + th * width + tw;   //Here blockIdx.z is the channel index (r/g/b)
    output[tid] = (float)(input[tid] / (float)(HISTOGRAM_LENGTH - 1));
  }
  
}



int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels; // Should be 3 for rbg
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
  //@@ insert code here
  
  //Self-defined Memory
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceRGBImageData;
  unsigned char *deviceGreyImageData;
  unsigned int *deviceHistogram;
  float *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  printf("Input image size is: ==Width: %d==, ==Heights: %d!!\n", imageWidth, imageHeight);
  //@@ Insert more code here
  wbTime_start(GPU, "Doing GPU memory allocation");
  
  cudaMalloc((void **) &deviceInputImageData, imageChannels * imageHeight * imageWidth * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageChannels * imageHeight * imageWidth * sizeof(float));
  cudaMalloc((void **) &deviceRGBImageData, imageChannels * imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGreyImageData, imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  //Need to initiate with all 0
  cudaMemset((void **)&deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  //Copy input from CPU memory to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  //Declare DimGrid, DimBlock
  dim3 dimGrid, dimBlock;

  //***Kernel 1: convert input data from float to unsigned char (unsigned char ranges from 0 ~ 255, float 0.0 ~ 1.0)
  dimGrid = dim3(ceil(1.0 * imageWidth/TILE_WIDTH), ceil(1.0 * imageHeight/TILE_WIDTH), imageChannels);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  convertFloatToChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceRGBImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  //***Kernel 2: convert input image from RGB to grey scale
  dimGrid = dim3(ceil(1.0 * imageWidth/TILE_WIDTH), ceil(1.0 * imageHeight/TILE_WIDTH), 1);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  convertRGBToGray<<<dimGrid, dimBlock>>>(deviceRGBImageData, deviceGreyImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  //***Kernel 3: Compute the histogram of grayImage
  dimGrid = dim3(ceil(1.0 * imageWidth/TILE_WIDTH), ceil(1.0 * imageHeight/TILE_WIDTH), 1);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  computeHistogram<<<dimGrid, dimBlock>>>(deviceGreyImageData, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();  
  
  //***Kernel 4: Compute the Cumulative Distribution Function(CDF) of histogram (only have 256 datapoints)
  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  computeCDF<<<dimGrid, dimBlock>>>(deviceHistogram, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();  
  
  
  //***Kernel 5: Define and Apply the histogram equalization function to grey image
  dimGrid = dim3(ceil(1.0 * imageWidth/TILE_WIDTH), ceil(1.0 * imageHeight/TILE_WIDTH), imageChannels);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  histogramEqualization<<<dimGrid, dimBlock>>>(deviceImageCDF, deviceRGBImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();  

  //***Kernel 6: Cast back to float
  dimGrid = dim3(ceil(1.0 * imageWidth/TILE_WIDTH), ceil(1.0 * imageHeight/TILE_WIDTH), imageChannels);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  convertCharToFloat<<<dimGrid, dimBlock>>>(deviceRGBImageData, deviceOutputImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();  

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  
  return 0;
}
    