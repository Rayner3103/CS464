# Keywords for different kernels
 - `__global__` void kernel_funct() {}; 

| Keyword      | Callable From | Executed On | Remarks |
| ------------ | ------------- | ----------- | ------- |
| `__host__`   | Host          | Host        | Default |
| `__global__` | Host / Device | Device      |         |
| `__device__` | Device        | Device      |         |


run kernel by the following:
```
kernel_funct <<< numBlocks, numThreadsPerBlock >>> (...);
```

# Thread indexes (.y shows y-direction; .z shows z-direction)
- `gridDim.x` : shows num of blocks in a grid
- `blockIdx.x` : shows position of block in a grid
- `blockDim.x` : shows num of thread in a block
- `threadIdx.x` : shows position of thread in a block

# Sync
`__syncthreads();` : synchronize the progress of all threads in the same block

# Error handling
```
cudaError_t err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
  // error handling
}
```
or 
```
// to get the latest error returned by cudaDeviceSynchronize()
cudaError_t cudaGetLastError();
```

# Multi-dimension Opn
Define the number of blocks and threads by:
```
dim3 numThreadsPerBlock(32, 32)
dim3 numBlocks(
  (width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
  (height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y
)
kernel_funct <<< numBlocks, numThreadsPerBlock >>> (...);
```


# Querying Available Rsrc
```
int devID = 0;
cudaDeviceProp* devProp;
cudaGetDeviceProperties(devProp, devID);
int maxThreadsPerBlock = devProp.maxThreadsPerBlock;
int maxThreadsPerMultiProcessor = devProp.maxThreadsPerMultiProcessor;
```


# Memory Assignment
| Variable Declarataion | Memory | Scope | Lifetime |
|-|-|-|-|
|`__device__ int globalVar; `|global|grid|application|
|`__device__ __constant__ int constantVar;`|constant| grid|application|
|`__device__ __shared__ sharedVar;`|shared|block|block|
|`int localVar`|register|thread|thread|
|`int localArr[N]`|global|thread|thread|

Dynamically allocating shared memory: `extern __shared__ A_s[];`
- Required config: `kernel <<< numBlocks, numThreadsPerBlock, smemPerBlock >>>(...);`

