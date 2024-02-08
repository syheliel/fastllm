#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-rocm.hiph"
#include "fastllm.h"

#define DEBUG false
#define DEBUG_PRINT(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)

#define checkHipErrors( a ) do { \
    if (hipSuccess != (a)) { \
    fprintf(stderr, "Hip runtime error in line %s:%d \
    : %s \n", __FILE__, __LINE__, hipGetErrorString(hipGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);

inline void gpuAssert(hipblasStatus_t code, const char *file, int line, bool abort=true) 
{
    if (code != HIPBLAS_STATUS_SUCCESS)
    {
        switch (code) {
            case HIPBLAS_STATUS_NOT_INITIALIZED:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_NOT_INITIALIZED in %s:%d \n", file, line);
            break; 
            
            case HIPBLAS_STATUS_ALLOC_FAILED:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_ALLOC_FAILED in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_INVALID_VALUE:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_INVALID_VALUE in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_ARCH_MISMATCH:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_ARCH_MISMATCH in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_MAPPING_ERROR:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_MAPPING_ERROR in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_EXECUTION_FAILED:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_EXECUTION_FAILED in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_INTERNAL_ERROR:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_INTERNAL_ERROR in %s:%d \n", file, line); 
            break; 

            case HIPBLAS_STATUS_NOT_SUPPORTED:
            fprintf(stderr, "cuBLAS Error: HIPBLAS_STATUS_NOT_SUPPORTED in %s:%d \n", file, line); 
            break; 

        }
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static std::map<int, hipblasHandle_t> s_fastllmCublasHandleMap;
hipblasHandle_t getFastllmHipblasHandle() {
    int id = -1;
    hipGetDevice(&id);
    auto it = s_fastllmCublasHandleMap.find(id);
    if (it != s_fastllmCublasHandleMap.end()) {
        return it->second;
    }
    hipblasHandle_t handler = nullptr;
    auto stat = hipblasCreate(&handler);

    if (stat != HIPBLAS_STATUS_SUCCESS) {
        checkHipErrors(stat);
    } else {
        s_fastllmCublasHandleMap[id] = handler;
    }

    return handler;
}

void DeviceSync() {
    //rocmDeviceSynchronize();
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

__global__ void FastllmRocmFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void FastllmRocmInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(scales[idx / per] * ((float)a[idx] - zeros[idx / per]));
    }
}

__global__ void FastllmRocmInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        if (idx % 2 == 1) {
            b[idx] = __float2half(scales[idx / per] * (a[idx / 2] & 0xF) + mins[idx / per]);
        } else {
            b[idx] = __float2half(scales[idx / per] * (a[idx / 2] >> 4) + mins[idx / per]);
        }
    }
}

__global__ void FastllmRocmHalf2FlotaKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void FastllmRocmBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

__global__ void FastllmGeluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void FastllmSiluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

__global__ void FastllmSwigluKernel(float* a, float *b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float x = a[id], y = a[id + mid];
        b[idx] = (x / (1.0 + expf(-x))) * y;
    }
}

__global__ void FastllmMulKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMulBatchKernel(float** pointer, int batch, float v) {
    float *input = pointer[blockIdx.x];
    float *output = pointer[blockIdx.x + batch];
    int len = (int)((unsigned long long)pointer[blockIdx.x + batch * 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        output[i] = input[i] * v;
    }
}

__global__ void FastllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

__global__ void FastllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(float* a, float *b, float maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (b[i] > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAlibiMaskKernel(float* a, float *b, float maskValue, int n, int m, int spn, int spm, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    float now = b[om];
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        int idi = i / spm, idj = i % spm;
        if (idj <= spm - spn + idi) {
            a[o * spatial + i] += now * idj;
        } else {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmApplyLognAttnKernel(float* input, float *logn, float *pos, int b, int s, int spatial) {
    int ob = blockIdx.x / s;
    int os = blockIdx.x % s;
    int o = ob * s + os;
    int idx = threadIdx.x;
    int curPos = (int)(pos[0]);

    float v = logn[os + curPos];
    float *curInput = input + o * spatial;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        curInput[i] = curInput[i] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m, col = blockIdx.x % m;
    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

__global__ void FastllmPermuteKernel(float *dst, float *ori, int *temp, int axisLen, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        int old = 0;
        int idx = i;
        for (int j = 0; j < axisLen; ++j) {
            int order = temp[j];
            old += (idx / temp[j + 2 * axisLen]) * temp[order + 1 * axisLen];
            idx %= temp[j + 2 * axisLen];
        }
        dst[i] = ori[old];
    }
}

__global__ void FastllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}

__global__ void FastllmNearlyRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
/*
    int len = data.dims[0], bs = data.dims[1];
    int spatial = data.Count(2);
    int n = data.dims[2], m = data.dims[3];
    int stride = (int)sinData.dims[1];
    for (int l = 0; l < len; l++) {
        for (int b = 0; b < bs; b++) {
            int index = (int) ((float *) positionIds.cpuData)[(b * 2) * positionIds.dims.back() + l];
            float *sin = ((float*)sinData.cpuData) + stride * index;
            float *cos = ((float*)cosData.cpuData) + stride * index;
            float *d = (float *) data.cpuData + (l * bs + b) * spatial;
            for (int i = 0; i < n; i++) {
                int j = 0;
                for (; j < rotaryDim; j += 2) {
                    float a = d[j], b = d[j + 1];
                    d[j] = a * cos[j / 2] - b * sin[j / 2];
                    d[j + 1] = a * sin[j / 2] + b * cos[j / 2];
                }
                d += m;
            }
        }
    }
*/
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * 2 * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + 1];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + 1] = va * curSin + vb * curCos;
}

__global__ void FastllmRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                              int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n) / 2;
    int l = o / bs;
    int b = o % bs;
    int part = (blockIdx.x / n) % 2;
    int j = threadIdx.x;
    int index = (int) (positionIds[(b * 2 + part) * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + part * m / 2 + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 4];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 4] = va * curSin + vb * curCos;
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    unsigned int per = (channels / THREAD_PER_BLOCK);
    unsigned int id = threadIdx.x * per;
    unsigned int len = per;
    if (tid == blockDim.x - 1) {
        len += (channels - per * THREAD_PER_BLOCK);
    }
    float maxValue = input[id];
    for (int i = 0; i < len; i++) {
        maxValue = max(maxValue, input[id + i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        maxV = sdata[0];
    }
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = 0; i < len; i++) {
        output[id + i] = exp(input[id + i] - maxV);
        sum += output[id + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.1;
        }
    }
    __syncthreads();

    for (int i = 0; i < len; i++) {
        output[id + i] /= sdata[0];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((float*)pointer[o * 3], (float*)pointer[o * 3 + 1],
                                                       (int)((size_t)pointer[o * 3 + 2]));
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] * scale * weight[i]);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(float *input, float *gamma, float *beta, float *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] - mean) / var * gamma[i] + beta[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    maxData[tid] = -1e100;
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        if (inputData[j] > maxData[tid]) {
            maxData[tid] = inputData[j];
            idData[tid] = j;
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (maxData[tid] < maxData[tid + s]) {
                maxData[tid] = maxData[tid + s];
                idData[tid] = idData[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputData[0] = idData[0];
        outputData[1] = maxData[0];
    }
}

template <int NBlock, int MBlock, int KBlock>
__global__ void FastllmRocmBaseGemmKernelInt8(float *A, uint8_t *B, float *C,
                                              float *bias, float *scales, uint8_t *zeros,
                                              int n, int m, int k) {
    int nStart = blockIdx.x * NBlock, nEnd = nStart + NBlock;
    int kStart = blockIdx.y * KBlock, kEnd = kStart + KBlock;

    int id = kStart + threadIdx.x;
    __shared__ float shareA[NBlock * MBlock];
    __shared__ float shareB[KBlock * MBlock];
    float localSum[NBlock] = {0.0f};
    uint8_t zero = zeros[id];
    int idx = threadIdx.x >> 3;
    int idy = threadIdx.x & 7;
    for (int l = 0; l < m; l += MBlock) {
        if (threadIdx.x < MBlock) {
            for (int i = nStart; i < nEnd; i++) {
                if (i < n && l + threadIdx.x < m) {
                    shareA[(i - nStart) * MBlock + threadIdx.x] = A[i * m + l + threadIdx.x];
                } else {
                    shareA[(i - nStart) * MBlock + threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < MBlock) {
            for (int i = kStart; i < kEnd; i++) {
                if (i < k && l + threadIdx.x < m) {
                    shareB[(i - kStart) * MBlock + threadIdx.x] = B[i * m + l + threadIdx.x];
                } else {
                    shareB[(i - kStart) * MBlock + threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int mStart = 0; mStart < MBlock; mStart += 4) {
            float curA[32] = {0.0f}, curB[32] = {0.0f};
            for (int i = 0; i < 8; i++) {
                for (int x = l + mStart; x < l + mStart + 4 && x < m; x++) {
                    curA[i * 4 + (x - l - mStart)] = shareA[(idx * 8 + i) * MBlock + (x - l)];
                }
            }
            for (int j = 0; j < 4; j++) {
                zero = zeros[kStart + (idy * 4 + j)];
                for (int x = l + mStart; x < l + mStart + 4 && x < m; x++) {
                    curB[j * 4 + (x - l - mStart)] = shareB[(idy * 4 + j) * MBlock + (x - l)] - zero;
                }
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    int cur = i * 4 + j;
                    localSum[cur] += curA[i * 4 + 0] * curB[j * 4 + 0];
                    localSum[cur] += curA[i * 4 + 1] * curB[j * 4 + 1];
                    localSum[cur] += curA[i * 4 + 2] * curB[j * 4 + 2];
                    localSum[cur] += curA[i * 4 + 3] * curB[j * 4 + 3];
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            if ((nStart + idx * 8 + i) < n && (kStart + idy * 4 + j) < k) {
                C[(nStart + idx * 8 + i) * k + (kStart + idy * 4 + j)] =
                        localSum[i * 4 + j] * scales[(kStart + idy * 4 + j)] + bias[(kStart + idy * 4 + j)];
            }
        }
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp32Kernel2(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp16Kernel2(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (float)B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt8Kernel2(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 读入fdata
    /*for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        fdata[i] = A[i];
    }
    __syncthreads();*/

    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (B[p * m + i] - zero);
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int SINGLE_COMPUTE, int REDUCE_NUMBER>
__global__ void FastllmGemvInt8Kernel1(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[REDUCE_NUMBER];
    unsigned int tid = threadIdx.x;

    int part = m / REDUCE_NUMBER;
    // 1. 每个线程计算一部分
    for (int p = 0; p < part; p++) {
        float v[SINGLE_COMPUTE];
        for (int i = 0; i < SINGLE_COMPUTE; i++) {
            v[i] = A[p * REDUCE_NUMBER + tid * SINGLE_COMPUTE + i];
        }
        for (int i = 0; i < SINGLE_COMPUTE / part; i++) {
            float sum = 0;
            int colId = (blockIdx.x * SINGLE_COMPUTE / part + i);
            if (colId >= k) {
                sdata[i * (m / SINGLE_COMPUTE) + p * (REDUCE_NUMBER / SINGLE_COMPUTE) + tid] = 0;
                continue;
            }
            int id = colId * m + p * REDUCE_NUMBER + tid * SINGLE_COMPUTE;
            uint8_t zero = zeros[colId];
            for (int j = 0; j < SINGLE_COMPUTE; j++) {
                sum += v[j] * (B[id + j] - zero);
            }
            sdata[i * (m / SINGLE_COMPUTE) + p * (REDUCE_NUMBER / SINGLE_COMPUTE) + tid] = sum;
            __syncthreads();
        }
    }

    // 2. 求和
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int i = 0; i < SINGLE_COMPUTE; i++) {
                sdata[i * THREAD_PER_BLOCK + tid] += sdata[i * THREAD_PER_BLOCK + tid + s];
            }
        }
        __syncthreads();
    }

    // 3. 写回结果
    if (tid == 0) {
        for (int i = 0; i < SINGLE_COMPUTE / part; i++) {
            int id = blockIdx.x * SINGLE_COMPUTE / part  + i;
            if (id >= k) {
                continue;
            }
            float sum = 0;
            for (int p = 0; p < part; p++) {
                sum += sdata[(i * part + p) * THREAD_PER_BLOCK];
            }
            C[id] = sum * scales[id] + bias[id];
        }
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4Kernel2(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += (A[i * 2] * ((now >> 4) - zero) + A[i * 2 + 1] * ((now & 15) - zero));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel2(float *A, uint8_t *B, float *C,
                                             float *bias, float *scales, float *mins,
                                             int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        float minv = mins[p] / scales[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += (A[i * 2] * (minv + (now >> 4)) + A[i * 2 + 1] * (minv + (now & 15)));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSplitBatchKernel(uint8_t *input, uint8_t **outputs, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = input + oid * channels * inner + bid * inner;
    uint8_t *curOutput = outputs[bid] + oid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmCatBatchKernel(uint8_t **inputs, uint8_t *output, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = inputs[bid] + oid * inner;
    uint8_t *curOutput = output + oid * channels * inner + bid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = input0[a * input0Stride + l + x];
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = input1[b * input1Stride + l + x];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = l + x < m ? input0[a * input0Stride + l + x] : 0;
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = l + x < m ? input1[(l + x) * input1Stride + b] : 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    //int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l * input1Stride];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionKernel(float *qd, float *kd, float *vd, float *maskd, float *od,
                                       float scale, int q1, int q2, int k1, int v2,
                                       int group, int qstride, int kstride, int vstride, int ostride,
                                       float *qk, float *temp) {
    int o = blockIdx.x;
    qd += o * qstride;
    kd += (o / group) * kstride;
    vd += (o / group) * vstride;
    od += o * ostride;
    qk += o * k1;
    temp += o * k1;
    for (int i = 0; i < q1; i++) {
        for (int j = threadIdx.x; j < k1; j += THREAD_PER_BLOCK) {
            if (maskd && maskd[i * k1 + j] > 0.99) {
                qk[j] = -10000;
                continue;
            }
            float sum = 0.0f;
            float *tempQd = qd + i * q2, *tempKd = kd + j * q2;
            for (int l = 0; l < q2; l++) {
                sum += tempQd[l] * tempKd[l];
            }
            qk[j] = sum * scale;
        }
        __syncthreads();
        FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (qk, temp, k1);
        __syncthreads();
        for (int j = threadIdx.x; j < v2; j += THREAD_PER_BLOCK) {
            float *curInput1 = vd + j;
            float sum = 0.0;
            for (int l = 0; l < k1; l++) {
                sum += temp[l] * curInput1[l * v2];
            }
            od[i * v2 + j] = sum;
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionBatchKernel(float** pointer, float scale, int group) {
    const int params = 16;
    int id = blockIdx.x;
    float *qd = (float*) pointer[id * params + 0];
    float *kd = (float*) pointer[id * params + 1];
    float *vd = (float*) pointer[id * params + 2];
    float *maskd = (float*) pointer[id * params + 3];
    float *od = (float*) pointer[id * params + 4];
    int q1 = (int)(unsigned long long)pointer[id * params + 5];
    int q2 = (int)(unsigned long long)pointer[id * params + 6];
    int k1 = (int)(unsigned long long)pointer[id * params + 7];
    int v2 = (int)(unsigned long long)pointer[id * params + 8];
    int qstride = (int)(unsigned long long)pointer[id * params + 9];
    int kstride = (int)(unsigned long long)pointer[id * params + 10];
    int vstride = (int)(unsigned long long)pointer[id * params + 11];
    int ostride = (int)(unsigned long long)pointer[id * params + 12];
    float *qk = (float*)pointer[id * params + 13];
    float *temp = (float*)pointer[id * params + 14];
    int q0 = (int)(unsigned long long)pointer[id * params + 15];

    for (int o = 0; o < q0; o++) {
        qd += o * qstride;
        kd += (o / group) * kstride;
        vd += (o / group) * vstride;
        od += o * ostride;
        qk += o * k1;
        temp += o * k1;

        for (int i = 0; i < q1; i++) {
            for (int j = threadIdx.x; j < k1; j += THREAD_PER_BLOCK) {
                if (maskd && maskd[i * k1 + j] > 0.99) {
                    qk[j] = -10000;
                    continue;
                }
                float sum = 0.0f;
                float *tempQd = qd + i * q2, *tempKd = kd + j * q2;
                for (int l = 0; l < q2; l++) {
                    sum += tempQd[l] * tempKd[l];
                }
                qk[j] = sum * scale;
            }
            __syncthreads();
            FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(qk, temp, k1);
            __syncthreads();
            for (int j = threadIdx.x; j < v2; j += THREAD_PER_BLOCK) {
                float *curInput1 = vd + j;
                float sum = 0.0;
                for (int l = 0; l < k1; l++) {
                    sum += temp[l] * curInput1[l * v2];
                }
                od[i * v2 + j] = sum;
            }
            __syncthreads();
        }

        qd -= o * qstride;
        kd -= (o / group) * kstride;
        vd -= (o / group) * vstride;
        od -= o * ostride;
        qk -= o * k1;
        temp -= o * k1;
    }
}

void *FastllmRocmPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::ROCM) {
        ret = (void*)input.rocmData;
    } else {
        ret = (void*)(input.expansionBytes); // this code is really strange
        // checkHipErrors(hipMalloc(&ret, input.expansionBytes));
        checkHipErrors(hipMemcpy(ret, input.cpuData, input.expansionBytes, hipMemcpyHostToDevice));
    }
    return ret;
}

void FastllmRocmFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::ROCM) {
        FastllmRocmFree(data);
    }
}

void *FastllmRocmPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::ROCM) {
        ret = (float*)output.rocmData;
    } else {
        ret = (float*)FastllmRocmMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmRocmFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::ROCM) {
        checkHipErrors(hipMemcpy(output.cpuData, data, output.expansionBytes, hipMemcpyDeviceToHost));
        FastllmRocmFree(data);
    }

    DeviceSync();
}

bool FastllmRocmMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.rocmData == nullptr || weight.extraRocmData.size() == 0) {
        float *rocmScales;
        checkHipErrors(hipMalloc(&rocmScales, k * sizeof(float)));
        checkHipErrors(hipMemcpy(rocmScales, weight.scales.data(), k * sizeof(float), hipMemcpyHostToDevice));
        weight.extraRocmData.push_back((void*)rocmScales);

        uint8_t *rocmZeropoints;
        checkHipErrors(hipMalloc(&rocmZeropoints, k));
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        checkHipErrors(hipMemcpy(rocmZeropoints, zeropoints, k, hipMemcpyHostToDevice));
        delete[] zeropoints;
        weight.extraRocmData.push_back((void*)rocmZeropoints);

        float *rocmBiasData;
        checkHipErrors(hipMalloc(&rocmBiasData, k * sizeof(float)));
        if (bias.dims.size() > 0) {
            checkHipErrors(hipMemcpy(rocmBiasData, (uint8_t*)bias.rocmData, k * sizeof(float), hipMemcpyDeviceToDevice));
        } else {
            hipMemset(rocmBiasData, 0, k * sizeof(float));
        }
        weight.extraRocmData.push_back((void*)rocmBiasData);
    }

    float *rocmScales = (float*)weight.extraRocmData[0];
    uint8_t *rocmZeropoints = (uint8_t*)weight.extraRocmData[1];
    float *rocmBiasData = (float*)weight.extraRocmData[2];

    float *rocmInput = (float*)FastllmRocmPrepareInput(input);
    float *rocmOutput = (float*)FastllmRocmPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        half *rocmFp16Input, *rocmFp16Output, *rocmFp16Weight;
        rocmFp16Input = (half *) FastllmRocmMalloc(n * m * sizeof(half));
        rocmFp16Output = (half *) FastllmRocmMalloc(n * k * sizeof(half));
        rocmFp16Weight = (half *) FastllmRocmMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        hipblasDatatype_t AType = HIPBLAS_R_16F, BType = HIPBLAS_R_16F, CType = HIPBLAS_R_16F, ComputeType = HIPBLAS_R_16F;
        hipblasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmRocmFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmFp16Input, len);

        len = k * m;
        FastllmRocmInt82HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight.rocmData,
                                                                                         rocmScales,
                                                                                         rocmZeropoints,
                                                                                         rocmFp16Weight, len, m);

        status = hipblasGemmEx(fastllmCublasHandle,
                              HIPBLAS_OP_T, HIPBLAS_OP_N,
                              k, n, m,
                              &h_alpha, rocmFp16Weight, AType,
                              m, rocmFp16Input, BType,
                              m, &h_beta,
                              rocmFp16Output, CType,
                              k, ComputeType, static_cast<hipblasGemmAlgo_t>(HIPBLAS_GEMM_DEFAULT));
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status);
            exit(0);
        }

        len = n * k;
        FastllmRocmHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(rocmFp16Output, rocmOutput, len);
        FastllmRocmBiasKernel <<< n, 256 >>> (rocmOutput, rocmBiasData, k);

        FastllmRocmFree(rocmFp16Input);
        FastllmRocmFree(rocmFp16Output);
        FastllmRocmFree(rocmFp16Weight);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt8Kernel2<256, 1> <<< k, 256 >>>(rocmInput + i * m,
                                                          (uint8_t *) weight.rocmData,
                                                          rocmOutput + i * k,
                                                          rocmBiasData,
                                                          rocmScales,
                                                          rocmZeropoints,
                                                          m, k);
        }
    }
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.rocmData == nullptr || weight.extraRocmData.size() == 0) {
        float *rocmScales;
        checkHipErrors(hipMalloc(&rocmScales, k * sizeof(float)));
        checkHipErrors(hipMemcpy(rocmScales, weight.scales.data(), k * sizeof(float), hipMemcpyHostToDevice));
        weight.extraRocmData.push_back((void*)rocmScales);

        uint8_t *rocmZeropoints;
        checkHipErrors(hipMalloc(&rocmZeropoints, k));
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        checkHipErrors(hipMemcpy(rocmZeropoints, zeropoints, k, hipMemcpyHostToDevice));
        delete[] zeropoints;
        weight.extraRocmData.push_back((void*)rocmZeropoints);

        float *rocmBiasData;
        checkHipErrors(hipMalloc(&rocmBiasData, k * sizeof(float)));
        if (bias.dims.size() > 0) {
            checkHipErrors(hipMemcpy(rocmBiasData, (uint8_t*)bias.rocmData, k * sizeof(float), hipMemcpyDeviceToDevice));
        } else {
            hipMemset(rocmBiasData, 0, k * sizeof(float));
        }
        weight.extraRocmData.push_back((void*)rocmBiasData);
    }

    float *rocmScales = (float*)weight.extraRocmData[0];
    uint8_t *rocmZeropoints = (uint8_t*)weight.extraRocmData[1];
    float *rocmBiasData = (float*)weight.extraRocmData[2];

    float *rocmInput = (float*)FastllmRocmPrepareInput(input);
    float *rocmOutput = (float*)FastllmRocmPrepareOutput(output);

    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel2<256, 1> <<< k, 256 >>>(rocmInput + i * m,
                                                      (uint8_t *) weight.rocmData,
                                                      rocmOutput + i * k,
                                                      rocmBiasData,
                                                      rocmScales,
                                                      rocmZeropoints,
                                                      m, k);
    }
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.rocmData == nullptr || weight.extraRocmData.size() == 0) {
        float *rocmScales;
        checkHipErrors(hipMalloc(&rocmScales, k * sizeof(float)));
        checkHipErrors(hipMemcpy(rocmScales, weight.scales.data(), k * sizeof(float), hipMemcpyHostToDevice));
        weight.extraRocmData.push_back((void*)rocmScales);

        float *rocmMins;
        checkHipErrors(hipMalloc(&rocmMins, k * sizeof(float)));
        float *mins = new float[k];
        for (int i = 0; i < k; i++) {
            mins[i] = weight.perChannelsConfigs[i].min;
        }
        checkHipErrors(hipMemcpy(rocmMins, mins, k * sizeof(float), hipMemcpyHostToDevice));
        delete[] mins;
        weight.extraRocmData.push_back((void*)rocmMins);

        float *rocmBiasData;
        checkHipErrors(hipMalloc(&rocmBiasData, k * sizeof(float)));
        if (bias.dims.size() > 0) {
            checkHipErrors(hipMemcpy(rocmBiasData, (uint8_t*)bias.rocmData, k * sizeof(float), hipMemcpyDeviceToDevice));
        } else {
            hipMemset(rocmBiasData, 0, k * sizeof(float));
        }
        weight.extraRocmData.push_back((void*)rocmBiasData);
    }

    float *rocmScales = (float*)weight.extraRocmData[0];
    float *rocmMins = (float*)weight.extraRocmData[1];
    float *rocmBiasData = (float*)weight.extraRocmData[2];

    float *rocmInput = (float*)FastllmRocmPrepareInput(input);
    float *rocmOutput = (float*)FastllmRocmPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        half *rocmFp16Input, *rocmFp16Output, *rocmFp16Weight;
        rocmFp16Input = (half *) FastllmRocmMalloc(n * m * sizeof(half));
        rocmFp16Output = (half *) FastllmRocmMalloc(n * k * sizeof(half));
        rocmFp16Weight = (half *) FastllmRocmMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        hipblasDatatype_t AType = HIPBLAS_R_16F, BType = HIPBLAS_R_16F, CType = HIPBLAS_R_16F, ComputeType = HIPBLAS_R_16F;
        hipblasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmRocmFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmFp16Input,
                                                                                          len);

        len = k * m;
        FastllmRocmInt42HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t *) weight.rocmData,
                                                                                         rocmScales,
                                                                                         rocmMins,
                                                                                         rocmFp16Weight, len, m);

        status = hipblasGemmEx(fastllmCublasHandle,
                              HIPBLAS_OP_T, HIPBLAS_OP_N,
                              k, n, m,
                              &h_alpha, rocmFp16Weight, AType,
                              m, rocmFp16Input, BType,
                              m, &h_beta,
                              rocmFp16Output, CType,
                              k, ComputeType, static_cast<hipblasGemmAlgo_t>(HIPBLAS_GEMM_DEFAULT));
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status)
        }

        len = n * k;
        FastllmRocmHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(rocmFp16Output, rocmOutput,
                                                                                           len);
        FastllmRocmBiasKernel <<< n, 256 >>>(rocmOutput, rocmBiasData, k);

        FastllmRocmFree(rocmFp16Input);
        FastllmRocmFree(rocmFp16Output);
        FastllmRocmFree(rocmFp16Weight);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt4NoZeroKernel2<256, 1> <<< k, 256 >>>(rocmInput + i * m,
                                                                (uint8_t *) weight.rocmData,
                                                                rocmOutput + i * k,
                                                                rocmBiasData,
                                                                rocmScales,
                                                                rocmMins,
                                                                m, k);
        }
    }
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.rocmData == nullptr || weight.extraRocmData.size() == 0) {
        float *rocmBiasData;
        checkHipErrors(hipMalloc(&rocmBiasData, k * sizeof(float)));
        if (bias.dims.size() > 0) {
            checkHipErrors(hipMemcpy(rocmBiasData, (uint8_t*)bias.rocmData, k * sizeof(float), hipMemcpyDeviceToDevice));
        } else {
            hipMemset(rocmBiasData, 0, k * sizeof(float));
        }
        weight.extraRocmData.push_back((void*)rocmBiasData);
    }

    float *rocmBiasData = (float*)weight.extraRocmData[0];
    float *rocmInput = (float*)FastllmRocmPrepareInput(input);
    float *rocmOutput = (float*)FastllmRocmPrepareOutput(output);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        //rocmDeviceSynchronize();
        hipblasDatatype_t AType = HIPBLAS_R_32F, BType = HIPBLAS_R_32F, CType = HIPBLAS_R_32F, ComputeType = HIPBLAS_R_32F;
        hipblasStatus_t status;

        status = hipblasGemmEx(fastllmCublasHandle,
                              HIPBLAS_OP_T, HIPBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weight.rocmData, AType,
                              m, rocmInput, BType,
                              m, &h_beta,
                              rocmOutput, CType,
                              k, ComputeType, static_cast<hipblasGemmAlgo_t>(HIPBLAS_GEMM_DEFAULT));
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status);
            FastllmRocmFinishInput(input, rocmInput);
            FastllmRocmFinishOutput(output, rocmOutput);
            exit(0);
        }

        FastllmRocmBiasKernel <<< n, 256 >>> (rocmOutput, (float*)weight.extraRocmData[0], k);
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(rocmInput, (float *) weight.rocmData, rocmOutput, rocmBiasData, m, k);
    }

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.rocmData == nullptr || weight.extraRocmData.size() == 0) {
        float *rocmBiasData;
        checkHipErrors(hipMalloc(&rocmBiasData, k * sizeof(float)));
        if (bias.dims.size() > 0) {
            checkHipErrors(hipMemcpy(rocmBiasData, (uint8_t*)bias.rocmData, k * sizeof(float), hipMemcpyDeviceToDevice));
        } else {
            hipMemset(rocmBiasData, 0, k * sizeof(float));
        }
        weight.extraRocmData.push_back((void*)rocmBiasData);
    }
    float *rocmBiasData = (float*)weight.extraRocmData[0];
    float *rocmInput = (float*)FastllmRocmPrepareInput(input);
    float *rocmOutput = (float*)FastllmRocmPrepareOutput(output);

    if (n > 1) {
        half *rocmFp16Input, *rocmFp16Output;
        rocmFp16Input = (half *) FastllmRocmMalloc(n * m * sizeof(half));
        rocmFp16Output = (half *) FastllmRocmMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        //rocmDeviceSynchronize();
        hipblasDatatype_t AType = HIPBLAS_R_16F, BType = HIPBLAS_R_16F, CType = HIPBLAS_R_16F, ComputeType = HIPBLAS_R_16F;
        hipblasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmRocmFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmFp16Input,
                                                                                          len);

        status = hipblasGemmEx(fastllmCublasHandle,
                              HIPBLAS_OP_T, HIPBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.rocmData, AType,
                              m, rocmFp16Input, BType,
                              m, &h_beta,
                              rocmFp16Output, CType,
                              k, ComputeType, static_cast<hipblasGemmAlgo_t>(HIPBLAS_GEMM_DEFAULT));
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status);
            exit(0);
        }

        len = n * k;
        FastllmRocmHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(rocmFp16Output, rocmOutput,
                                                                                           len);
        FastllmRocmBiasKernel <<< n, 256 >>> (rocmOutput, (float*)weight.extraRocmData[0], k);
        //rocmDeviceSynchronize();

        FastllmRocmFree(rocmFp16Input);
        FastllmRocmFree(rocmFp16Output);
    } else {
        FastllmGemvFp32Fp16Kernel2<256, 1> <<< k, 256 >>>(rocmInput, (half *) weight.rocmData, rocmOutput, rocmBiasData, m, k);
    }

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

struct RocmMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    RocmMemoryBuffer () {}

    RocmMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};
std::map<int, std::vector <RocmMemoryBuffer>> rocmBuffersMap;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector <RocmMemoryBuffer>> bigBuffersMap;

void * FastllmRocmDirectMalloc(size_t size) {
    void * ret;
    checkHipErrors(hipMalloc(&ret, size));
    return ret;
}

void FastllmRocmDirectFree(void *ret) {
    DEBUG_PRINT("hipFree: %x", ret);
    checkHipErrors(hipFree(ret));
}

void * FastllmRocmMalloc(size_t size) {
    int id = -1;
    hipGetDevice(&id);
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 1 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void * ret;
        checkHipErrors(hipMalloc(&ret, size));
        bigBuffers.push_back(RocmMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &rocmBuffers = rocmBuffersMap[id];
    for (int i = 0; i < rocmBuffers.size(); i++) {
        if (rocmBuffers[i].size >= size && !rocmBuffers[i].busy) {
            rocmBuffers[i].busy = true;
            noBusyCnt[id] -= rocmBuffers[i].size;
            return rocmBuffers[i].data;
        }
    }
    void * ret;
    checkHipErrors(hipMalloc(&ret, size));
    rocmBuffers.push_back(RocmMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmRocmFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    for (auto &it: rocmBuffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &rocmBuffers = it.second;
            std::vector <RocmMemoryBuffer> temp;
            for (int i = 0; i < rocmBuffers.size(); i++) {
                if (!rocmBuffers[i].busy) {
                    hipSetDevice(it.first);
    DEBUG_PRINT("hipFree: %x", rocmBuffers[i].data);
                    checkHipErrors(hipFree(rocmBuffers[i].data));
                } else {
                    temp.push_back(rocmBuffers[i]);
                }
            }
            rocmBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    for (auto &it: rocmBuffersMap) {
        auto &rocmBuffers = it.second;
        for (int i = 0; i < rocmBuffers.size(); i++) {
            if (rocmBuffers[i].data == ret) {
                noBusyCnt[it.first] += rocmBuffers[i].size;
                rocmBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    checkHipErrors(hipFree(ret));
}

void FastllmRocmMallocBigBuffer(size_t size) {
    void * ret;
    int id = -1;
    hipGetDevice(&id);
    auto &bigBuffers = bigBuffersMap[id];
    checkHipErrors(hipMalloc(&ret, size));
    bigBuffers.push_back(RocmMemoryBuffer(ret, size, false));
}

void FastllmRocmClearBigBuffer() {
    int id = -1;
    hipGetDevice(&id);
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector <RocmMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                hipSetDevice(it.first);
                checkHipErrors(hipFree(bigBuffers[i].data));
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    hipSetDevice(id);
}

void FastllmRocmCopyFromHostToDevice(void *dst, void *src, size_t size) {
    checkHipErrors(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
    //rocmDeviceSynchronize();
}

void FastllmRocmCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    checkHipErrors(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
    //rocmDeviceSynchronize();
}

void FastllmRocmCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    checkHipErrors(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
    //rocmDeviceSynchronize();
}

void FastllmRocmMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    checkHipErrors(hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice));
    //rocmDeviceSynchronize();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpyBatchKernel (uint8_t** pointer) {
    int id = blockIdx.x;
    uint8_t *dst = pointer[id * 3];
    uint8_t *src = pointer[id * 3 + 1];
    size_t len = (size_t)(pointer[id * 3 + 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

void FastllmRocmMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                            size_t *	spitchs, size_t *widths, size_t *	heights,
                                            int batch) {
    int total = 0;
    for (int i = 0; i < batch; i++) {
        total += heights[i];
    }
    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;
    for (int i = 0; i < batch; i++) {
        for (int h = 0; h < heights[i]; h++) {
            cpuPointers[cur * 3 + 0] = (uint8_t*)dsts[i] + h * dpitchs[i];
            cpuPointers[cur * 3 + 1] = (uint8_t*)srcs[i] + h * spitchs[i];
            cpuPointers[cur * 3 + 2] = (uint8_t*)(widths[i]);

            cur++;
        }
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, hipMemcpyHostToDevice));
    FastllmMemcpyBatchKernel <128> <<<total, 128>>> (pointers);

    FastllmRocmFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
}

bool FastllmRocmGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmOutput, len);
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmSilu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmOutput, len);
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmSwiglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;

    int threadPerBlock = min(256, len);
    FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmOutput, len, spatial, mid);

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmMul(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmInput, rocmOutput, v, len);
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *rocmData = (float *) FastllmRocmPrepareInput(input0);
    float *input1Data = (float *) FastllmRocmPrepareInput(input1);

    int threadPerBlock = min(256, len);
    FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmData, input1Data, alpha, len);
    FastllmRocmFinishInput(input1, input1Data);
    FastllmRocmFinishOutput(input0, rocmData);
    return true;
}

bool FastllmRocmMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *rocmData = (float *) FastllmRocmPrepareInput(input0);
    float *input1Data = (float *) FastllmRocmPrepareInput(input1);

    int threadPerBlock = min(256, len);
    FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(rocmData, input1Data, alpha, len);
    FastllmRocmFinishInput(input1, input1Data);
    FastllmRocmFinishOutput(input0, rocmData);
    return true;
}

bool FastllmRocmAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *rocmData = (float *) FastllmRocmPrepareInput(input);
    float *maskData = (float *) FastllmRocmPrepareInput(mask);

    FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(rocmData, maskData, maskValue,
                                                       n, m, spatial);
    FastllmRocmFinishInput(mask, maskData);
    FastllmRocmFinishOutput(input, rocmData);
    return true;
}

bool FastllmRocmAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int n = input.dims[0], m = input.dims[1];
    int spn = input.dims[2], spm = input.dims[3];
    int spatial = input.Count(2);
    float *rocmData = (float *) FastllmRocmPrepareInput(input);
    float *maskData = (float *) FastllmRocmPrepareInput(mask);

    FastllmAlibiMaskKernel <256> <<< n * m, 256>>>(rocmData, maskData, maskValue,
                                                   n, m, spn, spm, spatial);
    FastllmRocmFinishInput(mask, maskData);
    FastllmRocmFinishOutput(input, rocmData);
    return true;
}

bool FastllmRocmSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);

    if (inner == 1) {
        if (channels < 8) {
            FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (rocmInput, rocmOutput, outer, channels);
        } else if (channels < 64) {
            FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> (rocmInput, rocmOutput, outer, channels);
        } else if (channels < 512) {
            FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (rocmInput, rocmOutput, outer, channels);
        } else {
            FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (rocmInput, rocmOutput, outer, channels);
        }

    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch) {
    int total = 0;
    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        total += outer;
    }
    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;

    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        auto &output = *outputs[b];
        float *rocmInput = (float *) input.rocmData;
        float *rocmOutput = (float *) output.rocmData;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(rocmInput + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(rocmOutput + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        } else {
            printf("softmax error.\n");
            exit(0);
        }
    }

    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, hipMemcpyHostToDevice));
    FastllmSoftmaxKernelBatchInner1 <256> <<<total, 256>>> (pointers);

    FastllmRocmFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmRocmRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (channels < 64) {
        FastllmRMSNormKernelInner1<1> <<< outer, 1 >>>(rocmInput, (float *) weight.rocmData, rocmOutput, outer, channels, eps);
    } else if (channels < 512) {
        FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>(rocmInput, (float *) weight.rocmData, rocmOutput, outer, channels, eps);
    } else {
        FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>(rocmInput, (float *) weight.rocmData, rocmOutput, outer, channels, eps);
    }

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis) {
    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];

    if (inner == 1) {
        if (channels < 64) {
            FastllmLayerNormKernelInner1<1> <<< outer, 1 >>>(rocmInput, (float *) gamma.rocmData,
                                                             (float *) beta.rocmData, rocmOutput,
                                                             outer, channels);
        } else if (channels < 512) {
            FastllmLayerNormKernelInner1<64> <<< outer, 64 >>>(rocmInput, (float *) gamma.rocmData,
                                                               (float *) beta.rocmData, rocmOutput,
                                                               outer, channels);
        } else {
            FastllmLayerNormKernelInner1<512> <<< outer, 512 >>>(rocmInput, (float *) gamma.rocmData,
                                                                 (float *) beta.rocmData, rocmOutput,
                                                                 outer, channels);
        }
    } else {
        printf("layernorm error.\n");
        exit(0);
    }

    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmTopK(const fastllm::Data &input, fastllm::Data &output, int topk) {
    if (topk != 1) {
        printf("topk: unsupport topk > 1.");
        exit(0);
    }

    float *rocmInput = (float *) FastllmRocmPrepareInput(input);
    float *rocmOutput = (float *) FastllmRocmPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (rocmInput, rocmOutput, channels);
    FastllmRocmFinishInput(input, rocmInput);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmPermute(fastllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != fastllm::DataDevice::ROCM) {
        printf("permute: data should in rocm.\n");
        exit(0);
    }
    int len = input.Count(0);
    float *tempData = (float *)FastllmRocmMalloc(len * sizeof(float));
    checkHipErrors(hipMemcpy(tempData, input.rocmData, len * sizeof(float), hipMemcpyDeviceToDevice));

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }
    if (axis == std::vector <int> {1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.rocmData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.rocmData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else {
        std::vector<int> temp;
        int len = input.Count(0);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(axis[i]);
        }
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }
        input.Resize(new_dims);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }

        int *rocmTemp = (int *) FastllmRocmMalloc(temp.size() * sizeof(int));
        checkHipErrors(hipMemcpy(rocmTemp, temp.data(), temp.size() * sizeof(int), hipMemcpyHostToDevice));
        int threadPerBlock = min(256, len);
        FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>((float *) input.rocmData,
                                                                                    tempData, rocmTemp,
                                                                                    (int) axis.size(), len);
        FastllmRocmFree(rocmTemp);
    }

    FastllmRocmFree(tempData);
    return true;
}

bool FastllmRocmAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.rocmData;
    float *kd = (float*)k.rocmData;
    float *vd = (float*)v.rocmData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.rocmData : nullptr;
    float *od = (float*)output.rocmData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));
    if (false) {
        float *qk = (float *) FastllmRocmMalloc(q0 * k1 * sizeof(float));
        float *temp = (float *) FastllmRocmMalloc(q0 * k1 * sizeof(float));
        FastllmAttentionKernel<256> <<<q0, 256>>>(qd, kd, vd, maskd, od,
                                                  scale, q1, q2, k1, v2,
                                                  group, q.strides[0], k.strides[0], v.strides[0], output.strides[0],
                                                  qk, temp);
        FastllmRocmFree(qk);
        FastllmRocmFree(temp);
        return true;
    }

    if (q1 > 1024) {
        float *qk = (float *) FastllmRocmMalloc(q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        hipblasStatus_t status;


        for (int i = 0; i < q0; i++) {
            // status = hipblasSgemmStridedBatched(fastllmCublasHandle,
            //                                    HIPBLAS_OP_T, HIPBLAS_OP_N,
            //                                    k1, q1, q2, &scale,
            //                                    kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
            //                                    qd + i * q.Count(1), q.strides[1], q.Count(1),
            //                                    &beta,
            //                                    qk, k1, k1 * q1, 1);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                gpuErrchk(status);
                exit(0);
            }

            if (maskd) {
                SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, -10000, q1 * k1);
            }

            int outer = q1;
            if (k1 < 8) {
                FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
            } else if (k1 < 64) {
                FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
            } else if (k1 < 512) {
                FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
            } else {
                FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
            }

            status = hipblasSgemmStridedBatched(fastllmCublasHandle,
                                               HIPBLAS_OP_N, HIPBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                gpuErrchk(status);
                exit(0);
            }
        }

        FastllmRocmFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        float *qk = (float *) FastllmRocmMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *) FastllmRocmMalloc(q0 * q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmHipblasHandle();
        hipblasStatus_t status;

        status = hipblasSgemmStridedBatched(fastllmCublasHandle,
                                           HIPBLAS_OP_T, HIPBLAS_OP_N,
                                           k1, q1 * group, q2, &scale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status);
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = hipblasSgemmStridedBatched(fastllmCublasHandle,
                                           HIPBLAS_OP_N, HIPBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            gpuErrchk(status);
            exit(0);
        }
        FastllmRocmFree(qk);
        FastllmRocmFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

bool FastllmRocmBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *rocmInput0 = (float *) FastllmRocmPrepareInput(input0);
    float *rocmInput1 = (float *) FastllmRocmPrepareInput(input1);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmHipblasHandle();
    hipblasStatus_t status;

    status = hipblasSgemmStridedBatched(fastllmCublasHandle,
                                       HIPBLAS_OP_N, HIPBLAS_OP_N,
                                       k, n, m, &alpha,
                                       rocmInput1, input1Stride, input1Spatial,
                                       rocmInput0, input0Stride, input0Spatial,
                                       &beta,
                                       rocmOutput, k, k * n, batch);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        gpuErrchk(status);
        exit(0);
    }

    FastllmRocmFinishInput(input0, rocmInput0);
    FastllmRocmFinishInput(input1, rocmInput1);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    float *rocmInput0 = (float *) FastllmRocmPrepareInput(input0);
    float *rocmInput1 = (float *) FastllmRocmPrepareInput(input1);
    float *rocmOutput = (float *) FastllmRocmPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmHipblasHandle();
    hipblasStatus_t status;

    status = hipblasSgemmStridedBatched(fastllmCublasHandle,
                                       HIPBLAS_OP_T, HIPBLAS_OP_N,
                                       k, n, m, &alpha,
                                       rocmInput1, input1Stride, input1Spatial,
                                       rocmInput0, input0Stride, input0Spatial,
                                       &beta,
                                       rocmOutput, k, k * n, batch);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        gpuErrchk(status);
        exit(0);
    }

    FastllmRocmFinishInput(input0, rocmInput0);
    FastllmRocmFinishInput(input1, rocmInput1);
    FastllmRocmFinishOutput(output, rocmOutput);
    return true;
}

bool FastllmRocmRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *rocmData = (float *) FastllmRocmPrepareInput(data);
    float *rocmPositionIds = (float *) FastllmRocmPrepareInput(positionIds);
    float *rocmSin = (float *) FastllmRocmPrepareInput(sinData);
    float *rocmCos = (float *) FastllmRocmPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmRotatePosition2DKernel <<< outer * 2 * n, min(rotaryDim, m / 4) >>> (rocmData, rocmPositionIds, rocmSin, rocmCos,
                                                                                len, bs, spatial, n, m,
                                                                                (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmRocmFinishInput(positionIds, rocmPositionIds);
    FastllmRocmFinishInput(sinData, rocmSin);
    FastllmRocmFinishInput(cosData, rocmCos);
    FastllmRocmFinishOutput(data, rocmData);

    return true;
}

bool FastllmRocmNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                       const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *rocmData = (float *) FastllmRocmPrepareInput(data);
    float *rocmPositionIds = (float *) FastllmRocmPrepareInput(positionIds);
    float *rocmSin = (float *) FastllmRocmPrepareInput(sinData);
    float *rocmCos = (float *) FastllmRocmPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmNearlyRotatePosition2DKernel <<< outer * n, min(rotaryDim, m / 4) >>> (rocmData, rocmPositionIds, rocmSin, rocmCos,
                                                                                  len, bs, spatial, n, m,
                                                                                  (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmRocmFinishInput(positionIds, rocmPositionIds);
    FastllmRocmFinishInput(sinData, rocmSin);
    FastllmRocmFinishInput(cosData, rocmCos);
    FastllmRocmFinishOutput(data, rocmData);
    return true;
}

bool FastllmRocmLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *rocmData = (float *) FastllmRocmPrepareInput(data);
    float *rocmPositionIds = (float *) FastllmRocmPrepareInput(positionIds);
    float *rocmSin = (float *) FastllmRocmPrepareInput(sinData);
    float *rocmCos = (float *) FastllmRocmPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmLlamaRotatePosition2DKernel <<< outer * n, min(rotaryDim, m / 2) >>> (rocmData, rocmPositionIds, rocmSin, rocmCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmRocmFinishInput(positionIds, rocmPositionIds);
    FastllmRocmFinishInput(sinData, rocmSin);
    FastllmRocmFinishInput(cosData, rocmCos);
    FastllmRocmFinishOutput(data, rocmData);
    return true;
}

bool FastllmRocmApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds) {
    float *inputData = (float *) input.rocmData;
    float *lognData = (float *) lognAttn.rocmData;
    float *posData = (float *) positionIds.rocmData;
    int batch = input.dims[0];
    int seqLen = input.dims[1];
    int spatial = input.Count(2);

    FastllmApplyLognAttnKernel <256> <<<batch * seqLen, 256>>> (inputData, lognData, posData, batch, seqLen, spatial);
    return true;
}

bool FastllmRocmAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    int k0 = k[0]->dims[0];
    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }
    float *mem = (float*) FastllmRocmMalloc(memSum * sizeof(float));
    float **qk = new float*[batch];
    memSum = 0;
    for (int b = 0; b < batch; b++) {
        int s = q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
        qk[b] = mem + memSum;
        memSum += s;
    }

    if (true) {
        uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * batch * k0 * 8);
        uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) q[b]->rocmData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) k[b]->rocmData + i * k[b]->strides[0] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) q[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) q[b]->strides[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) k[b]->strides[1];
            }
        }
        checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, hipMemcpyHostToDevice));
        FastllmMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        FastllmRocmFree(pointers);
        delete[] cpuPointers;
    }
    if (true) {
        int total = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            total += outer;
        }
        uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * total * 3);
        uint8_t ** cpuPointers = new uint8_t*[total * 3];
        int cur = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            int channels = k[b]->dims[1];
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(qk[b] + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(qk[b] + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        }
        checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, hipMemcpyHostToDevice));
        FastllmSoftmaxKernelBatchInner1 <256> <<<total, 256>>> (pointers);

        FastllmRocmFree(pointers);
        delete[] cpuPointers;
    }
    if (true) {
        uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * batch * k0 * 8);
        uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) v[b]->rocmData + i * v[b]->strides[0] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) output[b]->rocmData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) v[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) v[b]->strides[1];
            }
        }
        checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, hipMemcpyHostToDevice));
        FastllmMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        FastllmRocmFree(pointers);
        delete[] cpuPointers;
    }

    FastllmRocmFree(mem);
    delete[] qk;
/*
    {
        const int params = 16;
        float **pointers = (float **) FastllmRocmMalloc(sizeof(float *) * batch * params);
        float **cpuPointers = new float *[batch * params];

        float **qk = new float *[batch];
        float **temp = new float *[batch];
        for (int b = 0; b < batch; b++) {
            qk[b] = (float *) FastllmRocmMalloc(q[b]->dims[0] * k[b]->dims[1] * sizeof(float));
            temp[b] = (float *) FastllmRocmMalloc(q[b]->dims[0] * k[b]->dims[1] * sizeof(float));

            cpuPointers[b * params + 0] = (float *) q[b]->rocmData;
            cpuPointers[b * params + 1] = (float *) k[b]->rocmData;
            cpuPointers[b * params + 2] = (float *) v[b]->rocmData;
            cpuPointers[b * params + 3] = (mask[b] && mask[b]->dims.size() > 0) ? (float *) mask[b]->rocmData : nullptr;
            cpuPointers[b * params + 4] = (float *) output[b]->rocmData;
            cpuPointers[b * params + 5] = (float *) (unsigned long long) q[b]->dims[1];
            cpuPointers[b * params + 6] = (float *) (unsigned long long) q[b]->dims[2];
            cpuPointers[b * params + 7] = (float *) (unsigned long long) k[b]->dims[1];
            cpuPointers[b * params + 8] = (float *) (unsigned long long) v[b]->dims[2];
            cpuPointers[b * params + 9] = (float *) (unsigned long long) q[b]->strides[0];
            cpuPointers[b * params + 10] = (float *) (unsigned long long) k[b]->strides[0];
            cpuPointers[b * params + 11] = (float *) (unsigned long long) v[b]->strides[0];
            cpuPointers[b * params + 12] = (float *) (unsigned long long) output[b]->strides[0];
            cpuPointers[b * params + 13] = (float *) (unsigned long long) qk[b];
            cpuPointers[b * params + 14] = (float *) (unsigned long long) temp[b];
            cpuPointers[b * params + 15] = (float *) (unsigned long long) q[b]->dims[0];
        }

        rocmMemcpy(pointers, cpuPointers, sizeof(float *) * batch * params, rocmMemcpyHostToDevice);
        FastllmAttentionBatchKernel<256> <<< batch, 256 >>>(pointers, scale, group);

        for (int i = 0; i < batch; i++) {
            FastllmRocmFree(qk[i]);
            FastllmRocmFree(temp[i]);
        }
        delete[] qk;
        delete[] temp;

        FastllmRocmFree(pointers);
        delete[] cpuPointers;
    }
*/
/*
    for (int b = 0; b < batch; b++) {
        int q0 = q[b]->dims[0], q1 = q[b]->dims[1], q2 = q[b]->dims[2], k0 = k[b]->dims[0], k1 = k[b]->dims[1], v2 = v[b]->dims[2];
        float *qd = (float *) q[b]->rocmData;
        float *kd = (float *) k[b]->rocmData;
        float *vd = (float *) v[b]->rocmData;
        float *maskd = (mask[b] && mask[b]->dims.size() > 0) ? (float *) mask[b]->rocmData : nullptr;
        float *od = (float *) output[b]->rocmData;
        int maskBatch = (mask[b] && mask[b]->dims.size() > 0) ? mask[b]->dims[0] : 1;

        float *qk = (float *) FastllmRocmMalloc(q0 * k1 * sizeof(float));
        float *temp = (float *) FastllmRocmMalloc(q0 * k1 * sizeof(float));
        FastllmAttentionKernel<256> <<<q0, 256>>>(qd, kd, vd, maskd, od,
                                                  scale, q1, q2, k1, v2,
                                                  group, q[b]->strides[0], k[b]->strides[0], v[b]->strides[0],
                                                  output[b]->strides[0],
                                                  qk, temp);
    }
*/
    DeviceSync();
    return true;
}

bool FastllmRocmSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis) {
    int part = input.dims[axis];
    int outer = input.Count(0) / input.Count(axis);
    int inputStride = input.Count(axis);
    int outputStride = outputs[0]->Count(axis);
    int inner = input.strides[axis];
    int unitSize = input.unitSize;

    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)outputs[i]->rocmData;
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, hipMemcpyHostToDevice));
    FastllmSplitBatchKernel <256> <<< part * outer, 256 >>> ((uint8_t*)input.rocmData, pointers, outer, part, inner * unitSize);

    FastllmRocmFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmRocmCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis) {
    int part = output.dims[axis];
    int outer = output.Count(0) / output.Count(axis);
    int inputStride = inputs[0]->Count(axis);
    int outputStride = output.Count(axis);
    int inner = output.strides[axis];
    int unitSize = output.unitSize;

    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)inputs[i]->rocmData;
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, hipMemcpyHostToDevice));
    FastllmCatBatchKernel <256> <<< part * outer, 256 >>> (pointers, (uint8_t*)output.rocmData, outer, part, inner * unitSize);

    FastllmRocmFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmRocmMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs) {
    float ** pointers = (float**)FastllmRocmMalloc(sizeof(float*) * batch * 3);
    float ** cpuPointers = new float*[batch * 3];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = (float*)inputs[i]->rocmData;
        cpuPointers[i + batch] = (float*)outputs[i]->rocmData;
        cpuPointers[i + batch * 2] = (float*)(inputs[i]->Count(0));
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(float*) * batch * 3, hipMemcpyHostToDevice));
    FastllmMulBatchKernel <256> <<< batch, 256 >>> (pointers, batch, v);

    FastllmRocmFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmRocmBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, hipMemcpyHostToDevice));
    FastllmMatMulTransBBatchKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmRocmFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmRocmBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                 int *ns, int *ms, int *ks,
                                 int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmRocmMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    checkHipErrors(hipMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, hipMemcpyHostToDevice));
    FastllmMatMulKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmRocmFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

void FastllmRocmSetDevice(int gpu_id) {
    hipSetDevice(gpu_id);
}
