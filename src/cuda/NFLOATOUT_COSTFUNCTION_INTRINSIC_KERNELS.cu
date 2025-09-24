// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix


#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

extern "C"   // ensure function name to be exactly "eeTanh"
{

    __global__ void tanhGradient(int N, float *z, float *tanh_grad_z) {
      
		int index = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
			
		for (int i = index; i < N; i += stride)
		{
			float c1 = __fdividef(2.0, 3.0);
			float el = __fmul_rn(z[i], c1);
			if (el > 4.97) {
				z[i] = 1.7159;
				tanh_grad_z[i] = 0.0; 
			}
			else if(el < -4.97) {
				z[i] = -1.7159;
				tanh_grad_z[i] = 0.0; 
			}
			else {
				float x2 = __fmul_rn(el, el);
				float a = __fmul_rn(el, __fmaf_rn(x2, __fmaf_rn(x2, __fadd_rn(378.0, x2), 17235.0), 135135.0));
				float b = __fmaf_rn(x2, __fmaf_rn(x2, __fmaf_rn(x2, 28.0, 3150.0), 62370.0), 135135.0);
				float tanh = __fdividef(a, b);
				z[i] = __fmul_rn(1.7159, tanh);
				tanh_grad_z[i] = __fmul_rn(1.7159, __fmul_rn(__fmaf_rn(-tanh, tanh, 1.0), c1));
			}
		}
	}

	__global__ void tanhGradientDropout(int N, float *z, float *tanh_grad_z, int seed, float D) { 
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			float c1 = __fdividef(2.0, 3.0);
			float scaleFactor1 = __fdividef(1.7159, __fsub_rn(1.0, D));
			float scaleFactor2 = __fdividef(-1.7159, __fsub_rn(1.0, D));
			curandState_t state;	
			curand_init( (seed << 20) + index, 0, 0, &state);

			float F = curand_uniform(&state);
			// float F = 0.5;

			if(F<D) {
				z[index] = 0.0;
				tanh_grad_z[index] = 0.0;
			}
			else {
				float el = __fmul_rn(z[index], c1);
				if(el > 4.97) {
					z[index] = scaleFactor1; 
					tanh_grad_z[index] = 0.0;
				}
				else if(el < -4.97) {
					z[index] = scaleFactor2;
					tanh_grad_z[index] = 0.0;
				}
				else {
					float x2 = __fmul_rn(el, el);
					float a = __fmul_rn(el, __fmaf_rn(x2, __fmaf_rn(x2, __fadd_rn(378.0, x2), 17235.0), 135135.0));
					float b = __fmaf_rn(x2, __fmaf_rn(x2, __fmaf_rn(x2, 28.0, 3150.0), 62370.0), 135135.0);
					float tanh = __fdividef(a, b);
					z[index] = __fmul_rn(scaleFactor1, tanh);
					tanh_grad_z[index] = __fmul_rn(scaleFactor1, __fmul_rn(__fmaf_rn(-tanh, tanh, 1.0), c1));
				}
			}
		}
	}

	__global__ void noactivationGradient(int N, float *z, float *tanh_grad_z, int seed, float D) { 
	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		 {
			float scaleFactor = __fdividef(1.0, __fsub_rn(1.0, D));
			curandState_t state;	
			curand_init( (seed << 20) + index, 0, 0, &state);

			float F = curand_uniform(&state);
			// float F = 0.5;

			if (D != 0.0) {
				if (F < D) {
					z[index] = 0.0;
					tanh_grad_z[index] = 0.0;
				}
				else {
					tanh_grad_z[index] = scaleFactor;
					z[index] = __fmul_rn(scaleFactor, z[index]);
				}
			}
			else {
				tanh_grad_z[index] = 1.0;
			}
		}
	}
	
	
	__global__ void tanhActivation(int N, float *z)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
	 
		float c1 = __fdividef(2.0, 3.0);
		float el = __fmul_rn(z[index], c1);
		if (el > 4.97)
		{
			z[index] = 1.7159; 
		}
		else if (el < -4.97)
		{
			z[index] = -1.7159;
		}
		else
		{
			float x2 = __fmul_rn(el, el);
			float a = __fmul_rn(el, __fmaf_rn(x2, __fmaf_rn(x2, __fadd_rn(378.0, x2), 17235.0), 135135.0));
			float b = __fmaf_rn(x2, __fmaf_rn(x2, __fmaf_rn(x2, 28.0, 3150.0), 62370.0), 135135.0);
			float tanh = __fdividef(a, b);
			z[index] = __fmul_rn(1.7159, tanh);
		}
		}
	}
	
	__global__ void fill_cols(int N, int M, float *X, float *V)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			X[index] = V[j];
		
		}
	}	
	
	__global__ void swap_matrix_col(int N, int C, float *X, float *V)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int index = (C-1)*N + i;
		
		if (i < N)
		{
			float a = X[index];
			X[index] = V[i];
			V[i] = a;
		}
	}	
	

	__global__ void finish_delta(int N, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			out[index] = copysignf(1.0, __fsub_rn(A[index], Y[index]));
			
			/*
			if (A[index] < Y[index])
			{
				out[index] = -1.0;
			}
			else if (A[index] > Y[index])
			{
				out[index] = 1.0;			
			}
			else 
			{
				out[index] = 0.0;
			}
			*/
		
		}
	}

	__global__ void sqErr(int N, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			float tmp = __fsub_rn(A[index], Y[index]);
			A[index] = __fmul_rn(tmp, tmp);
			// A[index] = (A[index]-Y[index])^2
		}
	}

	__global__ void absErr(int N, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			A[index] = fabsf(__fsub_rn(A[index], Y[index]));
			// A[index] = abs(A[index]-Y[index])
		}
	}

	__global__ void sqErrDeriv(int N, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			out[index] = __fmul_rn(2.0, __fsub_rn(A[index], Y[index]));
			// Out[index] = 2*(A[index] - Y[index])
		}
	}

	__global__ void outputIndex(int N, float *A, int idx)
    {	
	}

	__global__ void outputIndexDeriv(int N, float *deltas, const float *a, int idx)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		

		for (int index = i; index < N; index += stride)
		{
			if (index == idx)
			{
				deltas[index] = 1.0f;
			}
			else
			{
				deltas[index] = 0.0f;
			}
			// deltas[index] = (float)(index == idx);
		}
	}

	// Single block version
	__global__ void crossEntropy(int n, float* a, int target_index) {
	    extern __shared__ float sdata[];
    
		int tid = threadIdx.x;
		int stride = blockDim.x;
		
		// Phase 1: Find maximum value using shared memory reduction
		float thread_max = -INFINITY;
		for (int i = tid; i < n; i += stride) {
			thread_max = fmaxf(thread_max, a[i]);
		}
		
		sdata[tid] = thread_max;
		__syncthreads();
		
		// Block-level max reduction
		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
			}
			__syncthreads();
		}
		
		float global_max = sdata[0];
		__syncthreads();
		
		// Phase 2: Compute exp sum and modify array elements
		float thread_sum = 0.0f;
		for (int i = tid; i < n; i += stride) {
			float h = a[i] - global_max;       // h = a[i] - m
			float exp_h = expf(h);             // x = exp(h)
			thread_sum += exp_h;               // s += x
			
			// a[i] = h * (i == index) - store h only if target, else 0
			a[i] = (i == target_index) ? h : 0.0f;
		}
		
		sdata[tid] = thread_sum;
		__syncthreads();
		
		// Block-level sum reduction
		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}
		
		float global_sum = sdata[0];
		
		// Phase 3: Final adjustment for target index
		// a[index] = -a[index] + log(s)
		if (tid == 0) {
			float log_sum = logf(global_sum);
			a[target_index] = -a[target_index] + log_sum;
		}
	}

	//single block version
	__global__ void crossEntropyDeriv(int N, float *deltas, const float *a, int idx)
	{
		extern __shared__ float sdata[];
    
		int tid = threadIdx.x;
		int stride = blockDim.x;
		
		// Phase 1: Find maximum value
		float thread_max = -INFINITY;
		for (int i = tid; i < N; i += stride) {
			thread_max = fmaxf(thread_max, a[i]);
		}
		
		// Block reduction for max
		sdata[tid] = thread_max;
		__syncthreads();
		
		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
			}
			__syncthreads();
		}
		
		float global_max = sdata[0];
		__syncthreads();
		
		// Phase 2: Compute exponentials and sum
		float thread_sum = 0.0f;
		for (int i = tid; i < N; i += stride) {
			float exp_val = expf(a[i] - global_max);
			deltas[i] = exp_val;
			thread_sum += exp_val;
		}
		
		// Block reduction for sum
		sdata[tid] = thread_sum;
		__syncthreads();
		
		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}
		
		float global_sum = sdata[0];
		__syncthreads();
		
		// Phase 3: Normalize and adjust target
		float inv_sum = 1.0f / global_sum;
		for (int i = tid; i < N; i += stride) {
			deltas[i] *= inv_sum;
			if (i == idx) {
				deltas[i] -= 1.0f;
			}
		}
	}

	__global__ void absErrDeriv(int N, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			out[index] = copysignf(1.0, __fsub_rn(A[index], Y[index]));
		}
	}

	__global__ void normLogErr(int N, int M, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		int L = N*M;

		for (int index = i; index < L; index += stride)
		{
			// A2 in this case is stored in the doubled rows of A, the length of A is 
			// doublt that of Y 
			float a = __expf(__fmul_rn(2.0, A[index+L]));
			A[index] = __fmul_rn(a, __fmaf_rn(0.5, __fmul_rn(Y[index], Y[index]), __fsub_rn(__fmul_rn(0.5, __fmul_rn(A[index], A[index])),  __fmul_rn(A[index], Y[index]))));
			A[index+L] = __fsub_rn(0.9189385332, A[index+L]); // stick final sum factor in 2nd part of A so when it sums to total the cost will be correct
			// A[index] = a*(A[index]*(0.5*A[index] - Y[index]) + 0.5*Y[index]*Y[index]);
			// A[index+L] = __fsub_rn(0.9189385332, A[index+L]);
		}
	}

	__global__ void normLogErrDeriv(int N, int M, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		int L = N*M;

		for (int index = i; index < L; index += stride)
		{
			// A2 in this case is stored in the doubled rows of A, the length of A is 
			// doublt that of Y, out is the same length as A and will store both parts of the derivative 
			float a = __expf(__fmul_rn(2.0, A[index+L]));
			float b = __fsub_rn(A[index], Y[index]);
			out[index] = __fmul_rn(b, a);
			out[index+L] = __fsub_rn(__fmul_rn(out[index], b), 1.0);
		}
	}


	__global__ void cauchyLogErr(int N, int M, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		int L = N*M;

		for (int index = i; index < L; index += stride)
		{
			// A2 in this case is stored in the doubled rows of A, the length of A is 
			// doublt that of Y 
			float a = __expf(A[index+L]);
			A[index] = __fmul_rn(fabsf(__fsub_rn(A[index], Y[index])), a);
			A[index +L] = -__logf(__fmul_rn(0.5, a)); // stick final sum factor in 2nd part of A so when it sums to total the cost will be correct
		}
	}

	__global__ void cauchyLogErrDeriv(int N, int M, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		int L = N*M;

		for (int index = i; index < L; index += stride)
		{
			float a = __expf(A[index+L]);

			float diff = __fsub_rn(A[index], Y[index]);
			float sign = (diff > 0) - (diff < 0); // sign function
			
			out[index] = a*sign;

			out[index+L] = __fmaf_rn(a, fabsf(__fsub_rn(A[index],  Y[index])), -1.0);
			// A2 in this case is stored in the doubled rows of A, the length of A is 
			// doublt that of Y, out is the same length as A and will store both parts of the derivative 
		}
	}

	
		__global__ void finishAdvX(int N, float *X, float *advX)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			if (advX[index] < 0)
			{
				advX[index] = X[index] - 5.0e-5;
			}
			else if (advX[index] > 0)
			{
				advX[index] = X[index] + 5.0e-5;		
			}
			else 
			{
				advX[index] = X[index];
			}
		
		}
	}
	
	__global__ void elMul(int N, float *X1, float *X2)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			X1[index] = __fmul_rn(X1[index], X2[index]);
		}
	}
}