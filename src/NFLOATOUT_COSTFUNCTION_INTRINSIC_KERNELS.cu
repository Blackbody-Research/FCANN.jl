// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix


#include <curand.h>
#include <curand_kernel.h>

extern "C"   // ensure function name to be exactly "eeTanh"
{

    __global__ void tanhGradient(int N, int M, float *z, float *tanh_grad_z) {
      
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;

		float c1 = __fdividef(2.0, 3.0);
		
		if (i < N && j < M) {		
			float el = __fmul_rn(z[index], c1);
			if (el > 4.97) {
				z[index] = 1.7159;
				tanh_grad_z[index] = 0.0; 
			}
			else if(el < -4.97) {
				z[index] = -1.7159;
				tanh_grad_z[index] = 0.0; 
			}
			else {
				float x2 = __fmul_rn(el, el);
				float a = __fmul_rn(el, __fmaf_rn(x2, __fmaf_rn(x2, __fadd_rn(378.0, x2), 17235.0), 135135.0));
				float b = __fmaf_rn(x2, __fmaf_rn(x2, __fmaf_rn(x2, 28.0, 3150.0), 62370.0), 135135.0);
				float tanh = __fdividef(a, b);
				z[index] = __fmul_rn(1.7159, tanh);
				tanh_grad_z[index] = __fmul_rn(1.7159, __fmul_rn(__fmaf_rn(-tanh, tanh, 1.0), c1));
			}
		}
	}

	__global__ void tanhGradientDropout(int N, int M, float *z, float *tanh_grad_z, int seed, float D) { 
	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;

		float c1 = __fdividef(2.0, 3.0);
		
		if (i < N && j < M) {
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
					z[index] = 1.7159; 
					tanh_grad_z[index] = 0.0;
				}
				else if(el < -4.97) {
					z[index] = -1.7159;
					tanh_grad_z[index] = 0.0;
				}
				else {
					float x2 = __fmul_rn(el, el);
					float a = __fmul_rn(el, __fmaf_rn(x2, __fmaf_rn(x2, __fadd_rn(378.0, x2), 17235.0), 135135.0));
					float b = __fmaf_rn(x2, __fmaf_rn(x2, __fmaf_rn(x2, 28.0, 3150.0), 62370.0), 135135.0);
					float tanh = __fdividef(a, b);
					z[index] = __fmul_rn(1.7159, tanh);
					tanh_grad_z[index] = __fmul_rn(1.7159, __fmul_rn(__fmaf_rn(-tanh, tanh, 1.0), c1));
				}
			}
		}
	}
	
	
	__global__ void tanhActivation(int N, int M, float *z)
    {
       
	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		float c1 = __fdividef(2.0, 3.0);
		
		if (i < N && j < M)
		{
	 
		
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
	
	__global__ void finish_delta(int N, int M, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
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
	
		__global__ void finishAdvX(int N, int M, float *X, float *advX)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
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
	
	__global__ void elMul(int N, int M, float *X1, float *X2)
    {
       
	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			X1[index] = __fmul_rn(X1[index], X2[index]);
		}
	}
}