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
		float scaleFactor1 = __fdividef(1.7159, __fsub_rn(1.0, D));
		float scaleFactor2 = __fdividef(-1.7159, __fsub_rn(1.0, D));
		
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

	__global__ void sqErr(int N, int M, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			float tmp = __fsub_rn(A[index], Y[index]);
			A[index] = __fmul_rn(tmp, tmp);
			// A[index] = (A[index]-Y[index])^2
		}
	}

	__global__ void absErr(int N, int M, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			A[index] = fabsf(__fsub_rn(A[index], Y[index]));
			// A[index] = abs(A[index]-Y[index])
		}
	}

	__global__ void sqErrDeriv(int N, int M, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			out[index] = __fmul_rn(2.0, __fsub_rn(A[index], Y[index]));
			// Out[index] = 2*(A[index] - Y[index])
		}
	}

	__global__ void absErrDeriv(int N, int M, float *A, float *Y, float *out)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			out[index] = copysignf(1.0, __fsub_rn(A[index], Y[index]));
		}
	}

	__global__ void normLogErr(int N, int M, float *A, float *Y)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		int L = N*M;
		
		if (i < N && j < M)
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
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		int L = N*M;
		
		if (i < N && j < M)
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
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		int L = N*M;
		
		if (i < N && j < M)
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
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		int L = N*M;
		
		if (i < N && j < M)
		{
			float a = __expf(A[index+L]);
			if (A[index] > Y[index])
			{
				out[index] = a;
			}
			else if (A[index] < Y[index])
			{
				out[index] = -a;
			}
			else
			{
				out[index] = 0.0;
			}

			out[index+L] = __fmaf_rn(a, fabsf(__fsub_rn(A[index],  Y[index])), -1.0);
			// A2 in this case is stored in the doubled rows of A, the length of A is 
			// doublt that of Y, out is the same length as A and will store both parts of the derivative 
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