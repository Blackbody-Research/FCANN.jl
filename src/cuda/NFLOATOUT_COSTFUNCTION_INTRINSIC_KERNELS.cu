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

	__global__ void sqErrIndex(int N, int M, float *A, float *Y, int *indx)
    {	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			if (j == indx[i]) {
				float tmp = __fsub_rn(A[index], Y[i]);
				A[index] = __fmul_rn(tmp, tmp);
			}
			else {
				A[index] = 0.0;
			}
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

	__global__ void absErrIndex(int N, int M, float *A, float *Y, int *indx)
	{	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			if (j == indx[i]) {
				A[index] = fabsf(__fsub_rn(A[index], Y[i]));
			}
			else {
				A[index] = 0.0;
			}
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

	__global__ void sqErrIndexDeriv(int N, int M, float *A, float *Y, int *indx, float *out)
	{	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			if (j == indx[i]) {
				out[index] = __fmul_rn(2.0, __fsub_rn(A[index], Y[i]));
			}
			else {
				out[index] = 0.0;
			}
		}
	}

	__global__ void outputIndex(int N, float *A, int idx)
    {	
	}

	__global__ void outputIndexBatch(int N, float *A, int *idx)
	{
	}

	__global__ void outputIndexBatchDeriv(int N, int M, float *deltas, const float *a, int *inx)
	{
		// i = example index, each block handles one example (run_kernel_batch with N=m, M=output_layer_size)
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int target = inx[i];

		// deltas: (M, N) column-major (class j of example i at flat j*N + i)
		// set deltas[i, target] = 1, all else 0 (matches CPU calcDeltaOut!(deltas, indices))
		for (int j = tid; j < M; j += stride) {
			deltas[j * N + i] = (j == target) ? 1.0f : 0.0f;
		}
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

	// Single block version.  The loss at the target index is
	//     -h[target] + log(sum_j exp h_j) + beta*H,
	// where h = a - max(a) and H = -sum_j p_j*log(p_j) with p the softmax of h.  The beta term
	// mirrors the CPU calcFinalOut!(::CrossEntropyLoss, a::Vector, index) so the reported cost
	// matches the CPU for entropy regularized cross entropy.  Only the target entry is left
	// meaningful (the other entries are zeroed) which is what the caller reads back.
	__global__ void crossEntropy(int n, float* a, int target_index, float beta) {
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
		
		// Phase 2: sum of exponentials (read only - a[] is left untouched)
		float thread_sum = 0.0f;
		for (int i = tid; i < n; i += stride) {
			thread_sum += expf(a[i] - global_max);
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
		__syncthreads();
		
		// Phase 3: entropy regularization term H = -sum_j p_j*log(p_j) (shared memory reduction)
		float global_entropy = 0.0f;
		if (beta > 0.0f) {
			float thread_ent = 0.0f;
			for (int i = tid; i < n; i += stride) {
				float p = expf(a[i] - global_max) / global_sum;
				thread_ent -= p * logf(fmaxf(p, 1.1920929e-7f)); // eps(Float32), matches CPU
			}
			sdata[tid] = thread_ent;
			__syncthreads();
			for (int s = blockDim.x / 2; s > 0; s >>= 1) {
				if (tid < s) {
					sdata[tid] += sdata[tid + s];
				}
				__syncthreads();
			}
			global_entropy = sdata[0];
			__syncthreads();
		}
		
		// Phase 4: leave the loss at the target index, zero the remaining entries
		for (int i = tid; i < n; i += stride) {
			if (i == target_index) {
				a[i] = -(a[i] - global_max) + logf(global_sum) + beta * global_entropy;
			} else {
				a[i] = 0.0f;
			}
		}
	}

	// Single block version
	__global__ void crossEntropyDist(int n, float* a, float* target_dist) {
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

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
			}
			__syncthreads();
		}

		float global_max = sdata[0];
		__syncthreads();

		// Phase 2: Compute exp sum and weighted logit sum, zero out all but first entry
		float thread_sum = 0.0f;
		float thread_loss = 0.0f;
		for (int i = tid; i < n; i += stride) {
			float h = a[i] - global_max;
			thread_sum += expf(h);
			thread_loss += target_dist[i] * h;
			a[i] = 0.0f;
		}

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

		// Phase 3: Reduce weighted logit sum
		sdata[tid] = thread_loss;
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		float global_loss = sdata[0];
		__syncthreads();

		// Phase 4: store loss = log(sum(exp(h))) - sum(t * h) at index 0 so that summing all
		// entries (others are zero) yields the cross entropy loss for the single example
		if (tid == 0) {
			a[0] = logf(global_sum) - global_loss;
		}
	}

	__global__ void crossEntropyBatch(int N, int M, float* A, int* target_indices) {
		extern __shared__ float sdata[];

		// i = example index, each block handles one example
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int target = target_indices[i];

		// Phase 1: Find maximum value across M classes for this example
		// A[j * N + i] accesses class j of example i
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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

		// Phase 2: Compute exp sum, zero out non-target, store h at target
		float thread_sum = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float h = A[j * N + i] - global_max;
			thread_sum += expf(h);
			A[j * N + i] = (j == target) ? h : 0.0f;
		}

		sdata[tid] = thread_sum;
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		float global_sum = sdata[0];

		// Phase 3: Compute final loss at target location
		if (tid == 0) {
			A[target * N + i] = -A[target * N + i] + logf(global_sum);
		}
	}

	__global__ void crossEntropyDistBatch(int N, int M, float* A, float* target_dists) {
		extern __shared__ float sdata[];

		// i = example index, each block handles one example
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		// Phase 1: Find maximum value across M classes for this example
		// A[j * N + i] accesses class j of example i
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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

		// Phase 2: Compute exp sum and weighted logit sum, zero out all classes except the first
		float thread_sum = 0.0f;
		float thread_loss = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float h = A[j * N + i] - global_max;
			thread_sum += expf(h);
			thread_loss += target_dists[j * N + i] * h;
			A[j * N + i] = 0.0f;
		}

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

		// Phase 3: Reduce weighted logit sum
		sdata[tid] = thread_loss;
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		float global_loss = sdata[0];
		__syncthreads();

		// Phase 4: store loss for example i at class 0 (column 0) so that summing over column 0
		// divided by N yields the mean cross entropy loss over the batch
		if (tid == 0) {
			A[0 * N + i] = logf(global_sum) - global_loss;
		}
	}

	//single block version
	__global__ void crossEntropyDeriv(int N, float *deltas, const float *a, int idx, float beta)
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
		
		// Phase 3: Normalize to softmax
		float inv_sum = 1.0f / global_sum;
		for (int i = tid; i < N; i += stride) {
			deltas[i] *= inv_sum;
		}
		__syncthreads();
		
		// Phase 3b: Entropy regularization (matches CPU calcDeltaOut! for CrossEntropyLoss with beta)
		// entropy = -sum_j p_j * log(p_j)
		if (beta > 0.0f) {
			// compute -p*log(p) partial per thread
			float thread_entropy = 0.0f;
			for (int i = tid; i < N; i += stride) {
				float p = deltas[i];
				thread_entropy -= p * logf(fmaxf(p, 1.1920929e-7f)); // ~eps(Float32)
			}
			sdata[tid] = thread_entropy;
			__syncthreads();
			for (int s = blockDim.x / 2; s > 0; s >>= 1) {
				if (tid < s) {
					sdata[tid] += sdata[tid + s];
				}
				__syncthreads();
			}
			float entropy = sdata[0];
			__syncthreads();
			// deltas[i] -= beta * p * (entropy + log(p))
			for (int i = tid; i < N; i += stride) {
				float p = deltas[i];
				deltas[i] -= beta * p * (entropy + logf(fmaxf(p, 1.1920929e-7f)));
			}
			__syncthreads();
		}
		
		// Phase 4: subtract one at target index
		for (int i = tid; i < N; i += stride) {
			if (i == idx) {
				deltas[i] -= 1.0f;
			}
		}
	}

	__global__ void crossEntropyBatchDeriv(int N, int M, float* deltas, const float* A, const int* indices) {
		extern __shared__ float sdata[];

		// i = example index, each block handles one example
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int idx = indices[i];

		// Phase 1: Find maximum value across M classes for this example
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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

		// Phase 2: Compute exponentials and sum into deltas
		float thread_sum = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float exp_val = expf(A[j * N + i] - global_max);
			deltas[j * N + i] = exp_val;
			thread_sum += exp_val;
		}

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
		for (int j = tid; j < M; j += stride) {
			deltas[j * N + i] *= inv_sum;
			if (j == idx) {
				deltas[j * N + i] -= 1.0f;
			}
		}
	}
	//batch cross entropy derivative with entropy regularization (beta).  one block per example; the
	//entropy term uses the per-example (row) entropy H_i = -sum_j p_ij*log(p_ij) which makes the
	//gradient consistent with the batch forward loss that includes beta*H_i per example (and matches
	//the CPU single-example (m=1) semantics identically).  All of this is computed on the GPU with no
	//host round trip - only shared memory block reductions are used.
	__global__ void crossEntropyBatchDerivBeta(int N, int M, float* deltas, const float* A, const int* indices, float beta) {
		extern __shared__ float sdata[];

		// i = example index, each block handles one example
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int target = indices[i];

		// Phase 1: Find maximum value across M classes for this example
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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

		// Phase 2: Compute exponentials and sum into deltas
		float thread_sum = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float exp_val = expf(A[j * N + i] - global_max);
			deltas[j * N + i] = exp_val;
			thread_sum += exp_val;
		}

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

		// Phase 3: Normalize to softmax
		float inv_sum = 1.0f / global_sum;
		for (int j = tid; j < M; j += stride) {
			deltas[j * N + i] *= inv_sum;
		}
		__syncthreads();

		// Phase 4: Per-example entropy H = -sum_j p*log(p), block reduction
		float thread_ent = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float p = deltas[j * N + i];
			thread_ent -= p * logf(fmaxf(p, 1.1920929e-7f)); // eps(Float32), matches CPU
		}

		sdata[tid] = thread_ent;
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		float entropy = sdata[0];
		__syncthreads();

		// Phase 5: Apply entropy regularization to all classes (matches CPU calcDeltaOut! semantics)
		if (beta > 0.0f) {
			for (int j = tid; j < M; j += stride) {
				float p = deltas[j * N + i];
				deltas[j * N + i] = p - beta * p * (entropy + logf(fmaxf(p, 1.1920929e-7f)));
			}
			__syncthreads();
		}

		// Phase 6: Subtract one at the target index
		if (tid == 0) {
			deltas[target * N + i] -= 1.0f;
		}
	}

	//batch cross entropy forward loss with entropy regularization (beta) for per-example output
	//indices.  stores the per-example loss -h_target + log(sum_j exp h_j) + beta*H_i at column 0 and
	//zeros out every other entry so that summing the whole output matrix / N gives the mean batch
	//loss.  Fully on the GPU (block reductions only, no host round trip).
	__global__ void crossEntropyBatchLossBeta(int N, int M, float* A, const int* indices, float beta) {
		extern __shared__ float sdata[];

		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int target = indices[i];

		// Phase 1: Find maximum value across M classes for this example
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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

		// Phase 2: sum of exponentials
		float thread_sum = 0.0f;
		for (int j = tid; j < M; j += stride) {
			thread_sum += expf(A[j * N + i] - global_max);
		}

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

		// Phase 3: per-example entropy H = -sum_j p*log(p)
		float thread_ent = 0.0f;
		for (int j = tid; j < M; j += stride) {
			float p = expf(A[j * N + i] - global_max) / global_sum;
			thread_ent -= p * logf(fmaxf(p, 1.1920929e-7f));
		}

		sdata[tid] = thread_ent;
		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();
		}

		float entropy = sdata[0];
		__syncthreads();

		// Phase 4: write loss at column 0 and zero out all other columns
		if (tid == 0) {
			A[0 * N + i] = -A[target * N + i] + global_max + logf(global_sum) + beta * entropy;
		}
		for (int j = tid; j < M; j += stride) {
			if (j != 0) {
				A[j * N + i] = 0.0f;
			}
		}
	}

	//gather the output-index values for a batch forward cost: copies A[target*N+i] into A[0*N+i]
	//and zeros the rest so that sum(A)/N is the mean of the selected output activations for the
	//OutputIndex loss type.  Fully on the GPU.
	__global__ void outputIndexBatchGather(int N, int M, float* A, const int* indices) {
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		int target = indices[i];

		if (tid == 0) {
			A[0 * N + i] = A[target * N + i];
		}
		for (int j = tid; j < M; j += stride) {
			if (j != 0) {
				A[j * N + i] = 0.0f;
			}
		}
	}


	__global__ void crossEntropyDistDeriv(int N, float *deltas, const float *a, float* target_dist) {
		extern __shared__ float sdata[];

		int tid = threadIdx.x;
		int stride = blockDim.x;

		// Phase 1: Find maximum value
		float thread_max = -INFINITY;
		for (int i = tid; i < N; i += stride) {
			thread_max = fmaxf(thread_max, a[i]);
		}

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
			thread_sum += expf(a[i] - global_max);
		}

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

		// Phase 3: deltas = softmax - target_dist
		float inv_sum = 1.0f / global_sum;
		for (int i = tid; i < N; i += stride) {
			deltas[i] = expf(a[i] - global_max) * inv_sum - target_dist[i];
		}
	}

	__global__ void crossEntropyDistBatchDeriv(int N, int M, float* deltas, const float* A, const float* target_dists) {
		extern __shared__ float sdata[];

		// i = example index, each block handles one example
		int i = blockIdx.x;
		int tid = threadIdx.x;
		int stride = blockDim.x;

		if (i >= N) return;

		// Phase 1: Find maximum value across M classes for this example
		float thread_max = -INFINITY;
		for (int j = tid; j < M; j += stride) {
			thread_max = fmaxf(thread_max, A[j * N + i]);
		}

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
		for (int j = tid; j < M; j += stride) {
			thread_sum += expf(A[j * N + i] - global_max);
		}

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

		// Phase 3: deltas = softmax - target_dists
		float inv_sum = 1.0f / global_sum;
		for (int j = tid; j < M; j += stride) {
			deltas[j * N + i] = expf(A[j * N + i] - global_max) * inv_sum - target_dists[j * N + i];
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

	__global__ void absErrIndexDeriv(int N, int M, float *A, float *Y, int* indx, float *out)
	{	
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			if (j == indx[i]) {
				out[index] = copysignf(1.0, __fsub_rn(A[index], Y[i]));
			}
			else {
				out[index] = 0.0;
			}
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

	__global__ void rowMul(int N, int M, float *X, float *scales)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			float scale = scales[i];
			X[index] = __fmul_rn(X[index], scale);
		}
	}
}