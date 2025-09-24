
extern "C"  
{

	__global__ void updateParams(int N, float alpha, float beta1, float beta2, float t, float *PARAMS, float *GRADS, float *m, float *v)
    {
       
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			float beta1r = __fsub_rn(1.0, beta1);
			float alphar = __fmul_rn(-alpha, __frcp_rn(__fsub_rn(1.0, __powf(beta1, t))));
			m[index] = __fmaf_rn(beta1, m[index], __fmul_rn(beta1r, GRADS[index]));
			v[index] = fmaxf(fmaxf(__fmul_rn(beta2, v[index]), fabsf(GRADS[index])), 1.0e-16);
			PARAMS[index] = __fmaf_rn(alphar,__fdividef(m[index], v[index]), PARAMS[index]);
			//m[index] = beta1*m[index] + (1 - beta1)*GRADS[index];
			
			//float a = beta2*v[index];
			// float b = ((GRADS[index])>(0))?(GRADS[index]):(-GRADS[index]);
			//float c = fmaxf(a, fabsf(GRADS[index])); // ((a)>(fabsf(GRADS[index]))?(a):(b);
			//v[index] = fmaxf(c, 1.0e-16); // ((c)>(1.0e-16))?(c):(1.0e-16);
			//float tmp = alpha/(1.0-powf(beta1, t));
			//PARAMS[index] = PARAMS[index] - (alpha/(1.0-__powf(beta1, t)))*m[index]/v[index];
			//PARAMS[index] = tmp*m[index]/v[index];
		}
	}

	__global__ void updateEst(int N, float beta2, float scale, float *PARAMS, float *AVG, float *EST)
    {
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{	
			float beta2a = __fsub_rn(1.0, beta2);
			//AVG[index] = beta2*AVG[index] + (1.0-beta2)*PARAMS[index];
			//EST[index] = scale*AVG[index];
			AVG[index] = __fmaf_rn(beta2a,PARAMS[index],__fmul_rn(beta2,AVG[index]));
			EST[index] = __fmul_rn(scale, AVG[index]);	
		}
	}
	


	__global__ void elSq(int N, float *Mat)
    {
       
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			Mat[index] = __fmul_rn(Mat[index], Mat[index]); 
		}
	}

	__global__ void elSq2(int N, float *In, float *Out)
    {
       
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int stride = blockDim.x * gridDim.x;	
		
		for (int index = i; index < N; index += stride)
		{
			Out[index] = __fmul_rn(In[index], In[index]); 
		}
	}
	
	
	__global__ void scaleParams(int N, int M, float c, float *Mat, float *F)
    {
       
		int i = blockIdx.x * blockDim.x + threadIdx.x;	
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		
		int index = j*N + i;
		
		if (i < N && j < M)
		{
			float s = __saturatef( __fdividef(c, __fsqrt_rn(F[i])));
			//float s = (c/sqrt(F[i]) < 1.0) ? c/sqrt(F[i]) : 1.0;
			Mat[index] = __fmul_rn(Mat[index], s); 
		}
	}
	
}