#=*
* Check if CUDA runtime v10.0 functions are wrapped properly.
*
* Copyright (C) 2018-2020 Blackbody Research LLC
*       Author: Qijia (Michael) Jin
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*=#

println("cudaDeviceReset(): ", unsafe_string(cudaGetErrorName(cudaDeviceReset())))
@test (cudaDeviceReset() == cudaSuccess)


println("cudaDeviceSynchronize(): ", unsafe_string(cudaGetErrorName(cudaDeviceSynchronize())))
@test (cudaDeviceSynchronize() == cudaSuccess)


println("cudaDeviceSetLimit(): ", unsafe_string(cudaGetErrorName(cudaDeviceSetLimit(typemax(Cuint), typemax(Csize_t)))))
@test (cudaDeviceSetLimit(typemax(Cuint), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaDeviceGetLimit(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetLimit(Ptr{Csize_t}(C_NULL), typemax(Cuint)))))
@test (cudaDeviceGetLimit(Ptr{Csize_t}(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaDeviceGetCacheConfig(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetCacheConfig(Ptr{Cuint}(C_NULL)))))
@test (cudaDeviceGetCacheConfig(Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDeviceGetStreamPriorityRange(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetStreamPriorityRange(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL)))))
@test (cudaDeviceGetStreamPriorityRange(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL)) == cudaSuccess)


println("cudaDeviceSetCacheConfig(): ", unsafe_string(cudaGetErrorName(cudaDeviceSetCacheConfig(typemax(Cuint)))))
@test (cudaDeviceSetCacheConfig(typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaDeviceGetSharedMemConfig(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetSharedMemConfig(Ptr{Cuint}(C_NULL)))))
@test (cudaDeviceGetSharedMemConfig(Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDeviceSetSharedMemConfig(): ", unsafe_string(cudaGetErrorName(cudaDeviceSetSharedMemConfig(typemax(Cuint)))))
@test (cudaDeviceSetSharedMemConfig(typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaDeviceGetByPCIBusId(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetByPCIBusId(Ptr{Cint}(C_NULL), Ptr{UInt8}(C_NULL)))))
@test (cudaDeviceGetByPCIBusId(Ptr{Cint}(C_NULL), Ptr{UInt8}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDeviceGetPCIBusId(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetPCIBusId(Ptr{UInt8}(C_NULL), typemax(Cint), typemax(Cint)))))
@test (cudaDeviceGetPCIBusId(Ptr{UInt8}(C_NULL), typemax(Cint), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaIpcGetEventHandle(): ", unsafe_string(cudaGetErrorName(cudaIpcGetEventHandle(Ptr{cudaIpcEventHandle_t}(C_NULL), cudaEvent_t(C_NULL)))))
@test (cudaIpcGetEventHandle(Ptr{cudaIpcEventHandle_t}(C_NULL), cudaEvent_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaIpcOpenEventHandle(): ", unsafe_string(cudaGetErrorName(cudaIpcOpenEventHandle(Ptr{cudaEvent_t}(C_NULL), zero(cudaIpcEventHandle_t)))))
@test (cudaIpcOpenEventHandle(Ptr{cudaEvent_t}(C_NULL), zero(cudaIpcEventHandle_t)) == cudaErrorInvalidValue)


println("cudaIpcGetMemHandle(): ", unsafe_string(cudaGetErrorName(cudaIpcGetMemHandle(Ptr{cudaIpcMemHandle_t}(C_NULL), C_NULL))))
@test (cudaIpcGetMemHandle(Ptr{cudaIpcMemHandle_t}(C_NULL), C_NULL) == cudaErrorInvalidValue)


println("cudaIpcOpenMemHandle(): ", unsafe_string(cudaGetErrorName(cudaIpcOpenMemHandle(Ptr{Ptr{Cvoid}}(C_NULL), zero(cudaIpcMemHandle_t), typemax(Cuint)))))
@test (cudaIpcOpenMemHandle(Ptr{Ptr{Cvoid}}(C_NULL), zero(cudaIpcMemHandle_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaIpcCloseMemHandle(): ", unsafe_string(cudaGetErrorName(cudaIpcCloseMemHandle(C_NULL))))
@test (cudaIpcCloseMemHandle(C_NULL) == cudaErrorNotSupported)


println("cudaThreadExit(): ", unsafe_string(cudaGetErrorName(cudaThreadExit())))
@test (cudaThreadExit() == cudaSuccess)


println("cudaThreadSynchronize(): ", unsafe_string(cudaGetErrorName(cudaThreadSynchronize())))
@test (cudaThreadSynchronize() == cudaSuccess)


println("cudaThreadSetLimit(): ", unsafe_string(cudaGetErrorName(cudaThreadSetLimit(typemax(Cuint), typemax(Csize_t)))))
@test (cudaThreadSetLimit(typemax(Cuint), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaThreadGetLimit(): ", unsafe_string(cudaGetErrorName(cudaThreadGetLimit(Ptr{Csize_t}(C_NULL), typemax(Cuint)))))
@test (cudaThreadGetLimit(Ptr{Csize_t}(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaThreadGetCacheConfig(): ", unsafe_string(cudaGetErrorName(cudaThreadGetCacheConfig(Ptr{Cuint}(C_NULL)))))
@test (cudaThreadGetCacheConfig(Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaThreadSetCacheConfig(): ", unsafe_string(cudaGetErrorName(cudaThreadSetCacheConfig(typemax(Cuint)))))
@test (cudaThreadSetCacheConfig(typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaGetLastError(): ", unsafe_string(cudaGetErrorName(cudaGetLastError())))
# next error should be 'cudaSuccess'
@test (cudaGetLastError() == cudaSuccess)


println("cudaPeekAtLastError(): ", unsafe_string(cudaGetErrorName(cudaPeekAtLastError())))
@test (cudaPeekAtLastError() == cudaSuccess)


println("cudaGetDeviceCount(): ", unsafe_string(cudaGetErrorName(cudaGetDeviceCount(Ptr{Cint}(C_NULL)))))
@test (cudaGetDeviceCount(Ptr{Cint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGetDeviceProperties(): ", unsafe_string(cudaGetErrorName(cudaGetDeviceProperties(Ptr{cudaDeviceProp}(C_NULL), typemax(Cint)))))
@test (cudaGetDeviceProperties(Ptr{cudaDeviceProp}(C_NULL), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaDeviceGetAttribute(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetAttribute(Ptr{Cint}(C_NULL), typemax(Cuint), typemax(Cint)))))
@test (cudaDeviceGetAttribute(Ptr{Cint}(C_NULL), typemax(Cuint), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaDeviceGetP2PAttribute(): ", unsafe_string(cudaGetErrorName(cudaDeviceGetP2PAttribute(Ptr{Cint}(C_NULL), typemax(Cuint), typemax(Cint), typemax(Cint)))))
@test (cudaDeviceGetP2PAttribute(Ptr{Cint}(C_NULL), typemax(Cuint), typemax(Cint), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaChooseDevice(): ", unsafe_string(cudaGetErrorName(cudaChooseDevice(Ptr{Cint}(C_NULL), Ptr{cudaDeviceProp}(C_NULL)))))
@test (cudaChooseDevice(Ptr{Cint}(C_NULL), Ptr{cudaDeviceProp}(C_NULL)) == cudaErrorInvalidValue)


println("cudaSetDevice(): ", unsafe_string(cudaGetErrorName(cudaSetDevice(typemax(Cint)))))
@test (cudaSetDevice(typemax(Cint)) == cudaErrorInvalidDevice)


println("cudaGetDevice(): ", unsafe_string(cudaGetErrorName(cudaGetDevice(Ptr{Cint}(C_NULL)))))
@test (cudaGetDevice(Ptr{Cint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaSetValidDevices(): ", unsafe_string(cudaGetErrorName(cudaSetValidDevices(Ptr{Cint}(C_NULL), typemax(Cint)))))
@test (cudaSetValidDevices(Ptr{Cint}(C_NULL), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaSetDeviceFlags(): ", unsafe_string(cudaGetErrorName(cudaSetDeviceFlags(typemax(Cuint)))))
@test (cudaSetDeviceFlags(typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaGetDeviceFlags(): ", unsafe_string(cudaGetErrorName(cudaGetDeviceFlags(Ptr{Cuint}(C_NULL)))))
@test (cudaGetDeviceFlags(Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaStreamCreate(): ", unsafe_string(cudaGetErrorName(cudaStreamCreate(Ptr{cudaStream_t}(C_NULL)))))
@test (cudaStreamCreate(Ptr{cudaStream_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaStreamCreateWithFlags(): ", unsafe_string(cudaGetErrorName(cudaStreamCreateWithFlags(Ptr{cudaStream_t}(C_NULL), typemax(Cuint)))))
@test (cudaStreamCreateWithFlags(Ptr{cudaStream_t}(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaStreamCreateWithPriority(): ", unsafe_string(cudaGetErrorName(cudaStreamCreateWithPriority(Ptr{cudaStream_t}(C_NULL), typemax(Cuint), typemax(Cint)))))
@test (cudaStreamCreateWithPriority(Ptr{cudaStream_t}(C_NULL), typemax(Cuint), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaStreamGetPriority(): ", unsafe_string(cudaGetErrorName(cudaStreamGetPriority(cudaStream_t(C_NULL), Ptr{Cint}(C_NULL)))))
@test (cudaStreamGetPriority(cudaStream_t(C_NULL), Ptr{Cint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaStreamGetFlags(): ", unsafe_string(cudaGetErrorName(cudaStreamGetFlags(cudaStream_t(C_NULL), Ptr{Cuint}(C_NULL)))))
@test (cudaStreamGetFlags(cudaStream_t(C_NULL), Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaStreamDestroy(): ", unsafe_string(cudaGetErrorName(cudaStreamDestroy(cudaStream_t(C_NULL)))))
@test (cudaStreamDestroy(cudaStream_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaStreamWaitEvent(): ", unsafe_string(cudaGetErrorName(cudaStreamWaitEvent(cudaStream_t(C_NULL), cudaEvent_t(C_NULL), typemax(Cuint)))))
@test (cudaStreamWaitEvent(cudaStream_t(C_NULL), cudaEvent_t(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaStreamAddCallback(): ", unsafe_string(cudaGetErrorName(cudaStreamAddCallback(cudaStream_t(C_NULL), cudaStreamCallback_t(C_NULL), C_NULL, typemax(Cuint)))))
@test (cudaStreamAddCallback(cudaStream_t(C_NULL), cudaStreamCallback_t(C_NULL), C_NULL, typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaStreamSynchronize(): ", unsafe_string(cudaGetErrorName(cudaStreamSynchronize(cudaStream_t(C_NULL)))))
@test (cudaStreamSynchronize(cudaStream_t(C_NULL)) == cudaSuccess)


println("cudaStreamQuery(): ", unsafe_string(cudaGetErrorName(cudaStreamQuery(cudaStream_t(C_NULL)))))
@test (cudaStreamQuery(cudaStream_t(C_NULL)) == cudaSuccess)


println("cudaStreamAttachMemAsync(): ", unsafe_string(cudaGetErrorName(cudaStreamAttachMemAsync(cudaStream_t(C_NULL), C_NULL, typemax(Csize_t), typemax(Cuint)))))
@test (cudaStreamAttachMemAsync(cudaStream_t(C_NULL), C_NULL, typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaStreamBeginCapture(): ", unsafe_string(cudaGetErrorName(cudaStreamBeginCapture(cudaStream_t(C_NULL)))))
@test (cudaStreamBeginCapture(cudaStream_t(C_NULL)) == cudaErrorStreamCaptureUnsupported)


println("cudaStreamEndCapture(): ", unsafe_string(cudaGetErrorName(cudaStreamEndCapture(cudaStream_t(C_NULL), Ptr{cudaGraph_t}(C_NULL)))))
@test (cudaStreamEndCapture(cudaStream_t(C_NULL), Ptr{cudaGraph_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaStreamIsCapturing(): ", unsafe_string(cudaGetErrorName(cudaStreamIsCapturing(cudaStream_t(C_NULL), zeros(Cuint, 1)))))
@test (cudaStreamIsCapturing(cudaStream_t(C_NULL), zeros(Cuint, 1)) == cudaSuccess)


println("cudaEventCreate(): ", unsafe_string(cudaGetErrorName(cudaEventCreate(Ptr{cudaEvent_t}(C_NULL)))))
@test (cudaEventCreate(Ptr{cudaEvent_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaEventCreateWithFlags(): ", unsafe_string(cudaGetErrorName(cudaEventCreateWithFlags(Ptr{cudaEvent_t}(C_NULL), typemax(Cuint)))))
@test (cudaEventCreateWithFlags(Ptr{cudaEvent_t}(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaEventRecord(): ", unsafe_string(cudaGetErrorName(cudaEventRecord(cudaEvent_t(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaEventRecord(cudaEvent_t(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaEventQuery(): ", unsafe_string(cudaGetErrorName(cudaEventQuery(cudaEvent_t(C_NULL)))))
@test (cudaEventQuery(cudaEvent_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaEventSynchronize(): ", unsafe_string(cudaGetErrorName(cudaEventSynchronize(cudaEvent_t(C_NULL)))))
@test (cudaEventSynchronize(cudaEvent_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaEventDestroy(): ", unsafe_string(cudaGetErrorName(cudaEventDestroy(cudaEvent_t(C_NULL)))))
@test (cudaEventDestroy(cudaEvent_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaEventElapsedTime(): ", unsafe_string(cudaGetErrorName(cudaEventElapsedTime(Ptr{Cfloat}(C_NULL), cudaEvent_t(C_NULL), cudaEvent_t(C_NULL)))))
@test (cudaEventElapsedTime(Ptr{Cfloat}(C_NULL), cudaEvent_t(C_NULL), cudaEvent_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaImportExternalMemory(): ", unsafe_string(cudaGetErrorName(cudaImportExternalMemory(Ptr{cudaExternalMemory_t}(C_NULL), Ptr{cudaExternalMemoryHandleDesc}(C_NULL)))))
@test (cudaImportExternalMemory(Ptr{cudaExternalMemory_t}(C_NULL), Ptr{cudaExternalMemoryHandleDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaExternalMemoryGetMappedBuffer(): ", unsafe_string(cudaGetErrorName(cudaExternalMemoryGetMappedBuffer(Ptr{Ptr{Cvoid}}(C_NULL), cudaExternalMemory_t(C_NULL), Ptr{cudaExternalMemoryBufferDesc}(C_NULL)))))
@test (cudaExternalMemoryGetMappedBuffer(Ptr{Ptr{Cvoid}}(C_NULL), cudaExternalMemory_t(C_NULL), Ptr{cudaExternalMemoryBufferDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaExternalMemoryGetMappedMipmappedArray(): ", unsafe_string(cudaGetErrorName(cudaExternalMemoryGetMappedMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), cudaExternalMemory_t(C_NULL), Ptr{cudaExternalMemoryMipmappedArrayDesc}(C_NULL)))))
@test (cudaExternalMemoryGetMappedMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), cudaExternalMemory_t(C_NULL), Ptr{cudaExternalMemoryMipmappedArrayDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDestroyExternalMemory(): ", unsafe_string(cudaGetErrorName(cudaDestroyExternalMemory(cudaExternalMemory_t(C_NULL)))))
@test (cudaDestroyExternalMemory(cudaExternalMemory_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaImportExternalSemaphore(): ", unsafe_string(cudaGetErrorName(cudaImportExternalSemaphore(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreHandleDesc}(C_NULL)))))
@test (cudaImportExternalSemaphore(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreHandleDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaSignalExternalSemaphoresAsync(): ", unsafe_string(cudaGetErrorName(cudaSignalExternalSemaphoresAsync(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreSignalParams}(C_NULL), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaSignalExternalSemaphoresAsync(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreSignalParams}(C_NULL), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaWaitExternalSemaphoresAsync(): ", unsafe_string(cudaGetErrorName(cudaWaitExternalSemaphoresAsync(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreWaitParams}(C_NULL), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaWaitExternalSemaphoresAsync(Ptr{cudaExternalSemaphore_t}(C_NULL), Ptr{cudaExternalSemaphoreWaitParams}(C_NULL), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaDestroyExternalSemaphore(): ", unsafe_string(cudaGetErrorName(cudaDestroyExternalSemaphore(cudaExternalSemaphore_t(C_NULL)))))
@test (cudaDestroyExternalSemaphore(cudaExternalSemaphore_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaLaunchKernel(): ", unsafe_string(cudaGetErrorName(cudaLaunchKernel(C_NULL, zero(dim3), zero(dim3), Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaLaunchKernel(C_NULL, zero(dim3), zero(dim3), Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaErrorInvalidDeviceFunction)


println("cudaLaunchCooperativeKernel(): ", unsafe_string(cudaGetErrorName(cudaLaunchCooperativeKernel(C_NULL, zero(dim3), zero(dim3), Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaLaunchCooperativeKernel(C_NULL, zero(dim3), zero(dim3), Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaErrorInvalidDeviceFunction)


println("cudaLaunchCooperativeKernelMultiDevice(): ", unsafe_string(cudaGetErrorName(cudaLaunchCooperativeKernelMultiDevice(Ptr{cudaLaunchParams}(C_NULL), typemax(Cuint), typemax(Cuint)))))
@test (cudaLaunchCooperativeKernelMultiDevice(Ptr{cudaLaunchParams}(C_NULL), typemax(Cuint), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaFuncSetCacheConfig(): ", unsafe_string(cudaGetErrorName(cudaFuncSetCacheConfig(C_NULL, typemax(Cuint)))))
@test (cudaFuncSetCacheConfig(C_NULL, typemax(Cuint)) == cudaErrorInvalidDeviceFunction)


println("cudaFuncSetSharedMemConfig(): ", unsafe_string(cudaGetErrorName(cudaFuncSetSharedMemConfig(C_NULL, typemax(Cuint)))))
@test (cudaFuncSetSharedMemConfig(C_NULL, typemax(Cuint)) == cudaErrorInvalidDeviceFunction)


println("cudaFuncGetAttributes(): ", unsafe_string(cudaGetErrorName(cudaFuncGetAttributes(Ptr{cudaFuncAttributes}(C_NULL), C_NULL))))
@test (cudaFuncGetAttributes(Ptr{cudaFuncAttributes}(C_NULL), C_NULL) == cudaErrorInvalidValue)


println("cudaFuncSetAttribute(): ", unsafe_string(cudaGetErrorName(cudaFuncSetAttribute(C_NULL, typemax(Cuint), typemax(Cint)))))
@test (cudaFuncSetAttribute(C_NULL, typemax(Cuint), typemax(Cint)) == cudaErrorInvalidDeviceFunction)


println("cudaSetDoubleForDevice(): ", unsafe_string(cudaGetErrorName(cudaSetDoubleForDevice(Ptr{Cdouble}(C_NULL)))))
@test (cudaSetDoubleForDevice(Ptr{Cdouble}(C_NULL)) == cudaSuccess)


println("cudaSetDoubleForHost(): ", unsafe_string(cudaGetErrorName(cudaSetDoubleForHost(Ptr{Cdouble}(C_NULL)))))
@test (cudaSetDoubleForHost(Ptr{Cdouble}(C_NULL)) == cudaSuccess)


println("cudaLaunchHostFunc(): ", unsafe_string(cudaGetErrorName(cudaLaunchHostFunc(cudaStream_t(C_NULL), cudaHostFn_t(C_NULL), C_NULL))))
@test (cudaLaunchHostFunc(cudaStream_t(C_NULL), cudaHostFn_t(C_NULL), C_NULL) == cudaErrorInvalidValue)


println("cudaOccupancyMaxActiveBlocksPerMultiprocessor(): ", unsafe_string(cudaGetErrorName(cudaOccupancyMaxActiveBlocksPerMultiprocessor(Ptr{Cint}(C_NULL), C_NULL, typemax(Cint), typemax(Csize_t)))))
@test (cudaOccupancyMaxActiveBlocksPerMultiprocessor(Ptr{Cint}(C_NULL), C_NULL, typemax(Cint), typemax(Csize_t)) == cudaErrorInvalidDeviceFunction)


println("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(): ", unsafe_string(cudaGetErrorName(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(Ptr{Cint}(C_NULL), C_NULL, typemax(Cint), typemax(Csize_t), typemax(Cuint)))))
@test (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(Ptr{Cint}(C_NULL), C_NULL, typemax(Cint), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidDeviceFunction)


println("cudaConfigureCall(): ", unsafe_string(cudaGetErrorName(cudaConfigureCall(zero(dim3), zero(dim3), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaConfigureCall(zero(dim3), zero(dim3), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaSuccess)


println("cudaSetupArgument(): ", unsafe_string(cudaGetErrorName(cudaSetupArgument(C_NULL, typemax(Csize_t), typemax(Csize_t)))))
@test (cudaSetupArgument(C_NULL, typemax(Csize_t), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaLaunch(): ", unsafe_string(cudaGetErrorName(cudaLaunch(C_NULL))))
@test (cudaLaunch(C_NULL) == cudaErrorInvalidDeviceFunction)


println("cudaMallocManaged(): ", unsafe_string(cudaGetErrorName(cudaMallocManaged(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMallocManaged(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaMalloc(): ", unsafe_string(cudaGetErrorName(cudaMalloc(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t)))))
@test (cudaMalloc(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMallocHost(): ", unsafe_string(cudaGetErrorName(cudaMallocHost(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t)))))
@test (cudaMallocHost(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMallocPitch(): ", unsafe_string(cudaGetErrorName(cudaMallocPitch(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), typemax(Csize_t), typemax(Csize_t)))))
@test (cudaMallocPitch(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), typemax(Csize_t), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMallocArray(): ", unsafe_string(cudaGetErrorName(cudaMallocArray(Ptr{cudaArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMallocArray(Ptr{cudaArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaFree(): ", unsafe_string(cudaGetErrorName(cudaFree(C_NULL))))
@test (cudaFree(C_NULL) == cudaSuccess)


println("cudaFreeHost(): ", unsafe_string(cudaGetErrorName(cudaFreeHost(C_NULL))))
@test (cudaFreeHost(C_NULL) == cudaSuccess)


println("cudaFreeArray(): ", unsafe_string(cudaGetErrorName(cudaFreeArray(cudaArray_t(C_NULL)))))
@test (cudaFreeArray(cudaArray_t(C_NULL)) == cudaSuccess)


println("cudaFreeMipmappedArray(): ", unsafe_string(cudaGetErrorName(cudaFreeMipmappedArray(cudaMipmappedArray_t(C_NULL)))))
@test (cudaFreeMipmappedArray(cudaMipmappedArray_t(C_NULL)) == cudaSuccess)


println("cudaHostAlloc(): ", unsafe_string(cudaGetErrorName(cudaHostAlloc(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), typemax(Cuint)))))
@test (cudaHostAlloc(Ptr{Ptr{Cvoid}}(C_NULL), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaHostRegister(): ", unsafe_string(cudaGetErrorName(cudaHostRegister(C_NULL, typemax(Csize_t), typemax(Cuint)))))
@test (cudaHostRegister(C_NULL, typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaHostUnregister(): ", unsafe_string(cudaGetErrorName(cudaHostUnregister(C_NULL))))
@test (cudaHostUnregister(C_NULL) == cudaErrorInvalidValue)


println("cudaHostGetDevicePointer(): ", unsafe_string(cudaGetErrorName(cudaHostGetDevicePointer(Ptr{Ptr{Cvoid}}(C_NULL), C_NULL, typemax(Cuint)))))
@test (cudaHostGetDevicePointer(Ptr{Ptr{Cvoid}}(C_NULL), C_NULL, typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaHostGetFlags(): ", unsafe_string(cudaGetErrorName(cudaHostGetFlags(Ptr{Cuint}(C_NULL), C_NULL))))
@test (cudaHostGetFlags(Ptr{Cuint}(C_NULL), C_NULL) == cudaErrorInvalidValue)


println("cudaMalloc3D(): ", unsafe_string(cudaGetErrorName(cudaMalloc3D(Ptr{cudaPitchedPtr}(C_NULL), zero(cudaExtent)))))
@test (cudaMalloc3D(Ptr{cudaPitchedPtr}(C_NULL), zero(cudaExtent)) == cudaErrorInvalidValue)


println("cudaMalloc3DArray(): ", unsafe_string(cudaGetErrorName(cudaMalloc3DArray(Ptr{cudaArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), zero(cudaExtent), typemax(Cuint)))))
@test (cudaMalloc3DArray(Ptr{cudaArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), zero(cudaExtent), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaMallocMipmappedArray(): ", unsafe_string(cudaGetErrorName(cudaMallocMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), zero(cudaExtent), typemax(Cuint), typemax(Cuint)))))
@test (cudaMallocMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL), zero(cudaExtent), typemax(Cuint), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaGetMipmappedArrayLevel(): ", unsafe_string(cudaGetErrorName(cudaGetMipmappedArrayLevel(Ptr{cudaArray_t}(C_NULL), cudaMipmappedArray_const_t(C_NULL), typemax(Cuint)))))
@test (cudaGetMipmappedArrayLevel(Ptr{cudaArray_t}(C_NULL), cudaMipmappedArray_const_t(C_NULL), typemax(Cuint)) == cudaErrorInvalidResourceHandle)


println("cudaMemcpy3D(): ", unsafe_string(cudaGetErrorName(cudaMemcpy3D(Ptr{cudaMemcpy3DParms}(C_NULL)))))
@test (cudaMemcpy3D(Ptr{cudaMemcpy3DParms}(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemcpy3DPeer(): ", unsafe_string(cudaGetErrorName(cudaMemcpy3DPeer(Ptr{cudaMemcpy3DPeerParms}(C_NULL)))))
@test (cudaMemcpy3DPeer(Ptr{cudaMemcpy3DPeerParms}(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemcpy3DAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpy3DAsync(Ptr{cudaMemcpy3DParms}(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaMemcpy3DAsync(Ptr{cudaMemcpy3DParms}(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemcpy3DPeerAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpy3DPeerAsync(Ptr{cudaMemcpy3DPeerParms}(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaMemcpy3DPeerAsync(Ptr{cudaMemcpy3DPeerParms}(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemGetInfo(): ", unsafe_string(cudaGetErrorName(cudaMemGetInfo(Ptr{Csize_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaMemGetInfo(Ptr{Csize_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaSuccess)


println("cudaArrayGetInfo(): ", unsafe_string(cudaGetErrorName(cudaArrayGetInfo(Ptr{cudaChannelFormatDesc}(C_NULL), Ptr{cudaExtent}(C_NULL), Ptr{Cuint}(C_NULL), cudaArray_t(C_NULL)))))
@test (cudaArrayGetInfo(Ptr{cudaChannelFormatDesc}(C_NULL), Ptr{cudaExtent}(C_NULL), Ptr{Cuint}(C_NULL), cudaArray_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaMemcpy(): ", unsafe_string(cudaGetErrorName(cudaMemcpy(C_NULL, C_NULL, typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpy(C_NULL, C_NULL, typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyPeer(): ", unsafe_string(cudaGetErrorName(cudaMemcpyPeer(C_NULL, typemax(Cint), C_NULL, typemax(Cint), typemax(Csize_t)))))
@test (cudaMemcpyPeer(C_NULL, typemax(Cint), C_NULL, typemax(Cint), typemax(Csize_t)) == cudaErrorInvalidDevice)


println("cudaMemcpyToArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpyToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), Cuint(0)))))
@test (cudaMemcpyToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), Cuint(0)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyFromArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpyFromArray(C_NULL, cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)))))
@test (cudaMemcpyFromArray(C_NULL, cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyArrayToArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpyArrayToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpyArrayToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpy2D(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2D(C_NULL, typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpy2D(C_NULL, typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaMemcpy2DToArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)))))
@test (cudaMemcpy2DToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpy2DFromArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DFromArray(C_NULL, typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)))))
@test (cudaMemcpy2DFromArray(C_NULL, typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpy2DArrayToArray(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DArrayToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpy2DArrayToArray(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyToSymbol(): ", unsafe_string(cudaGetErrorName(cudaMemcpyToSymbol(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpyToSymbol(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidSymbol)


println("cudaMemcpyFromSymbol(): ", unsafe_string(cudaGetErrorName(cudaMemcpyFromSymbol(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint)))))
@test (cudaMemcpyFromSymbol(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint)) == cudaErrorInvalidSymbol)


println("cudaMemcpyAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)))))
@test (cudaMemcpyAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyPeerAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyPeerAsync(C_NULL, typemax(Cint), C_NULL, typemax(Cint), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaMemcpyPeerAsync(C_NULL, typemax(Cint), C_NULL, typemax(Cint), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaErrorInvalidDevice)


println("cudaMemcpyToArrayAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyToArrayAsync(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaMemcpyToArrayAsync(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyFromArrayAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyFromArrayAsync(C_NULL, cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaMemcpyFromArrayAsync(C_NULL, cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpy2DAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DAsync(C_NULL, typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)))))
@test (cudaMemcpy2DAsync(C_NULL, typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemcpy2DToArrayAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DToArrayAsync(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaMemcpy2DToArrayAsync(cudaArray_t(C_NULL), typemax(Csize_t), typemax(Csize_t), C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpy2DFromArrayAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpy2DFromArrayAsync(C_NULL, typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)))))
@test (cudaMemcpy2DFromArrayAsync(C_NULL, typemax(Csize_t), cudaArray_const_t(C_NULL), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), typemax(Csize_t), Cuint(0), cudaStream_t(C_NULL)) == cudaErrorInvalidMemcpyDirection)


println("cudaMemcpyToSymbolAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyToSymbolAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)))))
@test (cudaMemcpyToSymbolAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)) == cudaErrorInvalidSymbol)


println("cudaMemcpyFromSymbolAsync(): ", unsafe_string(cudaGetErrorName(cudaMemcpyFromSymbolAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)))))
@test (cudaMemcpyFromSymbolAsync(C_NULL, C_NULL, typemax(Csize_t), typemax(Csize_t), typemax(Cuint), cudaStream_t(C_NULL)) == cudaErrorInvalidSymbol)


println("cudaMemset(): ", unsafe_string(cudaGetErrorName(cudaMemset(C_NULL, typemax(Cint), typemax(Csize_t)))))
@test (cudaMemset(C_NULL, typemax(Cint), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMemset2D(): ", unsafe_string(cudaGetErrorName(cudaMemset2D(C_NULL, typemax(Csize_t), typemax(Cint), typemax(Csize_t), typemax(Csize_t)))))
@test (cudaMemset2D(C_NULL, typemax(Csize_t), typemax(Cint), typemax(Csize_t), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMemset3D(): ", unsafe_string(cudaGetErrorName(cudaMemset3D(zero(cudaPitchedPtr), typemax(Cint), cudaExtent(typemax(Csize_t), typemax(Csize_t), typemax(Csize_t))))))
@test (cudaMemset3D(zero(cudaPitchedPtr), typemax(Cint), cudaExtent(typemax(Csize_t), typemax(Csize_t), typemax(Csize_t))) == cudaErrorInvalidValue)


println("cudaMemsetAsync(): ", unsafe_string(cudaGetErrorName(cudaMemsetAsync(C_NULL, typemax(Cint), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaMemsetAsync(C_NULL, typemax(Cint), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemset2DAsync(): ", unsafe_string(cudaGetErrorName(cudaMemset2DAsync(C_NULL, typemax(Csize_t), typemax(Cint), typemax(Csize_t), typemax(Csize_t), cudaStream_t(C_NULL)))))
@test (cudaMemset2DAsync(C_NULL, typemax(Csize_t), typemax(Cint), typemax(Csize_t), typemax(Csize_t), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemset3DAsync(): ", unsafe_string(cudaGetErrorName(cudaMemset3DAsync(zero(cudaPitchedPtr), typemax(Cint), cudaExtent(typemax(Csize_t), typemax(Csize_t), typemax(Csize_t)), cudaStream_t(C_NULL)))))
@test (cudaMemset3DAsync(zero(cudaPitchedPtr), typemax(Cint), cudaExtent(typemax(Csize_t), typemax(Csize_t), typemax(Csize_t)), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGetSymbolAddress(): ", unsafe_string(cudaGetErrorName(cudaGetSymbolAddress(Ptr{Ptr{Cvoid}}(C_NULL), C_NULL))))
@test (cudaGetSymbolAddress(Ptr{Ptr{Cvoid}}(C_NULL), C_NULL) == cudaErrorInvalidSymbol)


println("cudaGetSymbolSize(): ", unsafe_string(cudaGetErrorName(cudaGetSymbolSize(Ptr{Csize_t}(C_NULL), C_NULL))))
@test (cudaGetSymbolSize(Ptr{Csize_t}(C_NULL), C_NULL) == cudaErrorInvalidSymbol)


println("cudaMemPrefetchAsync(): ", unsafe_string(cudaGetErrorName(cudaMemPrefetchAsync(C_NULL, typemax(Csize_t), typemax(Cint), cudaStream_t(C_NULL)))))
@test (cudaMemPrefetchAsync(C_NULL, typemax(Csize_t), typemax(Cint), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaMemAdvise(): ", unsafe_string(cudaGetErrorName(cudaMemAdvise(C_NULL, typemax(Csize_t), typemax(Cuint), typemax(Cint)))))
@test (cudaMemAdvise(C_NULL, typemax(Csize_t), typemax(Cuint), typemax(Cint)) == cudaErrorInvalidValue)


println("cudaMemRangeGetAttribute(): ", unsafe_string(cudaGetErrorName(cudaMemRangeGetAttribute(C_NULL, typemax(Csize_t), typemax(Cuint), C_NULL, typemax(Csize_t)))))
@test (cudaMemRangeGetAttribute(C_NULL, typemax(Csize_t), typemax(Cuint), C_NULL, typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaMemRangeGetAttributes(): ", unsafe_string(cudaGetErrorName(cudaMemRangeGetAttributes(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), Ptr{Cuint}(C_NULL), typemax(Csize_t), C_NULL, typemax(Csize_t)))))
@test (cudaMemRangeGetAttributes(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), Ptr{Cuint}(C_NULL), typemax(Csize_t), C_NULL, typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaPointerGetAttributes(): ", unsafe_string(cudaGetErrorName(cudaPointerGetAttributes(Ptr{cudaPointerAttributes}(C_NULL), C_NULL))))
@test (cudaPointerGetAttributes(Ptr{cudaPointerAttributes}(C_NULL), C_NULL) == cudaErrorInvalidValue)


println("cudaDeviceCanAccessPeer(): ", unsafe_string(cudaGetErrorName(cudaDeviceCanAccessPeer(Ptr{Cint}(C_NULL), typemax(Cint), typemax(Cint)))))
@test (cudaDeviceCanAccessPeer(Ptr{Cint}(C_NULL), typemax(Cint), typemax(Cint)) == cudaErrorInvalidDevice)


println("cudaDeviceEnablePeerAccess(): ", unsafe_string(cudaGetErrorName(cudaDeviceEnablePeerAccess(typemax(Cint), typemax(Cuint)))))
@test (cudaDeviceEnablePeerAccess(typemax(Cint), typemax(Cuint)) == cudaErrorInvalidDevice)


println("cudaDeviceDisablePeerAccess(): ", unsafe_string(cudaGetErrorName(cudaDeviceDisablePeerAccess(typemax(Cint)))))
@test (cudaDeviceDisablePeerAccess(typemax(Cint)) == cudaErrorInvalidDevice)


println("cudaGraphicsUnregisterResource(): ", unsafe_string(cudaGetErrorName(cudaGraphicsUnregisterResource(cudaGraphicsResource_t(C_NULL)))))
@test (cudaGraphicsUnregisterResource(cudaGraphicsResource_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaGraphicsResourceSetMapFlags(): ", unsafe_string(cudaGetErrorName(cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t(C_NULL), typemax(Cuint)))))
@test (cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t(C_NULL), typemax(Cuint)) == cudaErrorInvalidResourceHandle)


println("cudaGraphicsMapResources(): ", unsafe_string(cudaGetErrorName(cudaGraphicsMapResources(typemax(Cint), Ptr{cudaGraphicsResource_t}(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaGraphicsMapResources(typemax(Cint), Ptr{cudaGraphicsResource_t}(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphicsUnmapResources(): ", unsafe_string(cudaGetErrorName(cudaGraphicsUnmapResources(typemax(Cint), Ptr{cudaGraphicsResource_t}(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaGraphicsUnmapResources(typemax(Cint), Ptr{cudaGraphicsResource_t}(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphicsResourceGetMappedPointer(): ", unsafe_string(cudaGetErrorName(cudaGraphicsResourceGetMappedPointer(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), cudaGraphicsResource_t(C_NULL)))))
@test (cudaGraphicsResourceGetMappedPointer(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), cudaGraphicsResource_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaGraphicsSubResourceGetMappedArray(): ", unsafe_string(cudaGetErrorName(cudaGraphicsSubResourceGetMappedArray(Ptr{cudaArray_t}(C_NULL), cudaGraphicsResource_t(C_NULL), typemax(Cuint), typemax(Cuint)))))
@test (cudaGraphicsSubResourceGetMappedArray(Ptr{cudaArray_t}(C_NULL), cudaGraphicsResource_t(C_NULL), typemax(Cuint), typemax(Cuint)) == cudaErrorInvalidResourceHandle)


println("cudaGraphicsResourceGetMappedMipmappedArray(): ", unsafe_string(cudaGetErrorName(cudaGraphicsResourceGetMappedMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), cudaGraphicsResource_t(C_NULL)))))
@test (cudaGraphicsResourceGetMappedMipmappedArray(Ptr{cudaMipmappedArray_t}(C_NULL), cudaGraphicsResource_t(C_NULL)) == cudaErrorInvalidResourceHandle)


println("cudaGetChannelDesc(): ", unsafe_string(cudaGetErrorName(cudaGetChannelDesc(Ptr{cudaChannelFormatDesc}(C_NULL), cudaArray_const_t(C_NULL)))))
@test (cudaGetChannelDesc(Ptr{cudaChannelFormatDesc}(C_NULL), cudaArray_const_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaCreateChannelDesc(): ", cudaCreateChannelDesc(typemax(Cint), typemax(Cint), typemax(Cint), typemax(Cint), Cuint(0)))
@test (cudaCreateChannelDesc(typemax(Cint), typemax(Cint), typemax(Cint), typemax(Cint), Cuint(0)) == cudaChannelFormatDesc(typemax(Cint), typemax(Cint), typemax(Cint), typemax(Cint), Cuint(0)))


println("cudaBindTexture(): ", unsafe_string(cudaGetErrorName(cudaBindTexture(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL), C_NULL, Ptr{cudaChannelFormatDesc}(C_NULL), typemax(Csize_t)))))
@test (cudaBindTexture(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL), C_NULL, Ptr{cudaChannelFormatDesc}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidTexture)


println("cudaBindTexture2D(): ", unsafe_string(cudaGetErrorName(cudaBindTexture2D(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL), C_NULL, Ptr{cudaChannelFormatDesc}(C_NULL), Csize_t(0), Csize_t(0), Csize_t(0)))))
@test (cudaBindTexture2D(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL), C_NULL, Ptr{cudaChannelFormatDesc}(C_NULL), Csize_t(0), Csize_t(0), Csize_t(0)) == cudaErrorInvalidValue)


println("cudaBindTextureToArray(): ", unsafe_string(cudaGetErrorName(cudaBindTextureToArray(Ptr{textureReference}(C_NULL), cudaArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)))))
@test (cudaBindTextureToArray(Ptr{textureReference}(C_NULL), cudaArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)) == cudaErrorInvalidTexture)


println("cudaBindTextureToMipmappedArray(): ", unsafe_string(cudaGetErrorName(cudaBindTextureToMipmappedArray(Ptr{textureReference}(C_NULL), cudaMipmappedArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)))))
@test (cudaBindTextureToMipmappedArray(Ptr{textureReference}(C_NULL), cudaMipmappedArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)) == cudaErrorInvalidTexture)


println("cudaUnbindTexture(): ", unsafe_string(cudaGetErrorName(cudaUnbindTexture(Ptr{textureReference}(C_NULL)))))
@test (cudaUnbindTexture(Ptr{textureReference}(C_NULL)) == cudaErrorInvalidTexture)


println("cudaGetTextureAlignmentOffset(): ", unsafe_string(cudaGetErrorName(cudaGetTextureAlignmentOffset(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL)))))
@test (cudaGetTextureAlignmentOffset(Ptr{Csize_t}(C_NULL), Ptr{textureReference}(C_NULL)) == cudaErrorInvalidTexture)


println("cudaGetTextureReference(): ", unsafe_string(cudaGetErrorName(cudaGetTextureReference(Base.unsafe_convert(Ptr{Ptr{textureReference}}, Base.cconvert(Ref{Ptr{textureReference}}, [Ptr{textureReference}(C_NULL)])), C_NULL))))
@test (cudaGetTextureReference(Base.unsafe_convert(Ptr{Ptr{textureReference}}, Base.cconvert(Ref{Ptr{textureReference}}, [Ptr{textureReference}(C_NULL)])), C_NULL) == cudaErrorInvalidTexture)


println("cudaBindSurfaceToArray(): ", unsafe_string(cudaGetErrorName(cudaBindSurfaceToArray(Ptr{surfaceReference}(C_NULL), cudaArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)))))
@test (cudaBindSurfaceToArray(Ptr{surfaceReference}(C_NULL), cudaArray_const_t(C_NULL), Ptr{cudaChannelFormatDesc}(C_NULL)) == cudaErrorInvalidSurface)


println("cudaGetSurfaceReference(): ", unsafe_string(cudaGetErrorName(cudaGetSurfaceReference(Base.unsafe_convert(Ptr{Ptr{surfaceReference}}, Base.cconvert(Ref{Ptr{surfaceReference}}, [Ptr{surfaceReference}(C_NULL)])), C_NULL))))
@test (cudaGetSurfaceReference(Base.unsafe_convert(Ptr{Ptr{surfaceReference}}, Base.cconvert(Ref{Ptr{surfaceReference}}, [Ptr{surfaceReference}(C_NULL)])), C_NULL) == cudaErrorInvalidSurface)


println("cudaCreateTextureObject(): ", unsafe_string(cudaGetErrorName(cudaCreateTextureObject(Ptr{cudaTextureObject_t}(C_NULL), Ptr{cudaResourceDesc}(C_NULL), Ptr{cudaTextureDesc}(C_NULL), Ptr{cudaResourceViewDesc}(C_NULL)))))
@test (cudaCreateTextureObject(Ptr{cudaTextureObject_t}(C_NULL), Ptr{cudaResourceDesc}(C_NULL), Ptr{cudaTextureDesc}(C_NULL), Ptr{cudaResourceViewDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDestroyTextureObject(): ", unsafe_string(cudaGetErrorName(cudaDestroyTextureObject(cudaTextureObject_t(0)))))
@test (cudaDestroyTextureObject(cudaTextureObject_t(0)) == cudaSuccess)


println("cudaGetTextureObjectResourceDesc(): ", unsafe_string(cudaGetErrorName(cudaGetTextureObjectResourceDesc(Ptr{cudaResourceDesc}(C_NULL), cudaTextureObject_t(0)))))
@test (cudaGetTextureObjectResourceDesc(Ptr{cudaResourceDesc}(C_NULL), cudaTextureObject_t(0)) == cudaErrorInvalidValue)


println("cudaGetTextureObjectTextureDesc(): ", unsafe_string(cudaGetErrorName(cudaGetTextureObjectTextureDesc(Ptr{cudaTextureDesc}(C_NULL), cudaTextureObject_t(0)))))
@test (cudaGetTextureObjectTextureDesc(Ptr{cudaTextureDesc}(C_NULL), cudaTextureObject_t(0)) == cudaErrorInvalidValue)


println("cudaGetTextureObjectResourceViewDesc(): ", unsafe_string(cudaGetErrorName(cudaGetTextureObjectResourceViewDesc(Ptr{cudaResourceViewDesc}(C_NULL), cudaTextureObject_t(0)))))
@test (cudaGetTextureObjectResourceViewDesc(Ptr{cudaResourceViewDesc}(C_NULL), cudaTextureObject_t(0)) == cudaErrorInvalidValue)


println("cudaCreateSurfaceObject(): ", unsafe_string(cudaGetErrorName(cudaCreateSurfaceObject(Ptr{cudaSurfaceObject_t}(C_NULL), Ptr{cudaResourceDesc}(C_NULL)))))
@test (cudaCreateSurfaceObject(Ptr{cudaSurfaceObject_t}(C_NULL), Ptr{cudaResourceDesc}(C_NULL)) == cudaErrorInvalidValue)


println("cudaDestroySurfaceObject(): ", unsafe_string(cudaGetErrorName(cudaDestroySurfaceObject(cudaSurfaceObject_t(0)))))
@test (cudaDestroySurfaceObject(cudaSurfaceObject_t(0)) == cudaSuccess)


println("cudaGetSurfaceObjectResourceDesc(): ", unsafe_string(cudaGetErrorName(cudaGetSurfaceObjectResourceDesc(Ptr{cudaResourceDesc}(C_NULL), cudaSurfaceObject_t(0)))))
@test (cudaGetSurfaceObjectResourceDesc(Ptr{cudaResourceDesc}(C_NULL), cudaSurfaceObject_t(0)) == cudaErrorInvalidDevice)


println("cudaDriverGetVersion(): ", unsafe_string(cudaGetErrorName(cudaDriverGetVersion(Ptr{Cint}(C_NULL)))))
@test (cudaDriverGetVersion(Ptr{Cint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaRuntimeGetVersion(): ", unsafe_string(cudaGetErrorName(cudaRuntimeGetVersion(Ptr{Cint}(C_NULL)))))
@test (cudaRuntimeGetVersion(Ptr{Cint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphCreate(): ", unsafe_string(cudaGetErrorName(cudaGraphCreate(Ptr{cudaGraph_t}(C_NULL), typemax(Cuint)))))
@test (cudaGraphCreate(Ptr{cudaGraph_t}(C_NULL), typemax(Cuint)) == cudaErrorInvalidValue)


println("cudaGraphAddKernelNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddKernelNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaKernelNodeParams}(C_NULL)))))
@test (cudaGraphAddKernelNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaKernelNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphKernelNodeGetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphKernelNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaKernelNodeParams}(C_NULL)))))
@test (cudaGraphKernelNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaKernelNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphKernelNodeSetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphKernelNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaKernelNodeParams}(C_NULL)))))
@test (cudaGraphKernelNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaKernelNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddMemcpyNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddMemcpyNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaMemcpy3DParms}(C_NULL)))))
@test (cudaGraphAddMemcpyNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaMemcpy3DParms}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphMemcpyNodeGetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphMemcpyNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemcpy3DParms}(C_NULL)))))
@test (cudaGraphMemcpyNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemcpy3DParms}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphMemcpyNodeSetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphMemcpyNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemcpy3DParms}(C_NULL)))))
@test (cudaGraphMemcpyNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemcpy3DParms}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddMemsetNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddMemsetNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaMemsetParams}(C_NULL)))))
@test (cudaGraphAddMemsetNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaMemsetParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphMemsetNodeGetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphMemsetNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemsetParams}(C_NULL)))))
@test (cudaGraphMemsetNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemsetParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphMemsetNodeSetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphMemsetNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemsetParams}(C_NULL)))))
@test (cudaGraphMemsetNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaMemsetParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddHostNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddHostNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaHostNodeParams}(C_NULL)))))
@test (cudaGraphAddHostNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), Ptr{cudaHostNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphHostNodeGetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphHostNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaHostNodeParams}(C_NULL)))))
@test (cudaGraphHostNodeGetParams(cudaGraphNode_t(C_NULL), Ptr{cudaHostNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphHostNodeSetParams(): ", unsafe_string(cudaGetErrorName(cudaGraphHostNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaHostNodeParams}(C_NULL)))))
@test (cudaGraphHostNodeSetParams(cudaGraphNode_t(C_NULL), Ptr{cudaHostNodeParams}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddChildGraphNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddChildGraphNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), cudaGraph_t(C_NULL)))))
@test (cudaGraphAddChildGraphNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t), cudaGraph_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphChildGraphNodeGetGraph(): ", unsafe_string(cudaGetErrorName(cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t(C_NULL), Ptr{cudaGraph_t}(C_NULL)))))
@test (cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t(C_NULL), Ptr{cudaGraph_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddEmptyNode(): ", unsafe_string(cudaGetErrorName(cudaGraphAddEmptyNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)))))
@test (cudaGraphAddEmptyNode(Ptr{cudaGraphNode_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaGraphClone(): ", unsafe_string(cudaGetErrorName(cudaGraphClone(Ptr{cudaGraph_t}(C_NULL), cudaGraph_t(C_NULL)))))
@test (cudaGraphClone(Ptr{cudaGraph_t}(C_NULL), cudaGraph_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphNodeFindInClone(): ", unsafe_string(cudaGetErrorName(cudaGraphNodeFindInClone(Ptr{cudaGraphNode_t}(C_NULL), cudaGraphNode_t(C_NULL), cudaGraph_t(C_NULL)))))
@test (cudaGraphNodeFindInClone(Ptr{cudaGraphNode_t}(C_NULL), cudaGraphNode_t(C_NULL), cudaGraph_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphNodeGetType(): ", unsafe_string(cudaGetErrorName(cudaGraphNodeGetType(cudaGraphNode_t(C_NULL), Ptr{Cuint}(C_NULL)))))
@test (cudaGraphNodeGetType(cudaGraphNode_t(C_NULL), Ptr{Cuint}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphGetNodes(): ", unsafe_string(cudaGetErrorName(cudaGraphGetNodes(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaGraphGetNodes(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphGetRootNodes(): ", unsafe_string(cudaGetErrorName(cudaGraphGetRootNodes(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaGraphGetRootNodes(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphGetEdges(): ", unsafe_string(cudaGetErrorName(cudaGraphGetEdges(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaGraphGetEdges(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphNodeGetDependencies(): ", unsafe_string(cudaGetErrorName(cudaGraphNodeGetDependencies(cudaGraphNode_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaGraphNodeGetDependencies(cudaGraphNode_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphNodeGetDependentNodes(): ", unsafe_string(cudaGetErrorName(cudaGraphNodeGetDependentNodes(cudaGraphNode_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)))))
@test (cudaGraphNodeGetDependentNodes(cudaGraphNode_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphAddDependencies(): ", unsafe_string(cudaGetErrorName(cudaGraphAddDependencies(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)))))
@test (cudaGraphAddDependencies(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaGraphRemoveDependencies(): ", unsafe_string(cudaGetErrorName(cudaGraphRemoveDependencies(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)))))
@test (cudaGraphRemoveDependencies(cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaGraphDestroyNode(): ", unsafe_string(cudaGetErrorName(cudaGraphDestroyNode(cudaGraphNode_t(C_NULL)))))
@test (cudaGraphDestroyNode(cudaGraphNode_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphInstantiate(): ", unsafe_string(cudaGetErrorName(cudaGraphInstantiate(Ptr{cudaGraphExec_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{UInt8}(C_NULL), typemax(Csize_t)))))
@test (cudaGraphInstantiate(Ptr{cudaGraphExec_t}(C_NULL), cudaGraph_t(C_NULL), Ptr{cudaGraphNode_t}(C_NULL), Ptr{UInt8}(C_NULL), typemax(Csize_t)) == cudaErrorInvalidValue)


println("cudaGraphLaunch(): ", unsafe_string(cudaGetErrorName(cudaGraphLaunch(cudaGraphExec_t(C_NULL), cudaStream_t(C_NULL)))))
@test (cudaGraphLaunch(cudaGraphExec_t(C_NULL), cudaStream_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphExecDestroy(): ", unsafe_string(cudaGetErrorName(cudaGraphExecDestroy(cudaGraphExec_t(C_NULL)))))
@test (cudaGraphExecDestroy(cudaGraphExec_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGraphDestroy(): ", unsafe_string(cudaGetErrorName(cudaGraphDestroy(cudaGraph_t(C_NULL)))))
@test (cudaGraphDestroy(cudaGraph_t(C_NULL)) == cudaErrorInvalidValue)


println("cudaGetExportTable(): ", unsafe_string(cudaGetErrorName(cudaGetExportTable(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{cudaUUID_t}(C_NULL)))))
@test (cudaGetExportTable(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{cudaUUID_t}(C_NULL)) == cudaErrorInvalidValue)



