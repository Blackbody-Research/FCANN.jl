#=*
* CUDA runtime API v8.0 functions
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

function cudaDeviceReset()::cudaError_t
    return ccall((:cudaDeviceReset, libcudart), cudaError_t, ())
end

function cudaDeviceSynchronize()::cudaError_t
    return ccall((:cudaDeviceSynchronize, libcudart), cudaError_t, ())
end

function cudaDeviceSetLimit(limit::Cuint, value::Csize_t)::cudaError_t
    return ccall((:cudaDeviceSetLimit, libcudart), cudaError_t, (Cuint, Csize_t,), limit, value)
end

function cudaDeviceGetLimit(pValue::Array{Csize_t, 1}, limit::Cuint)::cudaError_t
    return ccall((:cudaDeviceGetLimit, libcudart), cudaError_t, (Ref{Csize_t}, Cuint,), Base.cconvert(Ref{Csize_t}, pValue), limit)
end

function cudaDeviceGetLimit(pValue::Ptr{Csize_t}, limit::Cuint)::cudaError_t
    return ccall((:cudaDeviceGetLimit, libcudart), cudaError_t, (Ptr{Csize_t}, Cuint,), pValue, limit)
end

function cudaDeviceGetCacheConfig(pCacheConfig::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaDeviceGetCacheConfig, libcudart), cudaError_t, (Ref{Cuint},), Base.cconvert(Ref{Cuint}, pCacheConfig))
end

function cudaDeviceGetCacheConfig(pCacheConfig::Ptr{Cuint})::cudaError_t
    return ccall((:cudaDeviceGetCacheConfig, libcudart), cudaError_t, (Ptr{Cuint},), pCacheConfig)
end

function cudaDeviceGetStreamPriorityRange(leastPriority::Array{Cint, 1}, greatestPriority::Array{Cint, 1})::cudaError_t
    return ccall((:cudaDeviceGetStreamPriorityRange, libcudart), cudaError_t, (Ref{Cint}, Ref{Cint},), Base.cconvert(Ref{Cint}, leastPriority), Base.cconvert(Ref{Cint}, greatestPriority))
end

function cudaDeviceGetStreamPriorityRange(leastPriority::Ptr{Cint}, greatestPriority::Ptr{Cint})::cudaError_t
    return ccall((:cudaDeviceGetStreamPriorityRange, libcudart), cudaError_t, (Ptr{Cint}, Ptr{Cint},), leastPriority, greatestPriority)
end

function cudaDeviceSetCacheConfig(cacheConfig::Cuint)::cudaError_t
    return ccall((:cudaDeviceSetCacheConfig, libcudart), cudaError_t, (Cuint,), cacheConfig)
end

function cudaDeviceGetSharedMemConfig(pConfig::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaDeviceGetSharedMemConfig, libcudart), cudaError_t, (Ref{Cuint},), Base.cconvert(Ref{Cuint}, pConfig))
end

function cudaDeviceGetSharedMemConfig(pConfig::Ptr{Cuint})::cudaError_t
    return ccall((:cudaDeviceGetSharedMemConfig, libcudart), cudaError_t, (Ptr{Cuint},), pConfig)
end

function cudaDeviceSetSharedMemConfig(config::Cuint)::cudaError_t
    return ccall((:cudaDeviceSetSharedMemConfig, libcudart), cudaError_t, (Cuint,), config)
end

function cudaDeviceGetByPCIBusId(device::Array{Cint, 1}, pciBusId::Array{UInt8, 1})::cudaError_t
    return ccall((:cudaDeviceGetByPCIBusId, libcudart), cudaError_t, (Ref{Cint}, Ref{UInt8},), Base.cconvert(Ref{Cint}, device), Base.cconvert(Ref{UInt8}, pciBusId))
end

function cudaDeviceGetByPCIBusId(device::Ptr{Cint}, pciBusId::Ptr{UInt8})::cudaError_t
    return ccall((:cudaDeviceGetByPCIBusId, libcudart), cudaError_t, (Ptr{Cint}, Ptr{UInt8},), device, pciBusId)
end

function cudaDeviceGetPCIBusId(pciBusId::Array{UInt8, 1}, len::Cint, device::Cint)::cudaError_t
    return ccall((:cudaDeviceGetPCIBusId, libcudart), cudaError_t, (Ref{UInt8}, Cint, Cint,), Base.cconvert(Ref{UInt8}, pciBusId), len, device)
end

function cudaDeviceGetPCIBusId(pciBusId::Ptr{UInt8}, len::Cint, device::Cint)::cudaError_t
    return ccall((:cudaDeviceGetPCIBusId, libcudart), cudaError_t, (Ptr{UInt8}, Cint, Cint,), pciBusId, len, device)
end

function cudaIpcGetEventHandle(handle::Array{cudaIpcEventHandle_t, 1}, event::cudaEvent_t)::cudaError_t
    return ccall((:cudaIpcGetEventHandle, libcudart), cudaError_t, (Ref{cudaIpcEventHandle_t}, cudaEvent_t,), Base.cconvert(Ref{cudaIpcEventHandle_t}, handle), event)
end

function cudaIpcGetEventHandle(handle::Ptr{cudaIpcEventHandle_t}, event::cudaEvent_t)::cudaError_t
    return ccall((:cudaIpcGetEventHandle, libcudart), cudaError_t, (Ptr{cudaIpcEventHandle_t}, cudaEvent_t,), handle, event)
end

function cudaIpcOpenEventHandle(event::Array{cudaEvent_t, 1}, handle::cudaIpcEventHandle_t)::cudaError_t
    return ccall((:cudaIpcOpenEventHandle, libcudart), cudaError_t, (Ref{cudaEvent_t}, cudaIpcEventHandle_t,), Base.cconvert(Ref{cudaEvent_t}, event), handle)
end

function cudaIpcOpenEventHandle(event::Ptr{cudaEvent_t}, handle::cudaIpcEventHandle_t)::cudaError_t
    return ccall((:cudaIpcOpenEventHandle, libcudart), cudaError_t, (Ptr{cudaEvent_t}, cudaIpcEventHandle_t,), event, handle)
end

function cudaIpcGetMemHandle(handle::Array{cudaIpcMemHandle_t, 1}, devPtr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaIpcGetMemHandle, libcudart), cudaError_t, (Ref{cudaIpcMemHandle_t}, Ptr{Cvoid},), Base.cconvert(Ref{cudaIpcMemHandle_t}, handle), devPtr)
end

function cudaIpcGetMemHandle(handle::Ptr{cudaIpcMemHandle_t}, devPtr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaIpcGetMemHandle, libcudart), cudaError_t, (Ptr{cudaIpcMemHandle_t}, Ptr{Cvoid},), handle, devPtr)
end

function cudaIpcOpenMemHandle(devPtr::Array{Ptr{Cvoid}, 1}, handle::cudaIpcMemHandle_t, flags::Cuint)::cudaError_t
    return ccall((:cudaIpcOpenMemHandle, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, cudaIpcMemHandle_t, Cuint,), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), handle, flags)
end

function cudaIpcOpenMemHandle(devPtr::Ptr{Ptr{Cvoid}}, handle::cudaIpcMemHandle_t, flags::Cuint)::cudaError_t
    return ccall((:cudaIpcOpenMemHandle, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, cudaIpcMemHandle_t, Cuint,), devPtr, handle, flags)
end

function cudaIpcCloseMemHandle(devPtr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaIpcCloseMemHandle, libcudart), cudaError_t, (Ptr{Cvoid},), devPtr)
end

function cudaThreadExit()::cudaError_t
    return ccall((:cudaThreadExit, libcudart), cudaError_t, ())
end

function cudaThreadSynchronize()::cudaError_t
    return ccall((:cudaThreadSynchronize, libcudart), cudaError_t, ())
end

function cudaThreadSetLimit(limit::Cuint, value::Csize_t)::cudaError_t
    return ccall((:cudaThreadSetLimit, libcudart), cudaError_t, (Cuint, Csize_t,), limit, value)
end

function cudaThreadGetLimit(pValue::Array{Csize_t, 1}, limit::Cuint)::cudaError_t
    return ccall((:cudaThreadGetLimit, libcudart), cudaError_t, (Ref{Csize_t}, Cuint,), Base.cconvert(Ref{Csize_t}, pValue), limit)
end

function cudaThreadGetLimit(pValue::Ptr{Csize_t}, limit::Cuint)::cudaError_t
    return ccall((:cudaThreadGetLimit, libcudart), cudaError_t, (Ptr{Csize_t}, Cuint,), pValue, limit)
end

function cudaThreadGetCacheConfig(pCacheConfig::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaThreadGetCacheConfig, libcudart), cudaError_t, (Ref{Cuint},), Base.cconvert(Ref{Cuint}, pCacheConfig))
end

function cudaThreadGetCacheConfig(pCacheConfig::Ptr{Cuint})::cudaError_t
    return ccall((:cudaThreadGetCacheConfig, libcudart), cudaError_t, (Ptr{Cuint},), pCacheConfig)
end

function cudaThreadSetCacheConfig(cacheConfig::Cuint)::cudaError_t
    return ccall((:cudaThreadSetCacheConfig, libcudart), cudaError_t, (Cuint,), cacheConfig)
end

function cudaGetLastError()::cudaError_t
    return ccall((:cudaGetLastError, libcudart), cudaError_t, ())
end

function cudaPeekAtLastError()::cudaError_t
    return ccall((:cudaPeekAtLastError, libcudart), cudaError_t, ())
end

function cudaGetErrorName(error::cudaError_t)::Ptr{UInt8}
    return ccall((:cudaGetErrorName, libcudart), Ptr{UInt8}, (cudaError_t,), error)
end

function cudaGetErrorString(error::cudaError_t)::Ptr{UInt8}
    return ccall((:cudaGetErrorString, libcudart), Ptr{UInt8}, (cudaError_t,), error)
end

function cudaGetDeviceCount(count::Array{Cint, 1})::cudaError_t
    return ccall((:cudaGetDeviceCount, libcudart), cudaError_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, count))
end

function cudaGetDeviceCount(count::Ptr{Cint})::cudaError_t
    return ccall((:cudaGetDeviceCount, libcudart), cudaError_t, (Ptr{Cint},), count)
end

function cudaGetDeviceProperties(prop::Array{cudaDeviceProp, 1}, device::Cint)::cudaError_t
    return ccall((:cudaGetDeviceProperties, libcudart), cudaError_t, (Ref{cudaDeviceProp}, Cint,), Base.cconvert(Ref{cudaDeviceProp}, prop), device)
end

function cudaGetDeviceProperties(prop::Ptr{cudaDeviceProp}, device::Cint)::cudaError_t
    return ccall((:cudaGetDeviceProperties, libcudart), cudaError_t, (Ptr{cudaDeviceProp}, Cint,), prop, device)
end

function cudaDeviceGetAttribute(value::Array{Cint, 1}, attr::Cuint, device::Cint)::cudaError_t
    return ccall((:cudaDeviceGetAttribute, libcudart), cudaError_t, (Ref{Cint}, Cuint, Cint,), Base.cconvert(Ref{Cint}, value), attr, device)
end

function cudaDeviceGetAttribute(value::Ptr{Cint}, attr::Cuint, device::Cint)::cudaError_t
    return ccall((:cudaDeviceGetAttribute, libcudart), cudaError_t, (Ptr{Cint}, Cuint, Cint,), value, attr, device)
end

function cudaDeviceGetP2PAttribute(value::Array{Cint, 1}, attr::Cuint, srcDevice::Cint, dstDevice::Cint)::cudaError_t
    return ccall((:cudaDeviceGetP2PAttribute, libcudart), cudaError_t, (Ref{Cint}, Cuint, Cint, Cint,), Base.cconvert(Ref{Cint}, value), attr, srcDevice, dstDevice)
end

function cudaDeviceGetP2PAttribute(value::Ptr{Cint}, attr::Cuint, srcDevice::Cint, dstDevice::Cint)::cudaError_t
    return ccall((:cudaDeviceGetP2PAttribute, libcudart), cudaError_t, (Ptr{Cint}, Cuint, Cint, Cint,), value, attr, srcDevice, dstDevice)
end

function cudaChooseDevice(device::Array{Cint, 1}, prop::Array{cudaDeviceProp, 1})::cudaError_t
    return ccall((:cudaChooseDevice, libcudart), cudaError_t, (Ref{Cint}, Ref{cudaDeviceProp},), Base.cconvert(Ref{Cint}, device), Base.cconvert(Ref{cudaDeviceProp}, prop))
end

function cudaChooseDevice(device::Ptr{Cint}, prop::Ptr{cudaDeviceProp})::cudaError_t
    return ccall((:cudaChooseDevice, libcudart), cudaError_t, (Ptr{Cint}, Ptr{cudaDeviceProp},), device, prop)
end

function cudaSetDevice(device::Cint)::cudaError_t
    return ccall((:cudaSetDevice, libcudart), cudaError_t, (Cint,), device)
end

function cudaGetDevice(device::Array{Cint, 1})::cudaError_t
    return ccall((:cudaGetDevice, libcudart), cudaError_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, device))
end

function cudaGetDevice(device::Ptr{Cint})::cudaError_t
    return ccall((:cudaGetDevice, libcudart), cudaError_t, (Ptr{Cint},), device)
end

function cudaSetValidDevices(device_arr::Array{Cint, 1}, len::Cint)::cudaError_t
    return ccall((:cudaSetValidDevices, libcudart), cudaError_t, (Ref{Cint}, Cint,), Base.cconvert(Ref{Cint}, device_arr), len)
end

function cudaSetValidDevices(device_arr::Ptr{Cint}, len::Cint)::cudaError_t
    return ccall((:cudaSetValidDevices, libcudart), cudaError_t, (Ptr{Cint}, Cint,), device_arr, len)
end

function cudaSetDeviceFlags(flags::Cuint)::cudaError_t
    return ccall((:cudaSetDeviceFlags, libcudart), cudaError_t, (Cuint,), flags)
end

function cudaGetDeviceFlags(flags::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaGetDeviceFlags, libcudart), cudaError_t, (Ref{Cuint},), Base.cconvert(Ref{Cuint}, flags))
end

function cudaGetDeviceFlags(flags::Ptr{Cuint})::cudaError_t
    return ccall((:cudaGetDeviceFlags, libcudart), cudaError_t, (Ptr{Cuint},), flags)
end

function cudaStreamCreate(pStream::Array{cudaStream_t, 1})::cudaError_t
    return ccall((:cudaStreamCreate, libcudart), cudaError_t, (Ref{cudaStream_t},), Base.cconvert(Ref{cudaStream_t}, pStream))
end

function cudaStreamCreate(pStream::Ptr{cudaStream_t})::cudaError_t
    return ccall((:cudaStreamCreate, libcudart), cudaError_t, (Ptr{cudaStream_t},), pStream)
end

function cudaStreamCreateWithFlags(pStream::Array{cudaStream_t, 1}, flags::Cuint)::cudaError_t
    return ccall((:cudaStreamCreateWithFlags, libcudart), cudaError_t, (Ref{cudaStream_t}, Cuint,), Base.cconvert(Ref{cudaStream_t}, pStream), flags)
end

function cudaStreamCreateWithFlags(pStream::Ptr{cudaStream_t}, flags::Cuint)::cudaError_t
    return ccall((:cudaStreamCreateWithFlags, libcudart), cudaError_t, (Ptr{cudaStream_t}, Cuint,), pStream, flags)
end

function cudaStreamCreateWithPriority(pStream::Array{cudaStream_t, 1}, flags::Cuint, priority::Cint)::cudaError_t
    return ccall((:cudaStreamCreateWithPriority, libcudart), cudaError_t, (Ref{cudaStream_t}, Cuint, Cint,), Base.cconvert(Ref{cudaStream_t}, pStream), flags, priority)
end

function cudaStreamCreateWithPriority(pStream::Ptr{cudaStream_t}, flags::Cuint, priority::Cint)::cudaError_t
    return ccall((:cudaStreamCreateWithPriority, libcudart), cudaError_t, (Ptr{cudaStream_t}, Cuint, Cint,), pStream, flags, priority)
end

function cudaStreamGetPriority(hStream::cudaStream_t, priority::Array{Cint, 1})::cudaError_t
    return ccall((:cudaStreamGetPriority, libcudart), cudaError_t, (cudaStream_t, Ref{Cint},), hStream, Base.cconvert(Ref{Cint}, priority))
end

function cudaStreamGetPriority(hStream::cudaStream_t, priority::Ptr{Cint})::cudaError_t
    return ccall((:cudaStreamGetPriority, libcudart), cudaError_t, (cudaStream_t, Ptr{Cint},), hStream, priority)
end

function cudaStreamGetPriority_ptsz(hStream::cudaStream_t, priority::Array{Cint, 1})::cudaError_t
    return ccall((:cudaStreamGetPriority_ptsz, libcudart), cudaError_t, (cudaStream_t, Ref{Cint},), hStream, Base.cconvert(Ref{Cint}, priority))
end

function cudaStreamGetPriority_ptsz(hStream::cudaStream_t, priority::Ptr{Cint})::cudaError_t
    return ccall((:cudaStreamGetPriority_ptsz, libcudart), cudaError_t, (cudaStream_t, Ptr{Cint},), hStream, priority)
end

function cudaStreamGetFlags(hStream::cudaStream_t, flags::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaStreamGetFlags, libcudart), cudaError_t, (cudaStream_t, Ref{Cuint},), hStream, Base.cconvert(Ref{Cuint}, flags))
end

function cudaStreamGetFlags(hStream::cudaStream_t, flags::Ptr{Cuint})::cudaError_t
    return ccall((:cudaStreamGetFlags, libcudart), cudaError_t, (cudaStream_t, Ptr{Cuint},), hStream, flags)
end

function cudaStreamGetFlags_ptsz(hStream::cudaStream_t, flags::Array{Cuint, 1})::cudaError_t
    return ccall((:cudaStreamGetFlags_ptsz, libcudart), cudaError_t, (cudaStream_t, Ref{Cuint},), hStream, Base.cconvert(Ref{Cuint}, flags))
end

function cudaStreamGetFlags_ptsz(hStream::cudaStream_t, flags::Ptr{Cuint})::cudaError_t
    return ccall((:cudaStreamGetFlags_ptsz, libcudart), cudaError_t, (cudaStream_t, Ptr{Cuint},), hStream, flags)
end

function cudaStreamDestroy(stream::cudaStream_t)::cudaError_t
    return ccall((:cudaStreamDestroy, libcudart), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamWaitEvent(stream::cudaStream_t, event::cudaEvent_t, flags::Cuint)::cudaError_t
    return ccall((:cudaStreamWaitEvent, libcudart), cudaError_t, (cudaStream_t, cudaEvent_t, Cuint,), stream, event, flags)
end

function cudaStreamWaitEvent_ptsz(stream::cudaStream_t, event::cudaEvent_t, flags::Cuint)::cudaError_t
    return ccall((:cudaStreamWaitEvent_ptsz, libcudart), cudaError_t, (cudaStream_t, cudaEvent_t, Cuint,), stream, event, flags)
end

function cudaStreamAddCallback(stream::cudaStream_t, callback::cudaStreamCallback_t, userData::Ptr{Cvoid}, flags::Cuint)::cudaError_t
    return ccall((:cudaStreamAddCallback, libcudart), cudaError_t, (cudaStream_t, cudaStreamCallback_t, Ptr{Cvoid}, Cuint,), stream, callback, userData, flags)
end

function cudaStreamSynchronize(stream::cudaStream_t)::cudaError_t
    return ccall((:cudaStreamSynchronize, libcudart), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamQuery(stream::cudaStream_t)::cudaError_t
    return ccall((:cudaStreamQuery, libcudart), cudaError_t, (cudaStream_t,), stream)
end

function cudaStreamAttachMemAsync(stream::cudaStream_t, devPtr::Ptr{Cvoid}, length::Csize_t = Csize_t(0), flags::Cuint = cudaMemAttachSingle)::cudaError_t
    return ccall((:cudaStreamAttachMemAsync, libcudart), cudaError_t, (cudaStream_t, Ptr{Cvoid}, Csize_t, Cuint,), stream, devPtr, length, flags)
end

function cudaEventCreate(event::Array{cudaEvent_t, 1})::cudaError_t
    return ccall((:cudaEventCreate, libcudart), cudaError_t, (Ref{cudaEvent_t},), Base.cconvert(Ref{cudaEvent_t}, event))
end

function cudaEventCreate(event::Ptr{cudaEvent_t})::cudaError_t
    return ccall((:cudaEventCreate, libcudart), cudaError_t, (Ptr{cudaEvent_t},), event)
end

function cudaEventCreateWithFlags(event::Array{cudaEvent_t, 1}, flags::Cuint)::cudaError_t
    return ccall((:cudaEventCreateWithFlags, libcudart), cudaError_t, (Ref{cudaEvent_t}, Cuint,), Base.cconvert(Ref{cudaEvent_t}, event), flags)
end

function cudaEventCreateWithFlags(event::Ptr{cudaEvent_t}, flags::Cuint)::cudaError_t
    return ccall((:cudaEventCreateWithFlags, libcudart), cudaError_t, (Ptr{cudaEvent_t}, Cuint,), event, flags)
end

function cudaEventRecord(event::cudaEvent_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaEventRecord, libcudart), cudaError_t, (cudaEvent_t, cudaStream_t,), event, stream)
end

function cudaEventRecord_ptsz(event::cudaEvent_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaEventRecord_ptsz, libcudart), cudaError_t, (cudaEvent_t, cudaStream_t,), event, stream)
end

function cudaEventQuery(event::cudaEvent_t)::cudaError_t
    return ccall((:cudaEventQuery, libcudart), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventSynchronize(event::cudaEvent_t)::cudaError_t
    return ccall((:cudaEventSynchronize, libcudart), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventDestroy(event::cudaEvent_t)::cudaError_t
    return ccall((:cudaEventDestroy, libcudart), cudaError_t, (cudaEvent_t,), event)
end

function cudaEventElapsedTime(ms::Array{Cfloat, 1}, event_start::cudaEvent_t, event_end::cudaEvent_t)::cudaError_t
    return ccall((:cudaEventElapsedTime, libcudart), cudaError_t, (Ref{Cfloat}, cudaEvent_t, cudaEvent_t,), Base.cconvert(Ref{Cfloat}, ms), event_start, event_end)
end

function cudaEventElapsedTime(ms::Ptr{Cfloat}, event_start::cudaEvent_t, event_end::cudaEvent_t)::cudaError_t
    return ccall((:cudaEventElapsedTime, libcudart), cudaError_t, (Ptr{Cfloat}, cudaEvent_t, cudaEvent_t,), ms, event_start, event_end)
end

function cudaLaunchKernel(func::Ptr{Cvoid}, gridDim::dim3, blockDim::dim3, args::Array{Ptr{Cvoid}, 1}, sharedMem::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaLaunchKernel, libcudart), cudaError_t, (Ptr{Cvoid}, dim3, dim3, Ref{Ptr{Cvoid}}, Csize_t, cudaStream_t,), func, gridDim, blockDim, Base.cconvert(Ref{Ptr{Cvoid}}, args), sharedMem, stream)
end

function cudaLaunchKernel(func::Ptr{Cvoid}, gridDim::dim3, blockDim::dim3, args::Ptr{Ptr{Cvoid}}, sharedMem::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaLaunchKernel, libcudart), cudaError_t, (Ptr{Cvoid}, dim3, dim3, Ptr{Ptr{Cvoid}}, Csize_t, cudaStream_t,), func, gridDim, blockDim, args, sharedMem, stream)
end

function cudaLaunchKernel_ptsz(func::Ptr{Cvoid}, gridDim::dim3, blockDim::dim3, args::Array{Ptr{Cvoid}, 1}, sharedMem::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaLaunchKernel_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, dim3, dim3, Ref{Ptr{Cvoid}}, Csize_t, cudaStream_t,), func, gridDim, blockDim, Base.cconvert(Ref{Ptr{Cvoid}}, args), sharedMem, stream)
end

function cudaLaunchKernel_ptsz(func::Ptr{Cvoid}, gridDim::dim3, blockDim::dim3, args::Ptr{Ptr{Cvoid}}, sharedMem::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaLaunchKernel_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, dim3, dim3, Ptr{Ptr{Cvoid}}, Csize_t, cudaStream_t,), func, gridDim, blockDim, args, sharedMem, stream)
end

function cudaFuncSetCacheConfig(func::Ptr{Cvoid}, cacheConfig::Cuint)::cudaError_t
    return ccall((:cudaFuncSetCacheConfig, libcudart), cudaError_t, (Ptr{Cvoid}, Cuint,), func, cacheConfig)
end

function cudaFuncSetSharedMemConfig(func::Ptr{Cvoid}, config::Cuint)::cudaError_t
    return ccall((:cudaFuncSetSharedMemConfig, libcudart), cudaError_t, (Ptr{Cvoid}, Cuint,), func, config)
end

function cudaFuncGetAttributes(p::Array{cudaFuncAttributes, 1}, c::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaFuncGetAttributes, libcudart), cudaError_t, (Ref{cudaFuncAttributes}, Ptr{Cvoid},), Base.cconvert(Ref{cudaFuncAttributes}, p), c)
end

function cudaFuncGetAttributes(p::Ptr{cudaFuncAttributes}, c::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaFuncGetAttributes, libcudart), cudaError_t, (Ptr{cudaFuncAttributes}, Ptr{Cvoid},), p, c)
end

function cudaSetDoubleForDevice(d::Array{Cdouble, 1})::cudaError_t
    return ccall((:cudaSetDoubleForDevice, libcudart), cudaError_t, (Ref{Cdouble},), Base.cconvert(Ref{Cdouble}, d))
end

function cudaSetDoubleForDevice(d::Ptr{Cdouble})::cudaError_t
    return ccall((:cudaSetDoubleForDevice, libcudart), cudaError_t, (Ptr{Cdouble},), d)
end

function cudaSetDoubleForHost(d::Array{Cdouble, 1})::cudaError_t
    return ccall((:cudaSetDoubleForHost, libcudart), cudaError_t, (Ref{Cdouble},), Base.cconvert(Ref{Cdouble}, d))
end

function cudaSetDoubleForHost(d::Ptr{Cdouble})::cudaError_t
    return ccall((:cudaSetDoubleForHost, libcudart), cudaError_t, (Ptr{Cdouble},), d)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Array{Cint, 1}, func::Ptr{Cvoid}, blockSize::Cint, dynamicSmemSize::Csize_t)::cudaError_t
    return ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessor, libcudart), cudaError_t, (Ref{Cint}, Ptr{Cvoid}, Cint, Csize_t,), Base.cconvert(Ref{Cint}, numBlocks), func, blockSize, dynamicSmemSize)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Ptr{Cint}, func::Ptr{Cvoid}, blockSize::Cint, dynamicSmemSize::Csize_t)::cudaError_t
    return ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessor, libcudart), cudaError_t, (Ptr{Cint}, Ptr{Cvoid}, Cint, Csize_t,), numBlocks, func, blockSize, dynamicSmemSize)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Array{Cint, 1}, func::Ptr{Cvoid}, blockSize::Cint, dynamicSmemSize::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcudart), cudaError_t, (Ref{Cint}, Ptr{Cvoid}, Cint, Csize_t, Cuint,), Base.cconvert(Ref{Cint}, numBlocks), func, blockSize, dynamicSmemSize, flags)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Ptr{Cint}, func::Ptr{Cvoid}, blockSize::Cint, dynamicSmemSize::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcudart), cudaError_t, (Ptr{Cint}, Ptr{Cvoid}, Cint, Csize_t, Cuint,), numBlocks, func, blockSize, dynamicSmemSize, flags)
end

function cudaConfigureCall(gridDim::dim3, blockDim::dim3, sharedMem::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaConfigureCall, libcudart), cudaError_t, (dim3, dim3, Csize_t, cudaStream_t,), gridDim, blockDim, sharedMem, stream)
end

function cudaSetupArgument(arg::Ptr{Cvoid}, size::Csize_t, offset::Csize_t)::cudaError_t
    return ccall((:cudaSetupArgument, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Csize_t,), arg, size, offset)
end

function cudaLaunch(func::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaLaunch, libcudart), cudaError_t, (Ptr{Cvoid},), func)
end

function cudaLaunch_ptsz(func::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaLaunch_ptsz, libcudart), cudaError_t, (Ptr{Cvoid},), func)
end

function cudaMallocManaged(devPtr::Array{Ptr{Cvoid}, 1}, size::Csize_t, flags::Cuint = cudaMemAttachGlobal)::cudaError_t
    return ccall((:cudaMallocManaged, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Csize_t, Cuint,), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), size, flags)
end

function cudaMallocManaged(devPtr::Ptr{Ptr{Cvoid}}, size::Csize_t, flags::Cuint = cudaMemAttachGlobal)::cudaError_t
    return ccall((:cudaMallocManaged, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Csize_t, Cuint,), devPtr, size, flags)
end

function cudaMalloc(devPtr::Array{Ptr{Cvoid}, 1}, size::Csize_t)::cudaError_t
    return ccall((:cudaMalloc, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), size)
end

function cudaMalloc(devPtr::Ptr{Ptr{Cvoid}}, size::Csize_t)::cudaError_t
    return ccall((:cudaMalloc, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Csize_t,), devPtr, size)
end

function cudaMallocHost(ptr::Array{Ptr{Cvoid}, 1}, size::Csize_t)::cudaError_t
    return ccall((:cudaMallocHost, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, ptr), size)
end

function cudaMallocHost(ptr::Ptr{Ptr{Cvoid}}, size::Csize_t)::cudaError_t
    return ccall((:cudaMallocHost, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Csize_t,), ptr, size)
end

function cudaMallocPitch(devPtr::Array{Ptr{Cvoid}, 1}, pitch::Array{Csize_t, 1}, width::Csize_t, height::Csize_t)::cudaError_t
    return ccall((:cudaMallocPitch, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ref{Csize_t}, Csize_t, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), Base.cconvert(Ref{Csize_t}, pitch), width, height)
end

function cudaMallocPitch(devPtr::Ptr{Ptr{Cvoid}}, pitch::Ptr{Csize_t}, width::Csize_t, height::Csize_t)::cudaError_t
    return ccall((:cudaMallocPitch, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Csize_t, Csize_t,), devPtr, pitch, width, height)
end

function cudaMallocArray(array::Array{cudaArray_t, 1}, desc::Array{cudaChannelFormatDesc, 1}, width::Csize_t, height::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaMallocArray, libcudart), cudaError_t, (Ref{cudaArray_t}, Ref{cudaChannelFormatDesc}, Csize_t, Csize_t, Cuint,), Base.cconvert(Ref{cudaArray_t}, array), Base.cconvert(Ref{cudaChannelFormatDesc}, desc), width, height, flags)
end

function cudaMallocArray(array::Ptr{cudaArray_t}, desc::Ptr{cudaChannelFormatDesc}, width::Csize_t, height::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaMallocArray, libcudart), cudaError_t, (Ptr{cudaArray_t}, Ptr{cudaChannelFormatDesc}, Csize_t, Csize_t, Cuint,), array, desc, width, height, flags)
end

function cudaFree(devPtr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaFree, libcudart), cudaError_t, (Ptr{Cvoid},), devPtr)
end

function cudaFreeHost(ptr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaFreeHost, libcudart), cudaError_t, (Ptr{Cvoid},), ptr)
end

function cudaFreeArray(array::cudaArray_t)::cudaError_t
    return ccall((:cudaFreeArray, libcudart), cudaError_t, (cudaArray_t,), array)
end

function cudaFreeMipmappedArray(mipmappedArray::cudaMipmappedArray_t)::cudaError_t
    return ccall((:cudaFreeMipmappedArray, libcudart), cudaError_t, (cudaMipmappedArray_t,), mipmappedArray)
end

function cudaHostAlloc(pHost::Array{Ptr{Cvoid}, 1}, size::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaHostAlloc, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Csize_t, Cuint,), Base.cconvert(Ref{Ptr{Cvoid}}, pHost), size, flags)
end

function cudaHostAlloc(pHost::Ptr{Ptr{Cvoid}}, size::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaHostAlloc, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Csize_t, Cuint,), pHost, size, flags)
end

function cudaHostRegister(ptr::Ptr{Cvoid}, size::Csize_t, flags::Cuint)::cudaError_t
    return ccall((:cudaHostRegister, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cuint,), ptr, size, flags)
end

function cudaHostUnregister(ptr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaHostUnregister, libcudart), cudaError_t, (Ptr{Cvoid},), ptr)
end

function cudaHostGetDevicePointer(pDevice::Array{Ptr{Cvoid}, 1}, pHost::Ptr{Cvoid}, flags::Cuint)::cudaError_t
    return ccall((:cudaHostGetDevicePointer, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ptr{Cvoid}, Cuint,), Base.cconvert(Ref{Ptr{Cvoid}}, pDevice), pHost, flags)
end

function cudaHostGetDevicePointer(pDevice::Ptr{Ptr{Cvoid}}, pHost::Ptr{Cvoid}, flags::Cuint)::cudaError_t
    return ccall((:cudaHostGetDevicePointer, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Cuint,), pDevice, pHost, flags)
end

function cudaHostGetFlags(pFlags::Array{Cuint, 1}, pHost::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaHostGetFlags, libcudart), cudaError_t, (Ref{Cuint}, Ptr{Cvoid},), Base.cconvert(Ref{Cuint}, pFlags), pHost)
end

function cudaHostGetFlags(pFlags::Ptr{Cuint}, pHost::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaHostGetFlags, libcudart), cudaError_t, (Ptr{Cuint}, Ptr{Cvoid},), pFlags, pHost)
end

function cudaMalloc3D(pitchedDevPtr::Array{cudaPitchedPtr, 1}, extent::cudaExtent)::cudaError_t
    return ccall((:cudaMalloc3D, libcudart), cudaError_t, (Ref{cudaPitchedPtr}, cudaExtent,), Base.cconvert(Ref{cudaPitchedPtr}, pitchedDevPtr), extent)
end

function cudaMalloc3D(pitchedDevPtr::Ptr{cudaPitchedPtr}, extent::cudaExtent)::cudaError_t
    return ccall((:cudaMalloc3D, libcudart), cudaError_t, (Ptr{cudaPitchedPtr}, cudaExtent,), pitchedDevPtr, extent)
end

function cudaMalloc3DArray(array::Array{cudaArray_t, 1}, desc::Array{cudaChannelFormatDesc, 1}, extent::cudaExtent, flags::Cuint)::cudaError_t
    return ccall((:cudaMalloc3DArray, libcudart), cudaError_t, (Ref{cudaArray_t}, Ref{cudaChannelFormatDesc}, cudaExtent, Cuint,), Base.cconvert(Ref{cudaArray_t}, array), Base.cconvert(Ref{cudaChannelFormatDesc}, desc), extent, flags)
end

function cudaMalloc3DArray(array::Ptr{cudaArray_t}, desc::Ptr{cudaChannelFormatDesc}, extent::cudaExtent, flags::Cuint)::cudaError_t
    return ccall((:cudaMalloc3DArray, libcudart), cudaError_t, (Ptr{cudaArray_t}, Ptr{cudaChannelFormatDesc}, cudaExtent, Cuint,), array, desc, extent, flags)
end

function cudaMallocMipmappedArray(mipmappedArray::Array{cudaMipmappedArray_t, 1}, desc::Array{cudaChannelFormatDesc, 1}, extent::cudaExtent, numLevels::Cuint, flags::Cuint)::cudaError_t
    return ccall((:cudaMallocMipmappedArray, libcudart), cudaError_t, (Ref{cudaMipmappedArray_t}, Ref{cudaChannelFormatDesc}, cudaExtent, Cuint, Cuint,), Base.cconvert(Ref{cudaMipmappedArray_t}, mipmappedArray), Base.cconvert(Ref{cudaChannelFormatDesc}, desc), extent, numLevels, flags)
end

function cudaMallocMipmappedArray(mipmappedArray::Ptr{cudaMipmappedArray_t}, desc::Ptr{cudaChannelFormatDesc}, extent::cudaExtent, numLevels::Cuint, flags::Cuint)::cudaError_t
    return ccall((:cudaMallocMipmappedArray, libcudart), cudaError_t, (Ptr{cudaMipmappedArray_t}, Ptr{cudaChannelFormatDesc}, cudaExtent, Cuint, Cuint,), mipmappedArray, desc, extent, numLevels, flags)
end

function cudaGetMipmappedArrayLevel(levelArray::Array{cudaArray_t, 1}, mipmappedArray::cudaMipmappedArray_t, level::Cuint)::cudaError_t
    return ccall((:cudaGetMipmappedArrayLevel, libcudart), cudaError_t, (Ref{cudaArray_t}, cudaMipmappedArray_t, Cuint,), Base.cconvert(Ref{cudaArray_t}, levelArray), mipmappedArray, level)
end

function cudaGetMipmappedArrayLevel(levelArray::Ptr{cudaArray_t}, mipmappedArray::cudaMipmappedArray_t, level::Cuint)::cudaError_t
    return ccall((:cudaGetMipmappedArrayLevel, libcudart), cudaError_t, (Ptr{cudaArray_t}, cudaMipmappedArray_t, Cuint,), levelArray, mipmappedArray, level)
end

function cudaMemcpy3D(p::Array{cudaMemcpy3DParms, 1})::cudaError_t
    return ccall((:cudaMemcpy3D, libcudart), cudaError_t, (Ref{cudaMemcpy3DParms},), Base.cconvert(Ref{cudaMemcpy3DParms}, p))
end

function cudaMemcpy3D(p::Ptr{cudaMemcpy3DParms})::cudaError_t
    return ccall((:cudaMemcpy3D, libcudart), cudaError_t, (Ptr{cudaMemcpy3DParms},), p)
end

function cudaMemcpy3DPeer(p::Array{cudaMemcpy3DPeerParms, 1})::cudaError_t
    return ccall((:cudaMemcpy3DPeer, libcudart), cudaError_t, (Ref{cudaMemcpy3DPeerParms},), Base.cconvert(Ref{cudaMemcpy3DPeerParms}, p))
end

function cudaMemcpy3DPeer(p::Ptr{cudaMemcpy3DPeerParms})::cudaError_t
    return ccall((:cudaMemcpy3DPeer, libcudart), cudaError_t, (Ptr{cudaMemcpy3DPeerParms},), p)
end

function cudaMemcpy3DAsync(p::Array{cudaMemcpy3DParms, 1}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DAsync, libcudart), cudaError_t, (Ref{cudaMemcpy3DParms}, cudaStream_t,), Base.cconvert(Ref{cudaMemcpy3DParms}, p), stream)
end

function cudaMemcpy3DAsync(p::Ptr{cudaMemcpy3DParms}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DAsync, libcudart), cudaError_t, (Ptr{cudaMemcpy3DParms}, cudaStream_t,), p, stream)
end

function cudaMemcpy3DAsync_ptsz(p::Array{cudaMemcpy3DParms, 1}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DAsync_ptsz, libcudart), cudaError_t, (Ref{cudaMemcpy3DParms}, cudaStream_t,), Base.cconvert(Ref{cudaMemcpy3DParms}, p), stream)
end

function cudaMemcpy3DAsync_ptsz(p::Ptr{cudaMemcpy3DParms}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DAsync_ptsz, libcudart), cudaError_t, (Ptr{cudaMemcpy3DParms}, cudaStream_t,), p, stream)
end

function cudaMemcpy3DPeerAsync(p::Array{cudaMemcpy3DPeerParms, 1}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DPeerAsync, libcudart), cudaError_t, (Ref{cudaMemcpy3DPeerParms}, cudaStream_t,), Base.cconvert(Ref{cudaMemcpy3DPeerParms}, p), stream)
end

function cudaMemcpy3DPeerAsync(p::Ptr{cudaMemcpy3DPeerParms}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy3DPeerAsync, libcudart), cudaError_t, (Ptr{cudaMemcpy3DPeerParms}, cudaStream_t,), p, stream)
end

function cudaMemGetInfo(free::Array{Csize_t, 1}, total::Array{Csize_t, 1})::cudaError_t
    return ccall((:cudaMemGetInfo, libcudart), cudaError_t, (Ref{Csize_t}, Ref{Csize_t},), Base.cconvert(Ref{Csize_t}, free), Base.cconvert(Ref{Csize_t}, total))
end

function cudaMemGetInfo(free::Ptr{Csize_t}, total::Ptr{Csize_t})::cudaError_t
    return ccall((:cudaMemGetInfo, libcudart), cudaError_t, (Ptr{Csize_t}, Ptr{Csize_t},), free, total)
end

function cudaArrayGetInfo(desc::Array{cudaChannelFormatDesc, 1}, extent::Array{cudaExtent, 1}, flags::Array{Cuint, 1}, array::cudaArray_t)::cudaError_t
    return ccall((:cudaArrayGetInfo, libcudart), cudaError_t, (Ref{cudaChannelFormatDesc}, Ref{cudaExtent}, Ref{Cuint}, cudaArray_t,), Base.cconvert(Ref{cudaChannelFormatDesc}, desc), Base.cconvert(Ref{cudaExtent}, extent), Base.cconvert(Ref{Cuint}, flags), array)
end

function cudaArrayGetInfo(desc::Ptr{cudaChannelFormatDesc}, extent::Ptr{cudaExtent}, flags::Ptr{Cuint}, array::cudaArray_t)::cudaError_t
    return ccall((:cudaArrayGetInfo, libcudart), cudaError_t, (Ptr{cudaChannelFormatDesc}, Ptr{cudaExtent}, Ptr{Cuint}, cudaArray_t,), desc, extent, flags, array)
end

function cudaMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpy, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cuint,), dst, src, count, kind)
end

function cudaMemcpyPeer(dst::Ptr{Cvoid}, dstDevice::Cint, src::Ptr{Cvoid}, srcDevice::Cint, count::Csize_t)::cudaError_t
    return ccall((:cudaMemcpyPeer, libcudart), cudaError_t, (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, Csize_t,), dst, dstDevice, src, srcDevice, count)
end

function cudaMemcpyToArray(dst::cudaArray_t, wOffset::Csize_t, hOffset::Csize_t, src::Ptr{Cvoid}, count::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpyToArray, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Cuint,), dst, wOffset, hOffset, src, count, kind)
end

function cudaMemcpyFromArray(dst::Ptr{Cvoid}, src::cudaArray_const_t, wOffset::Csize_t, hOffset::Csize_t, count::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpyFromArray, libcudart), cudaError_t, (Ptr{Cvoid}, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Cuint,), dst, src, wOffset, hOffset, count, kind)
end

function cudaMemcpyArrayToArray(dst::cudaArray_t, wOffsetDst::Csize_t, hOffsetDst::Csize_t, src::cudaArray_const_t, wOffsetSrc::Csize_t, hOffsetSrc::Csize_t, count::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpyArrayToArray, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Cuint,), dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)
end

function cudaMemcpy2D(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpy2D, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Cuint,), dst, dpitch, src, spitch, width, height, kind)
end

function cudaMemcpy2DToArray(dst::cudaArray_t, wOffset::Csize_t, hOffset::Csize_t, src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpy2DToArray, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Cuint,), dst, wOffset, hOffset, src, spitch, width, height, kind)
end

function cudaMemcpy2DFromArray(dst::Ptr{Cvoid}, dpitch::Csize_t, src::cudaArray_const_t, wOffset::Csize_t, hOffset::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpy2DFromArray, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,), dst, dpitch, src, wOffset, hOffset, width, height, kind)
end

function cudaMemcpy2DArrayToArray(dst::cudaArray_t, wOffsetDst::Csize_t, hOffsetDst::Csize_t, src::cudaArray_const_t, wOffsetSrc::Csize_t, hOffsetSrc::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpy2DArrayToArray, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Csize_t, Cuint,), dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)
end

function cudaMemcpyToSymbol(symbol::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Csize_t, offset::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpyToSymbol, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t, Cuint,), symbol, src, count, offset, kind)
end

function cudaMemcpyFromSymbol(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid}, count::Csize_t, offset::Csize_t, kind::Cuint)::cudaError_t
    return ccall((:cudaMemcpyFromSymbol, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t, Cuint,), dst, symbol, count, offset, kind)
end

function cudaMemcpyAsync(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cuint, cudaStream_t,), dst, src, count, kind, stream)
end

function cudaMemcpyAsync_ptsz(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyAsync_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cuint, cudaStream_t,), dst, src, count, kind, stream)
end

function cudaMemcpyPeerAsync(dst::Ptr{Cvoid}, dstDevice::Cint, src::Ptr{Cvoid}, srcDevice::Cint, count::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyPeerAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, Csize_t, cudaStream_t,), dst, dstDevice, src, srcDevice, count, stream)
end

function cudaMemcpyToArrayAsync(dst::cudaArray_t, wOffset::Csize_t, hOffset::Csize_t, src::Ptr{Cvoid}, count::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyToArrayAsync, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Cuint, cudaStream_t,), dst, wOffset, hOffset, src, count, kind, stream)
end

function cudaMemcpyFromArrayAsync(dst::Ptr{Cvoid}, src::cudaArray_const_t, wOffset::Csize_t, hOffset::Csize_t, count::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyFromArrayAsync, libcudart), cudaError_t, (Ptr{Cvoid}, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, src, wOffset, hOffset, count, kind, stream)
end

function cudaMemcpy2DAsync(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy2DAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, dpitch, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DAsync_ptsz(dst::Ptr{Cvoid}, dpitch::Csize_t, src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy2DAsync_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, dpitch, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DToArrayAsync(dst::cudaArray_t, wOffset::Csize_t, hOffset::Csize_t, src::Ptr{Cvoid}, spitch::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy2DToArrayAsync, libcudart), cudaError_t, (cudaArray_t, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, wOffset, hOffset, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DFromArrayAsync(dst::Ptr{Cvoid}, dpitch::Csize_t, src::cudaArray_const_t, wOffset::Csize_t, hOffset::Csize_t, width::Csize_t, height::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpy2DFromArrayAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, cudaArray_const_t, Csize_t, Csize_t, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)
end

function cudaMemcpyToSymbolAsync(symbol::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Csize_t, offset::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyToSymbolAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t, Cuint, cudaStream_t,), symbol, src, count, offset, kind, stream)
end

function cudaMemcpyFromSymbolAsync(dst::Ptr{Cvoid}, symbol::Ptr{Cvoid}, count::Csize_t, offset::Csize_t, kind::Cuint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemcpyFromSymbolAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t, Cuint, cudaStream_t,), dst, symbol, count, offset, kind, stream)
end

function cudaMemset(devPtr::Ptr{Cvoid}, value::Cint, count::Csize_t)::cudaError_t
    return ccall((:cudaMemset, libcudart), cudaError_t, (Ptr{Cvoid}, Cint, Csize_t,), devPtr, value, count)
end

function cudaMemset2D(devPtr::Ptr{Cvoid}, pitch::Csize_t, value::Cint, width::Csize_t, height::Csize_t)::cudaError_t
    return ccall((:cudaMemset2D, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t,), devPtr, pitch, value, width, height)
end

function cudaMemset3D(pitchedDevPtr::cudaPitchedPtr, value::Cint, extent::cudaExtent)::cudaError_t
    return ccall((:cudaMemset3D, libcudart), cudaError_t, (cudaPitchedPtr, Cint, cudaExtent,), pitchedDevPtr, value, extent)
end

function cudaMemsetAsync(devPtr::Ptr{Cvoid}, value::Cint, count::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemsetAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Cint, Csize_t, cudaStream_t,), devPtr, value, count, stream)
end

function cudaMemsetAsync_ptsz(devPtr::Ptr{Cvoid}, value::Cint, count::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemsetAsync_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, Cint, Csize_t, cudaStream_t,), devPtr, value, count, stream)
end

function cudaMemset2DAsync(devPtr::Ptr{Cvoid}, pitch::Csize_t, value::Cint, width::Csize_t, height::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemset2DAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, cudaStream_t), devPtr, pitch, value, width, height, stream)
end

function cudaMemset2DAsync_ptsz(devPtr::Ptr{Cvoid}, pitch::Csize_t, value::Cint, width::Csize_t, height::Csize_t, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemset2DAsync_ptsz, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, cudaStream_t), devPtr, pitch, value, width, height, stream)
end

function cudaMemset3DAsync(pitchedDevPtr::cudaPitchedPtr, value::Cint, extent::cudaExtent, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemset3DAsync, libcudart), cudaError_t, (cudaPitchedPtr, Cint, cudaExtent, cudaStream_t,), pitchedDevPtr, value, extent, stream)
end

function cudaMemset3DAsync_ptsz(pitchedDevPtr::cudaPitchedPtr, value::Cint, extent::cudaExtent, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemset3DAsync_ptsz, libcudart), cudaError_t, (cudaPitchedPtr, Cint, cudaExtent, cudaStream_t,), pitchedDevPtr, value, extent, stream)
end

function cudaGetSymbolAddress(devPtr::Array{Ptr{Cvoid}, 1}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSymbolAddress, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ptr{Cvoid},), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), symbol)
end

function cudaGetSymbolAddress(devPtr::Ptr{Ptr{Cvoid}}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSymbolAddress, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid},), devPtr, symbol)
end

function cudaGetSymbolSize(size::Array{Csize_t, 1}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSymbolSize, libcudart), cudaError_t, (Ref{Csize_t}, Ptr{Cvoid},), Base.cconvert(Ref{Csize_t}, size), symbol)
end

function cudaGetSymbolSize(size::Ptr{Csize_t}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSymbolSize, libcudart), cudaError_t, (Ptr{Csize_t}, Ptr{Cvoid},), size, symbol)
end

function cudaMemPrefetchAsync(devPtr::Ptr{Cvoid}, count::Csize_t, dstDevice::Cint, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaMemPrefetchAsync, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cint, cudaStream_t,), devPtr, count, dstDevice, stream)
end

function cudaMemAdvise(devPtr::Ptr{Cvoid}, count::Csize_t, advice::Cuint, device::Cint)::cudaError_t
    return ccall((:cudaMemAdvise, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cuint, Cint,), devPtr, count, advice, device)
end

function cudaMemRangeGetAttribute(data::Ptr{Cvoid}, dataSize::Csize_t, attribute::Cuint, devPtr::Ptr{Cvoid}, count::Csize_t)::cudaError_t
    return ccall((:cudaMemRangeGetAttribute, libcudart), cudaError_t, (Ptr{Cvoid}, Csize_t, Cuint, Ptr{Cvoid}, Csize_t,), data, dataSize, attribute, devPtr, count)
end

function cudaMemRangeGetAttributes(data::Array{Ptr{Cvoid}, 1}, dataSizes::Array{Csize_t, 1}, attributes::Array{Cuint, 1}, numAttributes::Csize_t, devPtr::Ptr{Cvoid}, count::Csize_t)::cudaError_t
    return ccall((:cudaMemRangeGetAttributes, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ref{Csize_t}, Ref{Cuint}, Csize_t, Ptr{Cvoid}, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, data), Base.cconvert(Ref{Csize_t}, dataSizes), Base.cconvert(Ref{Cuint}, attributes), numAttributes, devPtr, count)
end

function cudaMemRangeGetAttributes(data::Ptr{Ptr{Cvoid}}, dataSizes::Ptr{Csize_t}, attributes::Ptr{Cuint}, numAttributes::Csize_t, devPtr::Ptr{Cvoid}, count::Csize_t)::cudaError_t
    return ccall((:cudaMemRangeGetAttributes, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cuint}, Csize_t, Ptr{Cvoid}, Csize_t,), data, dataSizes, attributes, numAttributes, devPtr, count)
end

function cudaPointerGetAttributes(attributes::Array{cudaPointerAttributes, 1}, ptr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaPointerGetAttributes, libcudart), cudaError_t, (Ref{cudaPointerAttributes}, Ptr{Cvoid},), Base.cconvert(Ref{cudaPointerAttributes}, attributes), ptr)
end

function cudaPointerGetAttributes(attributes::Ptr{cudaPointerAttributes}, ptr::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaPointerGetAttributes, libcudart), cudaError_t, (Ptr{cudaPointerAttributes}, Ptr{Cvoid},), attributes, ptr)
end

function cudaDeviceCanAccessPeer(canAccessPeer::Array{Cint, 1}, device::Cint, peerDevice::Cint)::cudaError_t
    return ccall((:cudaDeviceCanAccessPeer, libcudart), cudaError_t, (Ref{Cint}, Cint, Cint,), Base.cconvert(Ref{Cint}, canAccessPeer), device, peerDevice)
end

function cudaDeviceCanAccessPeer(canAccessPeer::Ptr{Cint}, device::Cint, peerDevice::Cint)::cudaError_t
    return ccall((:cudaDeviceCanAccessPeer, libcudart), cudaError_t, (Ptr{Cint}, Cint, Cint,), canAccessPeer, device, peerDevice)
end

function cudaDeviceEnablePeerAccess(peerDevice::Cint, flags::Cuint)::cudaError_t
    return ccall((:cudaDeviceEnablePeerAccess, libcudart), cudaError_t, (Cint, Cuint,), peerDevice, flags)
end

function cudaDeviceDisablePeerAccess(peerDevice::Cint)::cudaError_t
    return ccall((:cudaDeviceDisablePeerAccess, libcudart), cudaError_t, (Cint,), peerDevice)
end

function cudaGraphicsUnregisterResource(resource::cudaGraphicsResource_t)::cudaError_t
    return ccall((:cudaGraphicsUnregisterResource, libcudart), cudaError_t, (cudaGraphicsResource_t,), resource)
end

function cudaGraphicsResourceSetMapFlags(resource::cudaGraphicsResource_t, flags::Cuint)::cudaError_t
    return ccall((:cudaGraphicsResourceSetMapFlags, libcudart), cudaError_t, (cudaGraphicsResource_t, Cuint,), resource, flags)
end

function cudaGraphicsMapResources(count::Cint, resources::Array{cudaGraphicsResource_t, 1}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaGraphicsMapResources, libcudart), cudaError_t, (Cint, Ref{cudaGraphicsResource_t}, cudaStream_t,), count, Base.cconvert(Ref{cudaGraphicsResource_t}, resources), stream)
end

function cudaGraphicsMapResources(count::Cint, resources::Ptr{cudaGraphicsResource_t}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaGraphicsMapResources, libcudart), cudaError_t, (Cint, Ptr{cudaGraphicsResource_t}, cudaStream_t,), count, resources, stream)
end

function cudaGraphicsUnmapResources(count::Cint, resources::Array{cudaGraphicsResource_t, 1}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaGraphicsUnmapResources, libcudart), cudaError_t, (Cint, Ref{cudaGraphicsResource_t}, cudaStream_t,), count, Base.cconvert(Ref{cudaGraphicsResource_t}, resources), stream)
end

function cudaGraphicsUnmapResources(count::Cint, resources::Ptr{cudaGraphicsResource_t}, stream::cudaStream_t)::cudaError_t
    return ccall((:cudaGraphicsUnmapResources, libcudart), cudaError_t, (Cint, Ptr{cudaGraphicsResource_t}, cudaStream_t,), count, resources, stream)
end

function cudaGraphicsResourceGetMappedPointer(devPtr::Array{Ptr{Cvoid}, 1}, size::Array{Csize_t, 1}, resource::cudaGraphicsResource_t)::cudaError_t
    return ccall((:cudaGraphicsResourceGetMappedPointer, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ref{Csize_t}, cudaGraphicsResource_t,), Base.cconvert(Ref{Ptr{Cvoid}}, devPtr), Base.cconvert(Ref{Csize_t}, size), resource)
end

function cudaGraphicsResourceGetMappedPointer(devPtr::Ptr{Ptr{Cvoid}}, size::Ptr{Csize_t}, resource::cudaGraphicsResource_t)::cudaError_t
    return ccall((:cudaGraphicsResourceGetMappedPointer, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, cudaGraphicsResource_t,), devPtr, size, resource)
end

function cudaGraphicsSubResourceGetMappedArray(array::Array{cudaArray_t, 1}, resource::cudaGraphicsResource_t, arrayIndex::Cuint, mipLevel::Cuint)::cudaError_t
    return ccall((:cudaGraphicsSubResourceGetMappedArray, libcudart), cudaError_t, (Ref{cudaArray_t}, cudaGraphicsResource_t, Cuint, Cuint,), Base.cconvert(Ref{cudaArray_t}, array), resource, arrayIndex, mipLevel)
end

function cudaGraphicsSubResourceGetMappedArray(array::Ptr{cudaArray_t}, resource::cudaGraphicsResource_t, arrayIndex::Cuint, mipLevel::Cuint)::cudaError_t
    return ccall((:cudaGraphicsSubResourceGetMappedArray, libcudart), cudaError_t, (Ptr{cudaArray_t}, cudaGraphicsResource_t, Cuint, Cuint,), array, resource, arrayIndex, mipLevel)
end

function cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray::Array{cudaMipmappedArray_t, 1}, resource::cudaGraphicsResource_t)::cudaError_t
    return ccall((:cudaGraphicsResourceGetMappedMipmappedArray, libcudart), cudaError_t, (Ref{cudaMipmappedArray_t}, cudaGraphicsResource_t,), Base.cconvert(Ref{cudaMipmappedArray_t}, mipmappedArray), resource)
end

function cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray::Ptr{cudaMipmappedArray_t}, resource::cudaGraphicsResource_t)::cudaError_t
    return ccall((:cudaGraphicsResourceGetMappedMipmappedArray, libcudart), cudaError_t, (Ptr{cudaMipmappedArray_t}, cudaGraphicsResource_t,), mipmappedArray, resource)
end

function cudaGetChannelDesc(desc::Array{cudaChannelFormatDesc, 1}, array::cudaArray_const_t)::cudaError_t
    return ccall((:cudaGetChannelDesc, libcudart), cudaError_t, (Ref{cudaChannelFormatDesc}, cudaArray_const_t,), Base.cconvert(Ref{cudaChannelFormatDesc}, desc), array)
end

function cudaGetChannelDesc(desc::Ptr{cudaChannelFormatDesc}, array::cudaArray_const_t)::cudaError_t
    return ccall((:cudaGetChannelDesc, libcudart), cudaError_t, (Ptr{cudaChannelFormatDesc}, cudaArray_const_t,), desc, array)
end

function cudaCreateChannelDesc(x::Cint, y::Cint, z::Cint, w::Cint, f::Cuint)::cudaChannelFormatDesc
    return ccall((:cudaCreateChannelDesc, libcudart), cudaChannelFormatDesc, (Cint, Cint, Cint, Cint, Cuint,), x, y, z, w, f)
end

function cudaBindTexture(offset::Array{Csize_t, 1}, texref::Array{textureReference, 1}, devPtr::Ptr{Cvoid}, desc::Array{cudaChannelFormatDesc, 1}, size::Csize_t)::cudaError_t
    return ccall((:cudaBindTexture, libcudart), cudaError_t, (Ref{Csize_t}, Ref{textureReference}, Ptr{Cvoid}, Ref{cudaChannelFormatDesc}, Csize_t,), Base.cconvert(Ref{Csize_t}, offset), Base.cconvert(Ref{textureReference}, texref), devPtr, Base.cconvert(Ref{cudaChannelFormatDesc}, desc), size)
end

function cudaBindTexture(offset::Ptr{Csize_t}, texref::Ptr{textureReference}, devPtr::Ptr{Cvoid}, desc::Ptr{cudaChannelFormatDesc}, size::Csize_t)::cudaError_t
    return ccall((:cudaBindTexture, libcudart), cudaError_t, (Ptr{Csize_t}, Ptr{textureReference}, Ptr{Cvoid}, Ptr{cudaChannelFormatDesc}, Csize_t,), offset, texref, devPtr, desc, size)
end

function cudaBindTexture2D(offset::Array{Csize_t, 1}, texref::Array{textureReference, 1}, devPtr::Ptr{Cvoid}, desc::Array{cudaChannelFormatDesc, 1}, width::Csize_t, height::Csize_t, pitch::Csize_t)::cudaError_t
    return ccall((:cudaBindTexture2D, libcudart), cudaError_t, (Ref{Csize_t}, Ref{textureReference}, Ptr{Cvoid}, Ref{cudaChannelFormatDesc}, Csize_t, Csize_t, Csize_t,), Base.cconvert(Ref{Csize_t}, offset), Base.cconvert(Ref{textureReference}, texref), devPtr, Base.cconvert(Ref{cudaChannelFormatDesc}, desc), width, height, pitch)
end

function cudaBindTexture2D(offset::Ptr{Csize_t}, texref::Ptr{textureReference}, devPtr::Ptr{Cvoid}, desc::Ptr{cudaChannelFormatDesc}, width::Csize_t, height::Csize_t, pitch::Csize_t)::cudaError_t
    return ccall((:cudaBindTexture2D, libcudart), cudaError_t, (Ptr{Csize_t}, Ptr{textureReference}, Ptr{Cvoid}, Ptr{cudaChannelFormatDesc}, Csize_t, Csize_t, Csize_t,), offset, texref, devPtr, desc, width, height, pitch)
end

function cudaBindTextureToArray(texref::Array{textureReference, 1}, array::cudaArray_const_t, desc::Array{cudaChannelFormatDesc, 1})::cudaError_t
    return ccall((:cudaBindTextureToArray, libcudart), cudaError_t, (Ref{textureReference}, cudaArray_const_t, Ref{cudaChannelFormatDesc},), Base.cconvert(Ref{textureReference}, texref), array, Base.cconvert(Ref{cudaChannelFormatDesc}, desc))
end

function cudaBindTextureToArray(texref::Ptr{textureReference}, array::cudaArray_const_t, desc::Ptr{cudaChannelFormatDesc})::cudaError_t
    return ccall((:cudaBindTextureToArray, libcudart), cudaError_t, (Ptr{textureReference}, cudaArray_const_t, Ptr{cudaChannelFormatDesc},), texref, array, desc)
end

function cudaBindTextureToMipmappedArray(texref::Array{textureReference, 1}, mipmappedArray::cudaMipmappedArray_t, desc::Array{cudaChannelFormatDesc, 1})::cudaError_t
    return ccall((:cudaBindTextureToMipmappedArray, libcudart), cudaError_t, (Ref{textureReference}, cudaMipmappedArray_t, Ref{cudaChannelFormatDesc},), Base.cconvert(Ref{textureReference}, texref), mipmappedArray, Base.cconvert(Ref{cudaChannelFormatDesc}, desc))
end

function cudaBindTextureToMipmappedArray(texref::Ptr{textureReference}, mipmappedArray::cudaMipmappedArray_t, desc::Ptr{cudaChannelFormatDesc})::cudaError_t
    return ccall((:cudaBindTextureToMipmappedArray, libcudart), cudaError_t, (Ptr{textureReference}, cudaMipmappedArray_t, Ptr{cudaChannelFormatDesc},), texref, mipmappedArray, desc)
end

function cudaUnbindTexture(texref::Array{textureReference, 1})::cudaError_t
    return ccall((:cudaUnbindTexture, libcudart), cudaError_t, (Ref{textureReference},), Base.cconvert(Ref{textureReference}, texref))
end

function cudaUnbindTexture(texref::Ptr{textureReference})::cudaError_t
    return ccall((:cudaUnbindTexture, libcudart), cudaError_t, (Ptr{textureReference},), texref)
end

function cudaGetTextureAlignmentOffset(offset::Array{Csize_t, 1}, texref::Array{textureReference, 1})::cudaError_t
    return ccall((:cudaGetTextureAlignmentOffset, libcudart), cudaError_t, (Ref{Csize_t}, Ref{textureReference},), Base.cconvert(Ref{Csize_t}, offset), Base.cconvert(Ref{textureReference}, texref))
end

function cudaGetTextureAlignmentOffset(offset::Ptr{Csize_t}, texref::Ptr{textureReference})::cudaError_t
    return ccall((:cudaGetTextureAlignmentOffset, libcudart), cudaError_t, (Ptr{Csize_t}, Ptr{textureReference},), offset, texref)
end

function cudaGetTextureReference(texref::Array{Ptr{textureReference}, 1}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetTextureReference, libcudart), cudaError_t, (Ref{Ptr{textureReference}}, Ptr{Cvoid},), Base.cconvert(Ref{Ptr{textureReference}}, texref), symbol)
end

function cudaGetTextureReference(texref::Ptr{Ptr{textureReference}}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetTextureReference, libcudart), cudaError_t, (Ptr{Ptr{textureReference}}, Ptr{Cvoid},), texref, symbol)
end

function cudaBindSurfaceToArray(surfref::Array{surfaceReference, 1}, array::cudaArray_const_t, desc::Array{cudaChannelFormatDesc, 1})::cudaError_t
    return ccall((:cudaBindSurfaceToArray, libcudart), cudaError_t, (Ref{surfaceReference}, cudaArray_const_t, Ref{cudaChannelFormatDesc},), Base.cconvert(Ref{surfaceReference}, surfref), array, Base.cconvert(Ref{cudaChannelFormatDesc}, desc))
end

function cudaBindSurfaceToArray(surfref::Ptr{surfaceReference}, array::cudaArray_const_t, desc::Ptr{cudaChannelFormatDesc})::cudaError_t
    return ccall((:cudaBindSurfaceToArray, libcudart), cudaError_t, (Ptr{surfaceReference}, cudaArray_const_t, Ptr{cudaChannelFormatDesc},), surfref, array, desc)
end

function cudaGetSurfaceReference(surfref::Array{Ptr{surfaceReference}, 1}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSurfaceReference, libcudart), cudaError_t, (Ref{Ptr{surfaceReference}}, Ptr{Cvoid},), Base.cconvert(Ref{Ptr{surfaceReference}}, surfref), symbol)
end

function cudaGetSurfaceReference(surfref::Ptr{Ptr{surfaceReference}}, symbol::Ptr{Cvoid})::cudaError_t
    return ccall((:cudaGetSurfaceReference, libcudart), cudaError_t, (Ptr{Ptr{surfaceReference}}, Ptr{Cvoid},), surfref, symbol)
end

function cudaCreateTextureObject(pTexObject::Array{cudaTextureObject_t, 1}, pResDesc::Array{cudaResourceDesc, 1}, pTexDesc::Array{cudaTextureDesc, 1}, pResViewDesc::Array{cudaResourceViewDesc, 1})::cudaError_t
    return ccall((:cudaCreateTextureObject, libcudart), cudaError_t, (Ref{cudaTextureObject_t}, Ref{cudaResourceDesc}, Ref{cudaTextureDesc}, Ref{cudaResourceViewDesc},), Base.cconvert(Ref{cudaTextureObject_t}, pTexObject), Base.cconvert(Ref{cudaResourceDesc}, pResDesc), Base.cconvert(Ref{cudaTextureDesc}, pTexDesc), Base.cconvert(Ref{cudaResourceViewDesc}, pResViewDesc))
end

function cudaCreateTextureObject(pTexObject::Ptr{cudaTextureObject_t}, pResDesc::Ptr{cudaResourceDesc}, pTexDesc::Ptr{cudaTextureDesc}, pResViewDesc::Ptr{cudaResourceViewDesc})::cudaError_t
    return ccall((:cudaCreateTextureObject, libcudart), cudaError_t, (Ptr{cudaTextureObject_t}, Ptr{cudaResourceDesc}, Ptr{cudaTextureDesc}, Ptr{cudaResourceViewDesc},), pTexObject, pResDesc, pTexDesc, pResViewDesc)
end

function cudaDestroyTextureObject(texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaDestroyTextureObject, libcudart), cudaError_t, (cudaTextureObject_t,), texObject)
end

function cudaGetTextureObjectResourceDesc(pResDesc::Array{cudaResourceDesc, 1}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectResourceDesc, libcudart), cudaError_t, (Ref{cudaResourceDesc}, cudaTextureObject_t,), Base.cconvert(Ref{cudaResourceDesc}, pResDesc), texObject)
end

function cudaGetTextureObjectResourceDesc(pResDesc::Ptr{cudaResourceDesc}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectResourceDesc, libcudart), cudaError_t, (Ptr{cudaResourceDesc}, cudaTextureObject_t,), pResDesc, texObject)
end

function cudaGetTextureObjectTextureDesc(pTexDesc::Array{cudaTextureDesc, 1}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectTextureDesc, libcudart), cudaError_t, (Ref{cudaTextureDesc}, cudaTextureObject_t,), Base.cconvert(Ref{cudaTextureDesc}, pTexDesc), texObject)
end

function cudaGetTextureObjectTextureDesc(pTexDesc::Ptr{cudaTextureDesc}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectTextureDesc, libcudart), cudaError_t, (Ptr{cudaTextureDesc}, cudaTextureObject_t,), pTexDesc, texObject)
end

function cudaGetTextureObjectResourceViewDesc(pResViewDesc::Array{cudaResourceViewDesc, 1}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectResourceViewDesc, libcudart), cudaError_t, (Ref{cudaResourceViewDesc}, cudaTextureObject_t,), Base.cconvert(Ref{cudaResourceViewDesc}, pResViewDesc), texObject)
end

function cudaGetTextureObjectResourceViewDesc(pResViewDesc::Ptr{cudaResourceViewDesc}, texObject::cudaTextureObject_t)::cudaError_t
    return ccall((:cudaGetTextureObjectResourceViewDesc, libcudart), cudaError_t, (Ptr{cudaResourceViewDesc}, cudaTextureObject_t,), pResViewDesc, texObject)
end

function cudaCreateSurfaceObject(pSurfObject::Array{cudaSurfaceObject_t, 1}, pResDesc::Array{cudaResourceDesc, 1})::cudaError_t
    return ccall((:cudaCreateSurfaceObject, libcudart), cudaError_t, (Ref{cudaSurfaceObject_t}, Ref{cudaResourceDesc},), Base.cconvert(Ref{cudaSurfaceObject_t}, pSurfObject), Base.cconvert(Ref{cudaResourceDesc}, pResDesc))
end

function cudaCreateSurfaceObject(pSurfObject::Ptr{cudaSurfaceObject_t}, pResDesc::Ptr{cudaResourceDesc})::cudaError_t
    return ccall((:cudaCreateSurfaceObject, libcudart), cudaError_t, (Ptr{cudaSurfaceObject_t}, Ptr{cudaResourceDesc},), pSurfObject, pResDesc)
end

function cudaDestroySurfaceObject(surfObject::cudaSurfaceObject_t)::cudaError_t
    return ccall((:cudaDestroySurfaceObject, libcudart), cudaError_t, (cudaSurfaceObject_t,), surfObject)
end

function cudaGetSurfaceObjectResourceDesc(pResDesc::Array{cudaResourceDesc, 1}, surfObject::cudaSurfaceObject_t)::cudaError_t
    return ccall((:cudaGetSurfaceObjectResourceDesc, libcudart), cudaError_t, (Ref{cudaResourceDesc}, cudaSurfaceObject_t,), Base.cconvert(Ref{cudaResourceDesc}, pResDesc), surfObject)
end

function cudaGetSurfaceObjectResourceDesc(pResDesc::Ptr{cudaResourceDesc}, surfObject::cudaSurfaceObject_t)::cudaError_t
    return ccall((:cudaGetSurfaceObjectResourceDesc, libcudart), cudaError_t, (Ptr{cudaResourceDesc}, cudaSurfaceObject_t,), pResDesc, surfObject)
end

function cudaDriverGetVersion(driverVersion::Array{Cint, 1})::cudaError_t
    return ccall((:cudaDriverGetVersion, libcudart), cudaError_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, driverVersion))
end

function cudaDriverGetVersion(driverVersion::Ptr{Cint})::cudaError_t
    return ccall((:cudaDriverGetVersion, libcudart), cudaError_t, (Ptr{Cint},), driverVersion)
end

function cudaRuntimeGetVersion(runtimeVersion::Array{Cint, 1})::cudaError_t
    return ccall((:cudaRuntimeGetVersion, libcudart), cudaError_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, runtimeVersion))
end

function cudaRuntimeGetVersion(runtimeVersion::Ptr{Cint})::cudaError_t
    return ccall((:cudaRuntimeGetVersion, libcudart), cudaError_t, (Ptr{Cint},), runtimeVersion)
end

function cudaGetExportTable(ppExportTable::Array{Ptr{Cvoid}, 1}, pExportTableId::Array{cudaUUID_t, 1})::cudaError_t
    return ccall((:cudaGetExportTable, libcudart), cudaError_t, (Ref{Ptr{Cvoid}}, Ref{cudaUUID_t},), Base.cconvert(Ref{Ptr{Cvoid}}, ppExportTable), Base.cconvert(Ref{cudaUUID_t}, pExportTableId))
end

function cudaGetExportTable(ppExportTable::Ptr{Ptr{Cvoid}}, pExportTableId::Ptr{cudaUUID_t})::cudaError_t
    return ccall((:cudaGetExportTable, libcudart), cudaError_t, (Ptr{Ptr{Cvoid}}, Ptr{cudaUUID_t},), ppExportTable, pExportTableId)
end
