#=*
* Check if CUDA v9.2 functions are wrapped properly.
*
* These tests assumed cuInit() has not been called.
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

function cuda_error_name(res::CUresult)::String
    local c_str::Array{Ptr{UInt8}, 1} = Array{Ptr{UInt8}, 1}()
    push!(c_str, Ptr{UInt8}(C_NULL))
    cuGetErrorName(res, c_str)
    return unsafe_string(c_str[1])
end

println("cuDriverGetVersion(): ", cuda_error_name(cuDriverGetVersion(Ptr{Cint}(C_NULL))))
@test (cuDriverGetVersion(Ptr{Cint}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuDeviceGet(): ", cuda_error_name(cuDeviceGet(Ptr{CUdevice}(C_NULL), Cint(0))))
@test (cuDeviceGet(Ptr{CUdevice}(C_NULL), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetCount(): ", cuda_error_name(cuDeviceGetCount(Ptr{Cint}(C_NULL))))
@test (cuDeviceGetCount(Ptr{Cint}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetName(): ", cuda_error_name(cuDeviceGetName(Ptr{UInt8}(C_NULL), Cint(0), typemax(CUdevice))))
@test (cuDeviceGetName(Ptr{UInt8}(C_NULL), Cint(0), typemax(CUdevice)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetUuid(): ", cuda_error_name(cuDeviceGetUuid(Ptr{CUuuid}(C_NULL), CUdevice(0))))
@test (cuDeviceGetUuid(Ptr{CUuuid}(C_NULL), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceTotalMem(): ", cuda_error_name(cuDeviceTotalMem(Ptr{Csize_t}(C_NULL), CUdevice(0))))
@test (cuDeviceTotalMem(Ptr{Csize_t}(C_NULL), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetAttribute(): ", cuda_error_name(cuDeviceGetAttribute(Ptr{Cint}(C_NULL), CUdevice_attribute(0), CUdevice(0))))
@test (cuDeviceGetAttribute(Ptr{Cint}(C_NULL), CUdevice_attribute(0), CUdevice(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuDeviceGetProperties(): ", cuda_error_name(cuDeviceGetProperties(Ptr{CUdevprop}(C_NULL), CUdevice(0))))
@test (cuDeviceGetProperties(Ptr{CUdevprop}(C_NULL), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceComputeCapability(): ", cuda_error_name(cuDeviceComputeCapability(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUdevice(0))))
@test (cuDeviceComputeCapability(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDevicePrimaryCtxRetain(): ", cuda_error_name(cuDevicePrimaryCtxRetain(Ptr{CUcontext}(C_NULL), CUdevice(0))))
@test (cuDevicePrimaryCtxRetain(Ptr{CUcontext}(C_NULL), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDevicePrimaryCtxRelease(): ", cuda_error_name(cuDevicePrimaryCtxRelease(CUdevice(0))))
@test (cuDevicePrimaryCtxRelease(CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDevicePrimaryCtxSetFlags(): ", cuda_error_name(cuDevicePrimaryCtxSetFlags(CUdevice(0), Cuint(0))))
@test (cuDevicePrimaryCtxSetFlags(CUdevice(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDevicePrimaryCtxGetState(): ", cuda_error_name(cuDevicePrimaryCtxGetState(CUdevice(0), Ptr{Cuint}(C_NULL), Ptr{Cint}(C_NULL))))
@test (cuDevicePrimaryCtxGetState(CUdevice(0), Ptr{Cuint}(C_NULL), Ptr{Cint}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDevicePrimaryCtxReset(): ", cuda_error_name(cuDevicePrimaryCtxReset(CUdevice(0))))
@test (cuDevicePrimaryCtxReset(CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxCreate(): ", cuda_error_name(cuCtxCreate(Ptr{CUcontext}(C_NULL), Cuint(0), CUdevice(0))))
@test (cuCtxCreate(Ptr{CUcontext}(C_NULL), Cuint(0), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxDestroy(): ", cuda_error_name(cuCtxDestroy(CUcontext(C_NULL))))
@test (cuCtxDestroy(CUcontext(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxPushCurrent(): ", cuda_error_name(cuCtxPushCurrent(CUcontext(C_NULL))))
@test (cuCtxPushCurrent(CUcontext(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxPopCurrent(): ", cuda_error_name(cuCtxPopCurrent(Ptr{CUcontext}(C_NULL))))
@test (cuCtxPopCurrent(Ptr{CUcontext}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxSetCurrent(): ", cuda_error_name(cuCtxSetCurrent(CUcontext(C_NULL))))
@test (cuCtxSetCurrent(CUcontext(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetCurrent(): ", cuda_error_name(cuCtxGetCurrent(Ptr{CUcontext}(C_NULL))))
@test (cuCtxGetCurrent(Ptr{CUcontext}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetDevice(): ", cuda_error_name(cuCtxGetDevice(Ptr{CUdevice}(C_NULL))))
@test (cuCtxGetDevice(Ptr{CUdevice}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetFlags(): ", cuda_error_name(cuCtxGetFlags(Ptr{Cuint}(C_NULL))))
@test (cuCtxGetFlags(Ptr{CUctx_flags}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuCtxSynchronize(): ", cuda_error_name(cuCtxSynchronize()))
@test (cuCtxSynchronize() == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxSetLimit(): ", cuda_error_name(cuCtxSetLimit(CUlimit(0), Csize_t(0))))
@test (cuCtxSetLimit(CUlimit(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetLimit(): ", cuda_error_name(cuCtxGetLimit(Ptr{Csize_t}(C_NULL), CUlimit(0))))
@test (cuCtxGetLimit(Ptr{Csize_t}(C_NULL), CUlimit(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetCacheConfig(): ", cuda_error_name(cuCtxGetCacheConfig(Ptr{CUfunc_cache}(C_NULL))))
@test (cuCtxGetCacheConfig(Ptr{CUfunc_cache}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxSetCacheConfig(): ", cuda_error_name(cuCtxSetCacheConfig(CUfunc_cache(0))))
@test (cuCtxSetCacheConfig(CUfunc_cache(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetSharedMemConfig(): ", cuda_error_name(cuCtxGetSharedMemConfig(Ptr{CUsharedconfig}(C_NULL))))
@test (cuCtxGetSharedMemConfig(Ptr{CUsharedconfig}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxSetSharedMemConfig(): ", cuda_error_name(cuCtxSetSharedMemConfig(CUsharedconfig(0))))
@test (cuCtxSetSharedMemConfig(CUsharedconfig(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetApiVersion(): ", cuda_error_name(cuCtxGetApiVersion(CUcontext(C_NULL), Ptr{Cuint}(C_NULL))))
@test (cuCtxGetApiVersion(CUcontext(C_NULL), Ptr{Cuint}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxGetStreamPriorityRange(): ", cuda_error_name(cuCtxGetStreamPriorityRange(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL))))
@test (cuCtxGetStreamPriorityRange(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxAttach(): ", cuda_error_name(cuCtxAttach(Ptr{CUcontext}(C_NULL), Cuint(0))))
@test (cuCtxAttach(Ptr{CUcontext}(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxDetach(): ", cuda_error_name(cuCtxDetach(CUcontext(C_NULL))))
@test (cuCtxDetach(CUcontext(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleLoad(): ", cuda_error_name(cuModuleLoad(Ptr{CUmodule}(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuModuleLoad(Ptr{CUmodule}(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleLoadData(): ", cuda_error_name(cuModuleLoadData(Ptr{CUmodule}(C_NULL), C_NULL)))
@test (cuModuleLoadData(Ptr{CUmodule}(C_NULL), C_NULL) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleLoadDataEx(): ", cuda_error_name(cuModuleLoadDataEx(Ptr{CUmodule}(C_NULL), C_NULL, Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL))))
@test (cuModuleLoadDataEx(Ptr{CUmodule}(C_NULL), C_NULL, Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleLoadFatBinary(): ", cuda_error_name(cuModuleLoadFatBinary(Ptr{CUmodule}(C_NULL), C_NULL)))
@test (cuModuleLoadFatBinary(Ptr{CUmodule}(C_NULL), C_NULL) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleUnload(): ", cuda_error_name(cuModuleUnload(CUmodule(C_NULL))))
@test (cuModuleUnload(CUmodule(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleGetFunction(): ", cuda_error_name(cuModuleGetFunction(Ptr{CUfunction}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuModuleGetFunction(Ptr{CUfunction}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleGetGlobal(): ", cuda_error_name(cuModuleGetGlobal(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuModuleGetGlobal(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleGetTexRef(): ", cuda_error_name(cuModuleGetTexRef(Ptr{CUtexref}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuModuleGetTexRef(Ptr{CUtexref}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuModuleGetSurfRef(): ", cuda_error_name(cuModuleGetSurfRef(Ptr{CUsurfref}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuModuleGetSurfRef(Ptr{CUsurfref}(C_NULL), CUmodule(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLinkCreate(): ", cuda_error_name(cuLinkCreate(Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{CUlinkState}(C_NULL))))
@test (cuLinkCreate(Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{CUlinkState}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLinkAddData(): ", cuda_error_name(cuLinkAddData(CUlinkState(C_NULL), CUjitInputType(0), C_NULL, Csize_t(0), Ptr{UInt8}(C_NULL), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL))))
@test (cuLinkAddData(CUlinkState(C_NULL), CUjitInputType(0), C_NULL, Csize_t(0), Ptr{UInt8}(C_NULL), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuLinkAddFile(): ", cuda_error_name(cuLinkAddFile(CUlinkState(C_NULL), CUjitInputType(0), Ptr{UInt8}(C_NULL), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL))))
@test (cuLinkAddFile(CUlinkState(C_NULL), CUjitInputType(0), Ptr{UInt8}(C_NULL), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuLinkComplete(): ", cuda_error_name(cuLinkComplete(CUlinkState(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL))))
@test (cuLinkComplete(CUlinkState(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuLinkDestroy(): ", cuda_error_name(cuLinkDestroy(CUlinkState(C_NULL))))
@test (cuLinkDestroy(CUlinkState(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuMemGetInfo(): ", cuda_error_name(cuMemGetInfo(Ptr{Csize_t}(C_NULL), Ptr{Csize_t}(C_NULL))))
@test (cuMemGetInfo(Ptr{Csize_t}(C_NULL), Ptr{Csize_t}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemAlloc(): ", cuda_error_name(cuMemAlloc(Ptr{CUdeviceptr}(C_NULL), Csize_t(0))))
@test (cuMemAlloc(Ptr{CUdeviceptr}(C_NULL), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemAllocPitch(): ", cuda_error_name(cuMemAllocPitch(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), Csize_t(0), Csize_t(0), Cuint(0))))
@test (cuMemAllocPitch(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), Csize_t(0), Csize_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemFree(): ", cuda_error_name(cuMemFree(CUdeviceptr(0))))
@test (cuMemFree(CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemGetAddressRange(): ", cuda_error_name(cuMemGetAddressRange(Ptr{CUdeviceptr}(C_NULL), Base.unsafe_convert(Ptr{Csize_t}, zeros(Csize_t, 1)), CUdeviceptr(0))))
@test (cuMemGetAddressRange(Ptr{CUdeviceptr}(C_NULL), Base.unsafe_convert(Ptr{Csize_t}, zeros(Csize_t, 1)), CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemAllocHost(): ", cuda_error_name(cuMemAllocHost(Ptr{Ptr{Cvoid}}(C_NULL), Csize_t(0))))
@test (cuMemAllocHost(Ptr{Ptr{Cvoid}}(C_NULL), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemFreeHost(): ", cuda_error_name(cuMemFreeHost(C_NULL)))
@test (cuMemFreeHost(C_NULL) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemHostAlloc(): ", cuda_error_name(cuMemHostAlloc(Ptr{Ptr{Cvoid}}(C_NULL), Csize_t(0), Cuint(0))))
@test (cuMemHostAlloc(Ptr{Ptr{Cvoid}}(C_NULL), Csize_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemHostGetDevicePointer(): ", cuda_error_name(cuMemHostGetDevicePointer(Ptr{CUdeviceptr}(C_NULL), C_NULL, Cuint(0))))
@test (cuMemHostGetDevicePointer(Ptr{CUdeviceptr}(C_NULL), C_NULL, Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemHostGetFlags(): ", cuda_error_name(cuMemHostGetFlags(Ptr{Cuint}(C_NULL), C_NULL)))
@test (cuMemHostGetFlags(Ptr{Cuint}(C_NULL), C_NULL) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemAllocManaged(): ", cuda_error_name(cuMemAllocManaged(Ptr{CUdeviceptr}(C_NULL), Csize_t(0), Cuint(0))))
@test (cuMemAllocManaged(Ptr{CUdeviceptr}(C_NULL), Csize_t(0), Cuint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuDeviceGetByPCIBusId(): ", cuda_error_name(cuDeviceGetByPCIBusId(Ptr{CUdevice}(C_NULL), Ptr{UInt8}(C_NULL))))
@test (cuDeviceGetByPCIBusId(Ptr{CUdevice}(C_NULL), Ptr{UInt8}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetPCIBusId(): ", cuda_error_name(cuDeviceGetPCIBusId(Ptr{UInt8}(C_NULL), Cint(0), CUdevice(0))))
@test (cuDeviceGetPCIBusId(Ptr{UInt8}(C_NULL), Cint(0), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuIpcGetEventHandle(): ", cuda_error_name(cuIpcGetEventHandle(Ptr{CUipcEventHandle}(C_NULL), CUevent(C_NULL))))
@test (cuIpcGetEventHandle(Ptr{CUipcEventHandle}(C_NULL), CUevent(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuIpcOpenEventHandle(): ", cuda_error_name(cuIpcOpenEventHandle(Ptr{CUevent}(C_NULL), zero(CUipcEventHandle))))
@test (cuIpcOpenEventHandle(Ptr{CUevent}(C_NULL), zero(CUipcEventHandle)) == CUDA_ERROR_INVALID_VALUE)

println("cuIpcGetMemHandle(): ", cuda_error_name(cuIpcGetMemHandle(Ptr{CUipcMemHandle}(C_NULL), CUdeviceptr(0))))
@test (cuIpcGetMemHandle(Ptr{CUipcMemHandle}(C_NULL), CUdeviceptr(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuIpcOpenMemHandle(): ", cuda_error_name(cuIpcOpenMemHandle(Ptr{CUdeviceptr}(C_NULL), zero(CUipcMemHandle), Cuint(0))))
@test (cuIpcOpenMemHandle(Ptr{CUdeviceptr}(C_NULL), zero(CUipcMemHandle), Cuint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuIpcCloseMemHandle(): ", cuda_error_name(cuIpcCloseMemHandle(CUdeviceptr(0))))
@test (cuIpcCloseMemHandle(CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemHostRegister(): ", cuda_error_name(cuMemHostRegister(C_NULL, Csize_t(0), Cuint(0))))
@test (cuMemHostRegister(C_NULL, Csize_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemHostUnregister(): ", cuda_error_name(cuMemHostUnregister(C_NULL)))
@test (cuMemHostUnregister(C_NULL) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy(): ", cuda_error_name(cuMemcpy(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0))))
@test (cuMemcpy(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyPeer(): ", cuda_error_name(cuMemcpyPeer(CUdeviceptr(0), CUcontext(C_NULL), CUdeviceptr(0), CUcontext(C_NULL), Csize_t(0))))
@test (cuMemcpyPeer(CUdeviceptr(0), CUcontext(C_NULL), CUdeviceptr(0), CUcontext(C_NULL), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyHtoD(): ", cuda_error_name(cuMemcpyHtoD(CUdeviceptr(0), C_NULL, Csize_t(0))))
@test (cuMemcpyHtoD(CUdeviceptr(0), C_NULL, Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyDtoH(): ", cuda_error_name(cuMemcpyDtoH(C_NULL, CUdeviceptr(0), Csize_t(0))))
@test (cuMemcpyDtoH(C_NULL, CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyDtoD(): ", cuda_error_name(cuMemcpyDtoD(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0))))
@test (cuMemcpyDtoD(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyDtoA(): ", cuda_error_name(cuMemcpyDtoA(CUarray(C_NULL), Csize_t(0), CUdeviceptr(0), Csize_t(0))))
@test (cuMemcpyDtoA(CUarray(C_NULL), Csize_t(0), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyAtoD(): ", cuda_error_name(cuMemcpyAtoD(CUdeviceptr(0), CUarray(C_NULL), Csize_t(0), Csize_t(0))))
@test (cuMemcpyAtoD(CUdeviceptr(0), CUarray(C_NULL), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyHtoA(): ", cuda_error_name(cuMemcpyHtoA(CUarray(C_NULL), Csize_t(0), C_NULL, Csize_t(0))))
@test (cuMemcpyHtoA(CUarray(C_NULL), Csize_t(0), C_NULL, Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyAtoH(): ", cuda_error_name(cuMemcpyAtoH(C_NULL, CUarray(C_NULL), Csize_t(0), Csize_t(0))))
@test (cuMemcpyAtoH(C_NULL, CUarray(C_NULL), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyAtoA(): ", cuda_error_name(cuMemcpyAtoA(CUarray(C_NULL), Csize_t(0), CUarray(C_NULL), Csize_t(0), Csize_t(0))))
@test (cuMemcpyAtoA(CUarray(C_NULL), Csize_t(0), CUarray(C_NULL), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy2D(): ", cuda_error_name(cuMemcpy2D(Ptr{CUDA_MEMCPY2D}(C_NULL))))
@test (cuMemcpy2D(Ptr{CUDA_MEMCPY2D}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy2DUnaligned(): ", cuda_error_name(cuMemcpy2DUnaligned(Ptr{CUDA_MEMCPY2D}(C_NULL))))
@test (cuMemcpy2DUnaligned(Ptr{CUDA_MEMCPY2D}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy3D(): ", cuda_error_name(cuMemcpy3D(Ptr{CUDA_MEMCPY3D}(C_NULL))))
@test (cuMemcpy3D(Ptr{CUDA_MEMCPY3D}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy3DPeer(): ", cuda_error_name(cuMemcpy3DPeer(Ptr{CUDA_MEMCPY3D_PEER}(C_NULL))))
@test (cuMemcpy3DPeer(Ptr{CUDA_MEMCPY3D_PEER}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyAsync(): ", cuda_error_name(cuMemcpyAsync(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyAsync(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyPeerAsync(): ", cuda_error_name(cuMemcpyPeerAsync(CUdeviceptr(0), CUcontext(C_NULL), CUdeviceptr(0), CUcontext(C_NULL), Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyPeerAsync(CUdeviceptr(0), CUcontext(C_NULL), CUdeviceptr(0), CUcontext(C_NULL), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyHtoDAsync(): ", cuda_error_name(cuMemcpyHtoDAsync(CUdeviceptr(0), C_NULL, Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyHtoDAsync(CUdeviceptr(0), C_NULL, Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyDtoHAsync(): ", cuda_error_name(cuMemcpyDtoHAsync(C_NULL, CUdeviceptr(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyDtoHAsync(C_NULL, CUdeviceptr(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyDtoDAsync(): ", cuda_error_name(cuMemcpyDtoDAsync(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyDtoDAsync(CUdeviceptr(0), CUdeviceptr(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyHtoAAsync(): ", cuda_error_name(cuMemcpyHtoAAsync(CUarray(C_NULL), Csize_t(0), C_NULL, Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyHtoAAsync(CUarray(C_NULL), Csize_t(0), C_NULL, Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpyAtoHAsync(): ", cuda_error_name(cuMemcpyAtoHAsync(C_NULL, CUarray(C_NULL), Csize_t(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemcpyAtoHAsync(C_NULL, CUarray(C_NULL), Csize_t(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy2DAsync(): ", cuda_error_name(cuMemcpy2DAsync(Ptr{CUDA_MEMCPY2D}(C_NULL), CUstream(C_NULL))))
@test (cuMemcpy2DAsync(Ptr{CUDA_MEMCPY2D}(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy3DAsync(): ", cuda_error_name(cuMemcpy3DAsync(Ptr{CUDA_MEMCPY3D}(C_NULL), CUstream(C_NULL))))
@test (cuMemcpy3DAsync(Ptr{CUDA_MEMCPY3D}(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemcpy3DPeerAsync(): ", cuda_error_name(cuMemcpy3DPeerAsync(Ptr{CUDA_MEMCPY3D_PEER}(C_NULL), CUstream(C_NULL))))
@test (cuMemcpy3DPeerAsync(Ptr{CUDA_MEMCPY3D_PEER}(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD8(): ", cuda_error_name(cuMemsetD8(CUdeviceptr(0), UInt8(0), Csize_t(0))))
@test (cuMemsetD8(CUdeviceptr(0), UInt8(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD16(): ", cuda_error_name(cuMemsetD16(CUdeviceptr(0), UInt16(0), Csize_t(0))))
@test (cuMemsetD16(CUdeviceptr(0), UInt16(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD32(): ", cuda_error_name(cuMemsetD32(CUdeviceptr(0), Cuint(0), Csize_t(0))))
@test (cuMemsetD32(CUdeviceptr(0), Cuint(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D8(): ", cuda_error_name(cuMemsetD2D8(CUdeviceptr(0), Csize_t(0), UInt8(0), Csize_t(0), Csize_t(0))))
@test (cuMemsetD2D8(CUdeviceptr(0), Csize_t(0), UInt8(0), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D16(): ", cuda_error_name(cuMemsetD2D16(CUdeviceptr(0), Csize_t(0), UInt16(0), Csize_t(0), Csize_t(0))))
@test (cuMemsetD2D16(CUdeviceptr(0), Csize_t(0), UInt16(0), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D32(): ", cuda_error_name(cuMemsetD2D32(CUdeviceptr(0), Csize_t(0), Cuint(0), Csize_t(0), Csize_t(0))))
@test (cuMemsetD2D32(CUdeviceptr(0), Csize_t(0), Cuint(0), Csize_t(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD8Async(): ", cuda_error_name(cuMemsetD8Async(CUdeviceptr(0), UInt8(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD8Async(CUdeviceptr(0), UInt8(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD16Async(): ", cuda_error_name(cuMemsetD16Async(CUdeviceptr(0), UInt16(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD16Async(CUdeviceptr(0), UInt16(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD32Async(): ", cuda_error_name(cuMemsetD32Async(CUdeviceptr(0), Cuint(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD32Async(CUdeviceptr(0), Cuint(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D8Async(): ", cuda_error_name(cuMemsetD2D8Async(CUdeviceptr(0), Csize_t(0), UInt8(0), Csize_t(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD2D8Async(CUdeviceptr(0), Csize_t(0), UInt8(0), Csize_t(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D16Async(): ", cuda_error_name(cuMemsetD2D16Async(CUdeviceptr(0), Csize_t(0), UInt16(0), Csize_t(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD2D16Async(CUdeviceptr(0), Csize_t(0), UInt16(0), Csize_t(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemsetD2D32Async(): ", cuda_error_name(cuMemsetD2D32Async(CUdeviceptr(0), Csize_t(0), Cuint(0), Csize_t(0), Csize_t(0), CUstream(C_NULL))))
@test (cuMemsetD2D32Async(CUdeviceptr(0), Csize_t(0), Cuint(0), Csize_t(0), Csize_t(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuArrayCreate(): ", cuda_error_name(cuArrayCreate(Ptr{CUarray}(C_NULL), Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL))))
@test (cuArrayCreate(Ptr{CUarray}(C_NULL), Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuArrayGetDescriptor(): ", cuda_error_name(cuArrayGetDescriptor(Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL), CUarray(C_NULL))))
@test (cuArrayGetDescriptor(Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL), CUarray(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuArrayDestroy(): ", cuda_error_name(cuArrayDestroy(CUarray(C_NULL))))
@test (cuArrayDestroy(CUarray(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuArray3DCreate(): ", cuda_error_name(cuArray3DCreate(Ptr{CUarray}(C_NULL), Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL))))
@test (cuArray3DCreate(Ptr{CUarray}(C_NULL), Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuArray3DGetDescriptor(): ", cuda_error_name(cuArray3DGetDescriptor(Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL), CUarray(C_NULL))))
@test (cuArray3DGetDescriptor(Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL), CUarray(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuMipmappedArrayCreate(): ", cuda_error_name(cuMipmappedArrayCreate(Ptr{CUmipmappedArray}(C_NULL), Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL), Cuint(0))))
@test (cuMipmappedArrayCreate(Ptr{CUmipmappedArray}(C_NULL), Ptr{CUDA_ARRAY3D_DESCRIPTOR}(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMipmappedArrayGetLevel(): ", cuda_error_name(cuMipmappedArrayGetLevel(Ptr{CUarray}(C_NULL), CUmipmappedArray(C_NULL), Cuint(0))))
@test (cuMipmappedArrayGetLevel(Ptr{CUarray}(C_NULL), CUmipmappedArray(C_NULL), Cuint(0)) == CUDA_ERROR_INVALID_HANDLE)

println("cuMipmappedArrayDestroy(): ", cuda_error_name(cuMipmappedArrayDestroy(CUmipmappedArray(C_NULL))))
@test (cuMipmappedArrayDestroy(CUmipmappedArray(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuPointerGetAttribute(): ", cuda_error_name(cuPointerGetAttribute(C_NULL, CUpointer_attribute(0), CUdeviceptr(0))))
@test (cuPointerGetAttribute(C_NULL, CUpointer_attribute(0), CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemPrefetchAsync(): ", cuda_error_name(cuMemPrefetchAsync(CUdeviceptr(0), Csize_t(0), CUdevice(0), CUstream(C_NULL))))
@test (cuMemPrefetchAsync(CUdeviceptr(0), Csize_t(0), CUdevice(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemAdvise(): ", cuda_error_name(cuMemAdvise(CUdeviceptr(0), Csize_t(0), CUmem_advise(0), CUdevice(0))))
@test (cuMemAdvise(CUdeviceptr(0), Csize_t(0), CUmem_advise(0), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemRangeGetAttribute(): ", cuda_error_name(cuMemRangeGetAttribute(C_NULL, Csize_t(0), CUmem_range_attribute(0), CUdeviceptr(0), Csize_t(0))))
@test (cuMemRangeGetAttribute(C_NULL, Csize_t(0), CUmem_range_attribute(0), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuMemRangeGetAttributes(): ", cuda_error_name(cuMemRangeGetAttributes(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), Ptr{CUmem_range_attribute}(C_NULL), Csize_t(0), CUdeviceptr(0), Csize_t(0))))
@test (cuMemRangeGetAttributes(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Csize_t}(C_NULL), Ptr{CUmem_range_attribute}(C_NULL), Csize_t(0), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuPointerSetAttribute(): ", cuda_error_name(cuPointerSetAttribute(C_NULL, CUpointer_attribute(0), CUdeviceptr(0))))
@test (cuPointerSetAttribute(C_NULL, CUpointer_attribute(0), CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuPointerGetAttributes(): ", cuda_error_name(cuPointerGetAttributes(Cuint(0), Ptr{CUpointer_attribute}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), CUdeviceptr(0))))
@test (cuPointerGetAttributes(Cuint(0), Ptr{CUpointer_attribute}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), CUdeviceptr(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamCreate(): ", cuda_error_name(cuStreamCreate(Ptr{CUstream}(C_NULL), Cuint(0))))
@test (cuStreamCreate(Ptr{CUstream}(C_NULL), Cuint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamCreateWithPriority(): ", cuda_error_name(cuStreamCreateWithPriority(Ptr{CUstream}(C_NULL), Cuint(0), Cint(0))))
@test (cuStreamCreateWithPriority(Ptr{CUstream}(C_NULL), Cuint(0), Cint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamGetPriority(): ", cuda_error_name(cuStreamGetPriority(CUstream(C_NULL), Ptr{Cint}(C_NULL))))
@test (cuStreamGetPriority(CUstream(C_NULL), Ptr{Cint}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamGetFlags(): ", cuda_error_name(cuStreamGetFlags(CUstream(C_NULL), Ptr{Cuint}(C_NULL))))
@test (cuStreamGetFlags(CUstream(C_NULL), Ptr{Cuint}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamGetCtx(): ", cuda_error_name(cuStreamGetCtx(CUstream(C_NULL), Ptr{CUcontext}(C_NULL))))
@test (cuStreamGetCtx(CUstream(C_NULL), Ptr{CUcontext}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamWaitEvent(): ", cuda_error_name(cuStreamWaitEvent(CUstream(C_NULL), CUevent(C_NULL), Cuint(0))))
@test (cuStreamWaitEvent(CUstream(C_NULL), CUevent(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamAddCallback(): ", cuda_error_name(cuStreamAddCallback(CUstream(C_NULL), CUstreamCallback(C_NULL), C_NULL, Cuint(0))))
@test (cuStreamAddCallback(CUstream(C_NULL), CUstreamCallback(C_NULL), C_NULL, Cuint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuStreamAttachMemAsync(): ", cuda_error_name(cuStreamAttachMemAsync(CUstream(C_NULL), CUdeviceptr(0), Csize_t(0), Cuint(0))))
@test (cuStreamAttachMemAsync(CUstream(C_NULL), CUdeviceptr(0), Csize_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamQuery(): ", cuda_error_name(cuStreamQuery(CUstream(C_NULL))))
@test (cuStreamQuery(CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamSynchronize(): ", cuda_error_name(cuStreamSynchronize(CUstream(C_NULL))))
@test (cuStreamSynchronize(CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamDestroy(): ", cuda_error_name(cuStreamDestroy(CUstream(C_NULL))))
@test (cuStreamDestroy(CUstream(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuEventCreate(): ", cuda_error_name(cuEventCreate(Ptr{CUevent}(C_NULL), Cuint(0))))
@test (cuEventCreate(Ptr{CUevent}(C_NULL), CUevent_flags(0)) == CUDA_ERROR_NOT_INITIALIZED)

# segmentation fault when passing C_NULL to cuEventRecord()
# println("cuEventRecord(): ", cuda_error_name(cuEventRecord(CUevent(C_NULL), CUstream(C_NULL))))
@test_skip (cuEventRecord(CUevent(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuEventQuery(): ", cuda_error_name(cuEventQuery(CUevent(C_NULL))))
@test (cuEventQuery(CUevent(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuEventSynchronize(): ", cuda_error_name(cuEventSynchronize(CUevent(C_NULL))))
@test (cuEventSynchronize(CUevent(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuEventDestroy(): ", cuda_error_name(cuEventDestroy(CUevent(C_NULL))))
@test (cuEventDestroy(CUevent(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuEventElapsedTime(): ", cuda_error_name(cuEventElapsedTime(Ptr{Cfloat}(C_NULL), CUevent(C_NULL), CUevent(C_NULL))))
@test (cuEventElapsedTime(Ptr{Cfloat}(C_NULL), CUevent(C_NULL), CUevent(C_NULL)) == CUDA_ERROR_INVALID_HANDLE)

println("cuStreamWaitValue32(): ", cuda_error_name(cuStreamWaitValue32(CUstream(C_NULL), CUdeviceptr(0), cuuint32_t(0), Cuint(0))))
@test (cuStreamWaitValue32(CUstream(C_NULL), CUdeviceptr(0), cuuint32_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamWaitValue64(): ", cuda_error_name(cuStreamWaitValue64(CUstream(C_NULL), CUdeviceptr(0), cuuint64_t(0), Cuint(0))))
@test (cuStreamWaitValue64(CUstream(C_NULL), CUdeviceptr(0), cuuint64_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamWriteValue32(): ", cuda_error_name(cuStreamWriteValue32(CUstream(C_NULL), CUdeviceptr(0), cuuint32_t(0), Cuint(0))))
@test (cuStreamWriteValue32(CUstream(C_NULL), CUdeviceptr(0), cuuint32_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamWriteValue64(): ", cuda_error_name(cuStreamWriteValue64(CUstream(C_NULL), CUdeviceptr(0), cuuint64_t(0), Cuint(0))))
@test (cuStreamWriteValue64(CUstream(C_NULL), CUdeviceptr(0), cuuint64_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuStreamBatchMemOp(): ", cuda_error_name(cuStreamBatchMemOp(CUstream(C_NULL), Cuint(0), Ptr{CUstreamBatchMemOpParams}(C_NULL), Cuint(0))))
@test (cuStreamBatchMemOp(CUstream(C_NULL), Cuint(0), Ptr{CUstreamBatchMemOpParams}(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuFuncGetAttribute(): ", cuda_error_name(cuFuncGetAttribute(Ptr{Cint}(C_NULL), CUfunction_attribute(0), CUfunction(C_NULL))))
@test (cuFuncGetAttribute(Ptr{Cint}(C_NULL), CUfunction_attribute(0), CUfunction(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuFuncSetAttribute(): ", cuda_error_name(cuFuncSetAttribute(CUfunction(C_NULL), CUfunction_attribute(0), Cint(0))))
@test (cuFuncSetAttribute(CUfunction(C_NULL), CUfunction_attribute(0), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuFuncSetCacheConfig(): ", cuda_error_name(cuFuncSetCacheConfig(CUfunction(C_NULL), CUfunc_cache(0))))
@test (cuFuncSetCacheConfig(CUfunction(C_NULL), CUfunc_cache(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuFuncSetSharedMemConfig(): ", cuda_error_name(cuFuncSetSharedMemConfig(CUfunction(C_NULL), CUsharedconfig(0))))
@test (cuFuncSetSharedMemConfig(CUfunction(C_NULL), CUsharedconfig(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunchKernel(): ", cuda_error_name(cuLaunchKernel(CUfunction(C_NULL), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), CUstream(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL))))
@test (cuLaunchKernel(CUfunction(C_NULL), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), CUstream(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunchCooperativeKernel(): ", cuda_error_name(cuLaunchCooperativeKernel(CUfunction(C_NULL), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), CUstream(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL))))
@test (cuLaunchCooperativeKernel(CUfunction(C_NULL), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), CUstream(C_NULL), Ptr{Ptr{Cvoid}}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunchCooperativeKernelMultiDevice(): ", cuda_error_name(cuLaunchCooperativeKernelMultiDevice(Ptr{CUDA_LAUNCH_PARAMS}(C_NULL), Cuint(0), Cuint(0))))
@test (cuLaunchCooperativeKernelMultiDevice(Ptr{CUDA_LAUNCH_PARAMS}(C_NULL), Cuint(0), Cuint(0)) == CUDA_ERROR_INVALID_VALUE)

println("cuFuncSetBlockShape(): ", cuda_error_name(cuFuncSetBlockShape(CUfunction(C_NULL), Cint(0), Cint(0), Cint(0))))
@test (cuFuncSetBlockShape(CUfunction(C_NULL), Cint(0), Cint(0), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuFuncSetSharedSize(): ", cuda_error_name(cuFuncSetSharedSize(CUfunction(C_NULL), Cuint(0))))
@test (cuFuncSetSharedSize(CUfunction(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuParamSetSize(): ", cuda_error_name(cuParamSetSize(CUfunction(C_NULL), Cuint(0))))
@test (cuParamSetSize(CUfunction(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuParamSeti(): ", cuda_error_name(cuParamSeti(CUfunction(C_NULL), Cint(0), Cuint(0))))
@test (cuParamSeti(CUfunction(C_NULL), Cint(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuParamSetf(): ", cuda_error_name(cuParamSetf(CUfunction(C_NULL), Cint(0), Cfloat(0))))
@test (cuParamSetf(CUfunction(C_NULL), Cint(0), Cfloat(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuParamSetv(): ", cuda_error_name(cuParamSetv(CUfunction(C_NULL), Cint(0), C_NULL, Cuint(0))))
@test (cuParamSetv(CUfunction(C_NULL), Cint(0), C_NULL, Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunch(): ", cuda_error_name(cuLaunch(CUfunction(C_NULL))))
@test (cuLaunch(CUfunction(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunchGrid(): ", cuda_error_name(cuLaunchGrid(CUfunction(C_NULL), Cint(0), Cint(0))))
@test (cuLaunchGrid(CUfunction(C_NULL), Cint(0), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuLaunchGridAsync(): ", cuda_error_name(cuLaunchGridAsync(CUfunction(C_NULL), Cint(0), Cint(0), CUstream(C_NULL))))
@test (cuLaunchGridAsync(CUfunction(C_NULL), Cint(0), Cint(0), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuParamSetTexRef(): ", cuda_error_name(cuParamSetTexRef(CUfunction(C_NULL), Cint(0), CUtexref(C_NULL))))
@test (cuParamSetTexRef(CUfunction(C_NULL), Cint(0), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuOccupancyMaxActiveBlocksPerMultiprocessor(): ", cuda_error_name(cuOccupancyMaxActiveBlocksPerMultiprocessor(Ptr{Cint}(C_NULL), CUfunction(C_NULL), Cint(0), Csize_t(0))))
@test (cuOccupancyMaxActiveBlocksPerMultiprocessor(Ptr{Cint}(C_NULL), CUfunction(C_NULL), Cint(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(): ", cuda_error_name(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(Ptr{Cint}(C_NULL), CUfunction(C_NULL), Cint(0), Csize_t(0), Cuint(0))))
@test (cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(Ptr{Cint}(C_NULL), CUfunction(C_NULL), Cint(0), Csize_t(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuOccupancyMaxPotentialBlockSize(): ", cuda_error_name(cuOccupancyMaxPotentialBlockSize(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUfunction(C_NULL), CUoccupancyB2DSize(C_NULL), Csize_t(0), Cint(0))))
@test (cuOccupancyMaxPotentialBlockSize(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUfunction(C_NULL), CUoccupancyB2DSize(C_NULL), Csize_t(0), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuOccupancyMaxPotentialBlockSizeWithFlags(): ", cuda_error_name(cuOccupancyMaxPotentialBlockSizeWithFlags(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUfunction(C_NULL), CUoccupancyB2DSize(C_NULL), Csize_t(0), Cint(0), Cuint(0))))
@test (cuOccupancyMaxPotentialBlockSizeWithFlags(Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), CUfunction(C_NULL), CUoccupancyB2DSize(C_NULL), Csize_t(0), Cint(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetArray(): ", cuda_error_name(cuTexRefSetArray(CUtexref(C_NULL), CUarray(C_NULL), Cuint(0))))
@test (cuTexRefSetArray(CUtexref(C_NULL), CUarray(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetMipmappedArray(): ", cuda_error_name(cuTexRefSetMipmappedArray(CUtexref(C_NULL), CUmipmappedArray(C_NULL), Cuint(0))))
@test (cuTexRefSetMipmappedArray(CUtexref(C_NULL), CUmipmappedArray(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetAddress(): ", cuda_error_name(cuTexRefSetAddress(Ptr{Csize_t}(C_NULL), CUtexref(C_NULL), CUdeviceptr(0), Csize_t(0))))
@test (cuTexRefSetAddress(Ptr{Csize_t}(C_NULL), CUtexref(C_NULL), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetAddress2D(): ", cuda_error_name(cuTexRefSetAddress2D(CUtexref(C_NULL), Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL), CUdeviceptr(0), Csize_t(0))))
@test (cuTexRefSetAddress2D(CUtexref(C_NULL), Ptr{CUDA_ARRAY_DESCRIPTOR}(C_NULL), CUdeviceptr(0), Csize_t(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetFormat(): ", cuda_error_name(cuTexRefSetFormat(CUtexref(C_NULL), CUarray_format(0), Cint(0))))
@test (cuTexRefSetFormat(CUtexref(C_NULL), CUarray_format(0), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetAddressMode(): ", cuda_error_name(cuTexRefSetAddressMode(CUtexref(C_NULL), Cint(0), CUaddress_mode(0))))
@test (cuTexRefSetAddressMode(CUtexref(C_NULL), Cint(0), CUaddress_mode(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetFilterMode(): ", cuda_error_name(cuTexRefSetFilterMode(CUtexref(C_NULL), CUfilter_mode(0))))
@test (cuTexRefSetFilterMode(CUtexref(C_NULL), CUfilter_mode(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetMipmapFilterMode(): ", cuda_error_name(cuTexRefSetMipmapFilterMode(CUtexref(C_NULL), CUfilter_mode(0))))
@test (cuTexRefSetMipmapFilterMode(CUtexref(C_NULL), CUfilter_mode(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetMipmapLevelBias(): ", cuda_error_name(cuTexRefSetMipmapLevelBias(CUtexref(C_NULL), Cfloat(0))))
@test (cuTexRefSetMipmapLevelBias(CUtexref(C_NULL), Cfloat(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetMipmapLevelClamp(): ", cuda_error_name(cuTexRefSetMipmapLevelClamp(CUtexref(C_NULL), Cfloat(0), Cfloat(0))))
@test (cuTexRefSetMipmapLevelClamp(CUtexref(C_NULL), Cfloat(0), Cfloat(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetMaxAnisotropy(): ", cuda_error_name(cuTexRefSetMaxAnisotropy(CUtexref(C_NULL), Cuint(0))))
@test (cuTexRefSetMaxAnisotropy(CUtexref(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetBorderColor(): ", cuda_error_name(cuTexRefSetBorderColor(CUtexref(C_NULL), Ptr{Cfloat}(C_NULL))))
@test (cuTexRefSetBorderColor(CUtexref(C_NULL), Ptr{Cfloat}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefSetFlags(): ", cuda_error_name(cuTexRefSetFlags(CUtexref(C_NULL), Cuint(0))))
@test (cuTexRefSetFlags(CUtexref(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetAddress(): ", cuda_error_name(cuTexRefGetAddress(Ptr{CUdeviceptr}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetAddress(Ptr{CUdeviceptr}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetArray(): ", cuda_error_name(cuTexRefGetArray(Ptr{CUarray}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetArray(Ptr{CUarray}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetMipmappedArray(): ", cuda_error_name(cuTexRefGetMipmappedArray(Ptr{CUmipmappedArray}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetMipmappedArray(Ptr{CUmipmappedArray}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetAddressMode(): ", cuda_error_name(cuTexRefGetAddressMode(Ptr{CUaddress_mode}(C_NULL), CUtexref(C_NULL), Cint(0))))
@test (cuTexRefGetAddressMode(Ptr{CUaddress_mode}(C_NULL), CUtexref(C_NULL), Cint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetFilterMode(): ", cuda_error_name(cuTexRefGetFilterMode(Ptr{CUfilter_mode}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetFilterMode(Ptr{CUfilter_mode}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetFormat(): ", cuda_error_name(cuTexRefGetFormat(Ptr{CUarray_format}(C_NULL), Ptr{Cint}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetFormat(Ptr{CUarray_format}(C_NULL), Ptr{Cint}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetMipmapFilterMode(): ", cuda_error_name(cuTexRefGetMipmapFilterMode(Ptr{CUfilter_mode}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetMipmapFilterMode(Ptr{CUfilter_mode}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetMipmapLevelBias(): ", cuda_error_name(cuTexRefGetMipmapLevelBias(Ptr{Cfloat}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetMipmapLevelBias(Ptr{Cfloat}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetMipmapLevelClamp(): ", cuda_error_name(cuTexRefGetMipmapLevelClamp(Ptr{Cfloat}(C_NULL), Ptr{Cfloat}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetMipmapLevelClamp(Ptr{Cfloat}(C_NULL), Ptr{Cfloat}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetMaxAnisotropy(): ", cuda_error_name(cuTexRefGetMaxAnisotropy(Ptr{Cint}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetMaxAnisotropy(Ptr{Cint}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetBorderColor(): ", cuda_error_name(cuTexRefGetBorderColor(Ptr{Cfloat}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetBorderColor(Ptr{Cfloat}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefGetFlags(): ", cuda_error_name(cuTexRefGetFlags(Ptr{Cuint}(C_NULL), CUtexref(C_NULL))))
@test (cuTexRefGetFlags(Ptr{Cuint}(C_NULL), CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefCreate(): ", cuda_error_name(cuTexRefCreate(Ptr{CUtexref}(C_NULL))))
@test (cuTexRefCreate(Ptr{CUtexref}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexRefDestroy(): ", cuda_error_name(cuTexRefDestroy(CUtexref(C_NULL))))
@test (cuTexRefDestroy(CUtexref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuSurfRefSetArray(): ", cuda_error_name(cuSurfRefSetArray(CUsurfref(C_NULL), CUarray(C_NULL), Cuint(0))))
@test (cuSurfRefSetArray(CUsurfref(C_NULL), CUarray(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuSurfRefGetArray(): ", cuda_error_name(cuSurfRefGetArray(Ptr{CUarray}(C_NULL), CUsurfref(C_NULL))))
@test (cuSurfRefGetArray(Ptr{CUarray}(C_NULL), CUsurfref(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexObjectCreate(): ", cuda_error_name(cuTexObjectCreate(Ptr{CUtexObject}(C_NULL), Ptr{CUDA_RESOURCE_DESC}(C_NULL), Ptr{CUDA_TEXTURE_DESC}(C_NULL), Ptr{CUDA_RESOURCE_VIEW_DESC}(C_NULL))))
@test (cuTexObjectCreate(Ptr{CUtexObject}(C_NULL), Ptr{CUDA_RESOURCE_DESC}(C_NULL), Ptr{CUDA_TEXTURE_DESC}(C_NULL), Ptr{CUDA_RESOURCE_VIEW_DESC}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexObjectDestroy(): ", cuda_error_name(cuTexObjectDestroy(CUtexObject(0))))
@test (cuTexObjectDestroy(CUtexObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexObjectGetResourceDesc(): ", cuda_error_name(cuTexObjectGetResourceDesc(Ptr{CUDA_RESOURCE_DESC}(C_NULL), CUtexObject(0))))
@test (cuTexObjectGetResourceDesc(Ptr{CUDA_RESOURCE_DESC}(C_NULL), CUtexObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexObjectGetTextureDesc(): ", cuda_error_name(cuTexObjectGetTextureDesc(Ptr{CUDA_TEXTURE_DESC}(C_NULL), CUtexObject(0))))
@test (cuTexObjectGetTextureDesc(Ptr{CUDA_TEXTURE_DESC}(C_NULL), CUtexObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuTexObjectGetResourceViewDesc(): ", cuda_error_name(cuTexObjectGetResourceViewDesc(Ptr{CUDA_RESOURCE_VIEW_DESC}(C_NULL), CUtexObject(0))))
@test (cuTexObjectGetResourceViewDesc(Ptr{CUDA_RESOURCE_VIEW_DESC}(C_NULL), CUtexObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuSurfObjectCreate(): ", cuda_error_name(cuSurfObjectCreate(Ptr{CUsurfObject}(C_NULL), Ptr{CUDA_RESOURCE_DESC}(C_NULL))))
@test (cuSurfObjectCreate(Ptr{CUsurfObject}(C_NULL), Ptr{CUDA_RESOURCE_DESC}(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuSurfObjectDestroy(): ", cuda_error_name(cuSurfObjectDestroy(CUsurfObject(0))))
@test (cuSurfObjectDestroy(CUsurfObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuSurfObjectGetResourceDesc(): ", cuda_error_name(cuSurfObjectGetResourceDesc(Ptr{CUDA_RESOURCE_DESC}(C_NULL), CUsurfObject(0))))
@test (cuSurfObjectGetResourceDesc(Ptr{CUDA_RESOURCE_DESC}(C_NULL), CUsurfObject(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceCanAccessPeer(): ", cuda_error_name(cuDeviceCanAccessPeer(Ptr{Cint}(C_NULL), CUdevice(0), CUdevice(0))))
@test (cuDeviceCanAccessPeer(Ptr{Cint}(C_NULL), CUdevice(0), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxEnablePeerAccess(): ", cuda_error_name(cuCtxEnablePeerAccess(CUcontext(C_NULL), Cuint(0))))
@test (cuCtxEnablePeerAccess(CUcontext(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuCtxDisablePeerAccess(): ", cuda_error_name(cuCtxDisablePeerAccess(CUcontext(C_NULL))))
@test (cuCtxDisablePeerAccess(CUcontext(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuDeviceGetP2PAttribute(): ", cuda_error_name(cuDeviceGetP2PAttribute(Ptr{Cint}(C_NULL), CUdevice_P2PAttribute(0), CUdevice(0), CUdevice(0))))
@test (cuDeviceGetP2PAttribute(Ptr{Cint}(C_NULL), CUdevice_P2PAttribute(0), CUdevice(0), CUdevice(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsUnregisterResource(): ", cuda_error_name(cuGraphicsUnregisterResource(CUgraphicsResource(C_NULL))))
@test (cuGraphicsUnregisterResource(CUgraphicsResource(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsSubResourceGetMappedArray(): ", cuda_error_name(cuGraphicsSubResourceGetMappedArray(Ptr{CUarray}(C_NULL), CUgraphicsResource(C_NULL), Cuint(0), Cuint(0))))
@test (cuGraphicsSubResourceGetMappedArray(Ptr{CUarray}(C_NULL), CUgraphicsResource(C_NULL), Cuint(0), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsResourceGetMappedMipmappedArray(): ", cuda_error_name(cuGraphicsResourceGetMappedMipmappedArray(Ptr{CUmipmappedArray}(C_NULL), CUgraphicsResource(C_NULL))))
@test (cuGraphicsResourceGetMappedMipmappedArray(Ptr{CUmipmappedArray}(C_NULL), CUgraphicsResource(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsResourceGetMappedPointer(): ", cuda_error_name(cuGraphicsResourceGetMappedPointer(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), CUgraphicsResource(C_NULL))))
@test (cuGraphicsResourceGetMappedPointer(Ptr{CUdeviceptr}(C_NULL), Ptr{Csize_t}(C_NULL), CUgraphicsResource(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsResourceSetMapFlags(): ", cuda_error_name(cuGraphicsResourceSetMapFlags(CUgraphicsResource(C_NULL), Cuint(0))))
@test (cuGraphicsResourceSetMapFlags(CUgraphicsResource(C_NULL), Cuint(0)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsMapResources(): ", cuda_error_name(cuGraphicsMapResources(Cuint(0), Ptr{CUgraphicsResource}(C_NULL), CUstream(C_NULL))))
@test (cuGraphicsMapResources(Cuint(0), Ptr{CUgraphicsResource}(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGraphicsUnmapResources(): ", cuda_error_name(cuGraphicsUnmapResources(Cuint(0), Ptr{CUgraphicsResource}(C_NULL), CUstream(C_NULL))))
@test (cuGraphicsUnmapResources(Cuint(0), Ptr{CUgraphicsResource}(C_NULL), CUstream(C_NULL)) == CUDA_ERROR_NOT_INITIALIZED)

println("cuGetExportTable(): ", cuda_error_name(cuGetExportTable(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{CUuuid}(C_NULL))))
@test (cuGetExportTable(Ptr{Ptr{Cvoid}}(C_NULL), Ptr{CUuuid}(C_NULL)) == CUDA_ERROR_INVALID_VALUE)


