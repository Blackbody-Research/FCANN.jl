#=*
* High level CUDA functions
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

function cuGetErrorString(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorString(error, verbose_error)
    @assert (result == CUDA_SUCCESS)
    return unsafe_string(pop!(verbose_error))
end

function cuGetErrorName(error::CUresult)::String
    local verbose_error::Array{Ptr{UInt8}, 1} = [Ptr{UInt8}(C_NULL)]
    local result::CUresult = cuGetErrorName(error, verbose_error)
    @assert (result == CUDA_SUCCESS) ("cuGetErrorName() error: " * cuGetErrorString(result))
    return unsafe_string(pop!(verbose_error))
end

cuInit(Flags::Integer) = cuInit(Cuint(Flags))

function cuDriverGetVersion()::Cint
    local driver_version::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDriverGetVersion(driver_version)
    @assert (result == CUDA_SUCCESS) ("cuDriverGetVersion() error: " * cuGetErrorString(result))
    return pop!(driver_version)
end

function cuDeviceGet(ordinal::Cint)::CUdevice
    local cuda_device::Array{CUdevice, 1} = zeros(CUdevice, 1)
    local result::CUresult = cuDeviceGet(Base.unsafe_convert(Ptr{CUdevice}, cuda_device), ordinal)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGet() error: " * cuGetErrorString(result))
    return pop!(cuda_device)
end

cuDeviceGet(ordinal::Integer) = cuDeviceGet(Cint(ordinal))

function cuDeviceGetCount()::Cint
    local device_count::Array{CUdevice, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceGetCount(device_count)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetCount() error: " * cuGetErrorString(result))
    return pop!(device_count)
end

function cuDeviceGetName(dev::CUdevice)::String
    # allow up to 50 characters for returned device name
    local device_name::Array{UInt8, 1} = zeros(UInt8, 40)
    local result::CUresult = cuDeviceGetName(Base.unsafe_convert(Ptr{UInt8}, device_name), Cint(40), dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetName() error: " * cuGetErrorString(result))
    return unsafe_string(Base.unsafe_convert(Ptr{UInt8}, device_name))
end

function cuDeviceTotalMem(dev::CUdevice)::Csize_t
    local bytes::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuDeviceTotalMem(bytes, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceTotalMem() error: " * cuGetErrorString(result))
    return pop!(bytes)
end

function cuDeviceGetAttribute(attrib::CUdevice_attribute, dev::CUdevice)::Cint
    local cint_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceGetAttribute(cint_array, attrib, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetAttribute() error: " * cuGetErrorString(result))
    return pop!(cint_array)
end

function cuDeviceGetProperties(dev::CUdevice)::CUdevprop
    local prop_array::Array{CUdevprop, 1} = zeros(CUdevprop, 1)
    local result::CUresult = cuDeviceGetProperties(prop_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetProperties() error: " * cuGetErrorString(result))
    return pop!(prop_array)
end

function cuDeviceComputeCapability(dev::CUdevice)::Tuple{Cint, Cint}
    local major_array::Array{Cint, 1} = zeros(Cint, 1)
    local minor_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDeviceComputeCapability(major_array, minor_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceComputeCapability() error: " * cuGetErrorString(result))
    return (pop!(major_array), pop!(minor_array))
end

# primary context management functions
function cuDevicePrimaryCtxRetain(dev::CUdevice)::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuDevicePrimaryCtxRetain(ctx_array, dev)
    @assert (result == CUDA_SUCCESS) ("cuDevicePrimaryCtxRetain() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuDevicePrimaryCtxGetState(dev::CUdevice)::Tuple{CUctx_flags, Cint}
    local flags_array::Array{CUctx_flags, 1} = zeros(CUctx_flags, 1)
    local active_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuDevicePrimaryCtxGetState(dev, flags_array, active_array)
    @assert (result == CUDA_SUCCESS) ("cuDevicePrimaryCtxGetState() error: " * cuGetErrorString(result))
    return (pop!(flags_array), pop!(active_array))
end

# context management functions
function cuCtxCreate(flags::CUctx_flags, dev::CUdevice)::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxCreate(ctx_array, flags, dev)
    @assert (result == CUDA_SUCCESS) ("cuCtxCreate() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxPopCurrent()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxPopCurrent(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxPopCurrent() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetCurrent()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    local result::CUresult = cuCtxGetCurrent(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetCurrent() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetDevice()::CUdevice
    local dev_array::Array{CUdevice, 1} = [C_NULL]
    local result::CUresult = cuCtxGetDevice(dev_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetDevice() error: " * cuGetErrorString(result))
    return pop!(dev_array)
end

function cuCtxGetFlags()::CUctx_flags
    local ctx_array::Array{CUctx_flags, 1} = zeros(CUctx_flags, 1)
    local result::CUresult = cuCtxGetFlags(ctx_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetFlags() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

function cuCtxGetLimit(limit::CUlimit)::CUlimit
    local limit_value_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuCtxGetLimit(limit_value_array, limit)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetLimit() error: " * cuGetErrorString(result))
    return pop!(limit_value_array)
end

function cuCtxGetCacheConfig()::CUfunc_cache
    local func_cache_array::Array{CUfunc_cache, 1} = zeros(CUfunc_cache, 1)
    local result::CUresult = cuCtxGetCacheConfig(func_cache_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetCacheConfig() error: " * cuGetErrorString(result))
    return pop!(func_cache_array)
end

function cuCtxGetSharedMemConfig()::CUsharedconfig
    local shared_cfg_array::Array{CUsharedconfig, 1} = zeros(CUsharedconfig, 1)
    local result::CUresult = cuCtxGetSharedMemConfig(shared_cfg_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetSharedMemConfig() error: " * cuGetErrorString(result))
    return pop!(shared_cfg_array)
end

function cuCtxGetApiVersion(ctx::CUcontext)::Cuint
    local version_array::Array{Cuint, 1} = zeros(Cuint, 1)
    local result::CUresult = cuCtxGetApiVersion(ctx, version_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetApiVersion() error: " * cuGetErrorString(result))
    return pop!(version_array)
end

function cuCtxGetStreamPriorityRange()::Tuple{Cint, Cint}
    local lp_array::Array{Cint, 1} = zeros(Cint, 1)
    local gp_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuCtxGetStreamPriorityRange(lp_array, gp_array)
    @assert (result == CUDA_SUCCESS) ("cuCtxGetStreamPriorityRange() error: " * cuGetErrorString(result))
    return (pop!(lp_array), pop!(gp_array))
end

function cuCtxAttach()::CUcontext
    local ctx_array::Array{CUcontext, 1} = [C_NULL]
    # context attach flags argument must be 0
    local result::CUresult = cuCtxAttach(ctx_array, Cuint(0))
    @assert (result == CUDA_SUCCESS) ("cuCtxAttach() error: " * cuGetErrorString(result))
    return pop!(ctx_array)
end

# module management functions
function cuModuleLoad(fname::Ptr{UInt8})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoad(Base.unsafe_convert(Ptr{CUmodule}, module_array), fname)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoad() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoad(fname::Array{UInt8, 1}) = cuModuleLoad(Base.unsafe_convert(Ptr{UInt8}, fname))
cuModuleLoad(fname::String) = cuModuleLoad(Base.unsafe_convert(Ptr{UInt8}, map(UInt8, collect(fname))))

function cuModuleLoadData(image::Ptr{Nothing})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadData(module_array, image)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadData() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoadData(image::Ptr{UInt8}) = cuModuleLoadData(Ptr{Nothing}(image))
cuModuleLoadData(image::Array{UInt8, 1}) = cuModuleLoadData(Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt8}, image)))
cuModuleLoadData(image::String) = cuModuleLoadData(map(UInt8, collect(image)))

function cuModuleLoadDataEx(image::Ptr{Nothing}, numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadDataEx(module_array, image, numOptions, options, optionValues)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadDataEx() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

function cuModuleLoadDataEx(image::Ptr{Nothing}, numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadDataEx(Base.unsafe_convert(Ptr{CUmodule}, module_array), image, numOptions, options, optionValues)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadDataEx() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

function cuModuleLoadFatBinary(fatCubin::Ptr{Nothing})::CUmodule
    local module_array::Array{CUmodule, 1} = [C_NULL]
    local result::CUresult = cuModuleLoadFatBinary(module_array, fatCubin)
    @assert (result == CUDA_SUCCESS) ("cuModuleLoadFatBinary() error: " * cuGetErrorString(result))
    return pop!(module_array)
end

cuModuleLoadFatBinary(fatCubin::Ptr{UInt8}) = cuModuleLoadFatBinary(Ptr{Nothing}(fatCubin))
cuModuleLoadFatBinary(fatCubin::Array{UInt8, 1}) = cuModuleLoadFatBinary(Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt8}, fatCubin)))
cuModuleLoadFatBinary(fatCubin::String) = cuModuleLoadFatBinary(map(UInt8, collect(fatCubin)))

function cuModuleGetFunction(hmod::CUmodule, name::Ptr{UInt8})::CUfunction
    local function_array::Array{CUfunction, 1} = [C_NULL]
    local result::CUresult = cuModuleGetFunction(Base.unsafe_convert(Ptr{CUfunction}, function_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetFunction() error: " * cuGetErrorString(result))
    return pop!(function_array)
end

cuModuleGetFunction(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetFunction(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetFunction(hmod::CUmodule, name::String) = cuModuleGetFunction(hmod, map(UInt8, collect(name)))

function cuModuleGetGlobal(hmod::CUmodule, name::Ptr{UInt8})::Tuple{CUdeviceptr, Csize_t}
    local dptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local bytes_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuModuleGetGlobal(Base.unsafe_convert(Ptr{CUdeviceptr}, dptr_array), bytes_array, hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetGlobal() error: " * cuGetErrorString(result))
    return (pop!(dptr_array), pop!(bytes_array))
end

cuModuleGetGlobal(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetGlobal(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetGlobal(hmod::CUmodule, name::String) = cuModuleGetGlobal(hmod, map(UInt8, collect(name)))

function cuModuleGetTexRef(hmod::CUmodule, name::Ptr{UInt8})::CUtexref
    local texref_array::Array{CUtexref, 1} = [C_NULL]
    local result::CUresult = cuModuleGetTexRef(Base.unsafe_convert(Ptr{CUtexref}, texref_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetTexRef() error: " * cuGetErrorString(result))
    return pop!(texref_array)
end

cuModuleGetTexRef(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetTexRef(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetTexRef(hmod::CUmodule, name::String) = cuModuleGetTexRef(hmod, map(UInt8, collect(name)))

function cuModuleGetSurfRef(hmod::CUmodule, name::Ptr{UInt8})::CUsurfref
    local surfref_array::Array{CUsurfref, 1} = [C_NULL]
    local result::CUresult = cuModuleGetSurfRef(Base.unsafe_convert(Ptr{CUsurfref}, surfref_array), hmod, name)
    @assert (result == CUDA_SUCCESS) ("cuModuleGetSurfRef() error: " * cuGetErrorString(result))
    return pop!(surfref_array)
end

cuModuleGetSurfRef(hmod::CUmodule, name::Array{UInt8, 1}) = cuModuleGetSurfRef(hmod, Base.unsafe_convert(Ptr{UInt8}, name))
cuModuleGetSurfRef(hmod::CUmodule, name::String) = cuModuleGetSurfRef(hmod, map(UInt8, collect(name)))

function cuLinkCreate(numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{T, 1})::CUlinkState where T
    local state_array::Array{CUlinkState, 1} = [C_NULL]
    local result::CUresult = cuLinkCreate(numOptions, Base.unsafe_convert(Ptr{CUjit_option}, options), Ptr{Ptr{Nothing}}(Base.unsafe_convert(Ptr{Nothing}, optionValues)), Base.unsafe_convert(Ptr{CUlinkState}, state_array))
    @assert (result == CUDA_SUCCESS) ("cuLinkCreate() error: " * cuGetErrorString(result))
    return pop!(state_array)
end

cuLinkCreate(numOptions::Integer, options::Array{CUjit_option, 1}, optionValues::Array) = cuLinkCreate(Cuint(numOptions), options, optionValues)

function cuLinkAddData(state::CUlinkState, jitype::CUjitInputType, data::Array{UInt8, 1}, name::Array{UInt8, 1})::Nothing
    local result::CUresult = cuLinkAddData(state, jitype, Base.unsafe_convert(Ptr{Nothing}, data), Csize_t(sizeof(data) + 1), Base.unsafe_convert(Ptr{UInt8}, name), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Nothing}}(C_NULL))
    @assert (result == CUDA_SUCCESS) ("cuLinkAddData() error: " * cuGetErrorString(result))
    return nothing
end

cuLinkAddData(state::CUlinkState, jitype::CUjitInputType, data::String, name::String) = cuLinkAddData(state, jitype, map(UInt8, collect(data)), map(UInt8, collect(name)))

function cuLinkAddFile(state::CUlinkState, jitype::CUjitInputType, path::Array{UInt8, 1})::Nothing
    local result::CUresult = cuLinkAddFile(state, jitype, Base.unsafe_convert(Ptr{UInt8}, path), Cuint(0), Ptr{CUjit_option}(C_NULL), Ptr{Ptr{Nothing}}(C_NULL))
    @assert (result == CUDA_SUCCESS) ("cuLinkAddFile() error: " * cuGetErrorString(result))
    return nothing
end

cuLinkAddFile(state::CUlinkState, jitype::CUjitInputType, path::String) = cuLinkAddFile(state, jitype, map(UInt8, collect(path)))

function cuLinkComplete(state::CUlinkState)::Array{UInt8, 1}
    local size_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local cubin_ptr_array::Array{Ptr{Nothing}, 1} = [C_NULL]
    local result::CUresult = cuLinkComplete(state, cubin_ptr_array, size_array)
    @assert (result == CUDA_SUCCESS) ("cuLinkComplete() error: " * cuGetErrorString(result))
    # copy cubin image from C pointer to julia array
    local cubin_data::Array{UInt8, 1} = zeros(UInt8, size_array[1])
    ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt8}, cubin_data)),
            cubin_ptr_array[1],
            size_array[1])
    return cubin_data
end

# memory management functions
function cuMemGetInfo()::Tuple{Csize_t, Csize_t}
    local free_size::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local total_size::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuMemGetInfo(free_size, total_size)
    @assert (result == CUDA_SUCCESS) ("cuMemGetInfo() error: " * cuGetErrorString(result))
    return (pop!(free_size), pop!(total_size))
end

function cuMemAlloc(bytesize::Csize_t)::CUdeviceptr
    local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuMemAlloc(device_ptr_array, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemAlloc() error: " * cuGetErrorString(result))
    return pop!(device_ptr_array)
end

cuMemAlloc(bytesize::Integer) = cuMemAlloc(Csize_t(bytesize))

function cuMemAllocPitch(WidthInBytes::Csize_t, Height::Csize_t, ElementSizeBytes::Cuint)::Tuple{CUdeviceptr, Csize_t}
    local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local pitch_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuMemAllocPitch(dev_ptr_array, pitch_array, WidthInBytes, Height, ElementSizeBytes)
    @assert (result == CUDA_SUCCESS) ("cuMemAllocPitch() error: " * cuGetErrorString(result))
    return (pop!(dev_ptr_array), pop!(pitch_array))
end

cuMemAllocPitch(WidthInBytes::Integer, Height::Integer, ElementSizeBytes::Integer) = cuMemAllocPitch(Csize_t(WidthInBytes), Csize_t(Height), Cuint(ElementSizeBytes))

function cuMemGetAddressRange(dptr::CUdeviceptr)::Tuple{CUdeviceptr, Csize_t}
    local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local size_array::Array{Csize_t, 1} = zeros(Csize_t, 1)
    local result::CUresult = cuMemGetAddressRange(dev_ptr_array, size_array, dptr)
    @assert (result == CUDA_SUCCESS) ("cuMemGetAddressRange() error: " * cuGetErrorString(result))
    return (pop!(dev_ptr_array), pop!(size_array))
end

function cuMemAllocHost(bytesize::Csize_t)::Ptr{Nothing}
    local host_ptr_array::Array{Ptr{Nothing}, 1} = [C_NULL]
    local result::CUresult = cuMemAllocHost(host_ptr_array, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemAllocHost() error: " * cuGetErrorString(result))
    return pop!(host_ptr_array)
end

cuMemAllocHost(bytesize::Integer) = cuMemAllocHost(Csize_t(bytesize))

function cuMemHostAlloc(bytesize::Csize_t)::Ptr{Nothing}
    local host_ptr_array::Array{Ptr{Nothing}, 1} = [C_NULL]
    local result::CUresult = cuMemHostAlloc(host_ptr_array, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemHostAlloc() error: " * cuGetErrorString(result))
    return pop!(host_ptr_array)
end

cuMemHostAlloc(bytesize::Integer) = cuMemHostAlloc(Csize_t(bytesize))

function cuMemHostGetDevicePointer(p::Ptr{Nothing})::CUdeviceptr
    local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuMemHostGetDevicePointer(dev_ptr_array, p, Cuint(0))
    @assert (result == CUDA_SUCCESS) ("cuMemHostGetDevicePointer() error: " * cuGetErrorString(result))
    return pop!(dev_ptr_array)
end

function cuMemHostGetFlags(p::Ptr{Nothing})::Cuint
    local flags_array::Array{Cuint, 1} = zeros(Cuint, 1)
    local result::CUresult = cuMemHostGetFlags(flags_array, p)
    @assert (result == CUDA_SUCCESS) ("cuMemHostGetFlags() error: " * cuGetErrorString(result))
    return pop!(flags_array)
end

function cuMemAllocManaged(bytesize::Csize_t, flags::CUmemAttach_flags)::CUdeviceptr
    local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuMemAllocManaged(dev_ptr_array, bytesize, flags)
    @assert (result == CUDA_SUCCESS) ("cuMemAllocManaged() error: " * cuGetErrorString(result))
    return pop!(dev_ptr_array)
end

cuMemAllocManaged(bytesize::Integer, flags::CUmemAttach_flags) = cuMemAllocManaged(Csize_t(bytesize), flags)

function cuDeviceGetByPCIBusId(pciBusId::Array{UInt8, 1})::CUdevice
    local dev_array::Array{CUdevice, 1} = zeros(CUdevice, 1)
    local result::CUresult = cuDeviceGetByPCIBusId(dev_array, pciBusId)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetByPCIBusId() error: " * cuGetErrorString(result))
    return pop!(dev_array)
end

cuDeviceGetByPCIBusId(pciBusId::String) = cuDeviceGetByPCIBusId(map(UInt8, collect(pciBusId)))

function cuDeviceGetPCIBusId(dev::CUdevice)::String
    local pcibus_array::Array{UInt8, 1} = zeros(UInt8, 100)
    local result::CUresult = cuDeviceGetPCIBusId(pcibus_array, Cint(100), dev)
    @assert (result == CUDA_SUCCESS) ("cuDeviceGetPCIBusId() error: " * cuGetErrorString(result))
    return unsafe_string(Base.unsafe_convert(Ptr{UInt8}, pcibus_array))
end

function cuIpcGetEventHandle(event::CUevent)::CUipcEventHandle
    local handle_array::Array{CUipcEventHandle, 1} = zeros(CUipcEventHandle, 1)
    local result::CUresult = cuIpcGetEventHandle(handle_array, event)
    @assert (result == CUDA_SUCCESS) ("cuIpcGetEventHandle() error: " * cuGetErrorString(result))
    return pop!(handle_array)
end

function cuIpcOpenEventHandle(handle::CUipcEventHandle)::CUevent
    local event_array::Array{CUevent, 1} = [C_NULL]
    local result::CUresult = cuIpcOpenEventHandle(event_array, handle)
    @assert (result == CUDA_SUCCESS) ("cuIpcOpenEventHandle() error: " * cuGetErrorString(result))
    return pop!(event_array)
end

function cuIpcGetMemHandle(dptr::CUdeviceptr)::CUipcMemHandle
    local handle_array::Array{CUipcMemHandle, 1} = zeros(CUipcMemHandle, 1)
    local result::CUresult = cuIpcGetMemHandle(handle_array, dptr)
    @assert (result == CUDA_SUCCESS) ("cuIpcGetMemHandle() error: " * cuGetErrorString(result))
    return pop!(handle_array)
end

function cuIpcOpenMemHandle(handle::CUipcMemHandle, Flags::CUipcMem_flags)::CUdeviceptr
    local device_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
    local result::CUresult = cuIpcOpenMemHandle(device_ptr_array, handle, Flags)
    @assert (result == CUDA_SUCCESS) ("cuIpcOpenMemHandle() error: " * cuGetErrorString(result))
    return pop!(device_ptr_array)
end

function cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcHost::Array{T}, srcOffset::Csize_t, bytesize::Csize_t)::Nothing where T
    local result::CUresult = cuMemcpyHtoD(dstDevice + dstOffset, Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, srcHost)) + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyHtoD() error: " * cuGetErrorString(result))
    nothing
end

function cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcHost::Ptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyHtoD(dstDevice + dstOffset, srcHost + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyHtoD() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcHost::Array, srcOffset::Integer, bytesize::Integer) = cuMemcpyHtoD(dstDevice, Csize_t(dstOffset), srcHost, Csize_t(srcOffset), Csize_t(bytesize))
cuMemcpyHtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcHost::Ptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyHtoD(dstDevice, Csize_t(dstOffset), srcHost, Csize_t(srcOffset), Csize_t(bytesize))

function cuMemcpyDtoH(dstHost::Array{T}, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing where T
    local result::CUresult = cuMemcpyDtoH(Ptr{Nothing}(Base.unsafe_convert(Ptr{T}, dstHost)) + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoH() error: " * cuGetErrorString(result))
    nothing
end

function cuMemcpyDtoH(dstHost::Ptr, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyDtoH(dstHost + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoH() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyDtoH(dstHost::Array, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoH(dstHost, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))
cuMemcpyDtoH(dstHost::Ptr, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoH(dstHost, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))

function cuMemcpyDtoD(dstDevice::CUdeviceptr, dstOffset::Csize_t, srcDevice::CUdeviceptr, srcOffset::Csize_t, bytesize::Csize_t)::Nothing
    local result::CUresult = cuMemcpyDtoD(dstDevice + dstOffset, srcDevice + srcOffset, bytesize)
    @assert (result == CUDA_SUCCESS) ("cuMemcpyDtoD() error: " * cuGetErrorString(result))
    nothing
end

cuMemcpyDtoD(dstDevice::CUdeviceptr, dstOffset::Integer, srcDevice::CUdeviceptr, srcOffset::Integer, bytesize::Integer) = cuMemcpyDtoD(dstDevice, Csize_t(dstOffset), srcDevice, Csize_t(srcOffset), Csize_t(bytesize))

function cuArrayCreate(pAllocateArray::CUDA_ARRAY_DESCRIPTOR)::CUarray
    local handle_array::Array{CUarray, 1} = [C_NULL]
    local allocatearray_array::Array{CUDA_ARRAY_DESCRIPTOR, 1} = [pAllocateArray]
    local result::CUresult = cuArrayCreate(handle_array, allocatearray_array)
    @assert (result == CUDA_SUCCESS) ("cuArrayCreate() error: " * cuGetErrorString(result))
    return pop!(handle_array)
end

function cuArrayGetDescriptor(hArray::CUarray)::CUDA_ARRAY_DESCRIPTOR
    local array_descriptor_array::Array{CUDA_ARRAY_DESCRIPTOR, 1} = zeros(CUDA_ARRAY_DESCRIPTOR, 1)
    local result::CUresult = cuArrayGetDescriptor(array_descriptor_array, hArray)
    @assert (result == CUDA_SUCCESS) ("cuArrayGetDescriptor() error: " * cuGetErrorString(result))
    return pop!(array_descriptor_array)
end

function cuArray3DCreate(pAllocateArray::CUDA_ARRAY3D_DESCRIPTOR)::CUarray
    local handle_array::Array{CUarray, 1} = [C_NULL]
    local allocatearray_array::Array{CUDA_ARRAY3D_DESCRIPTOR, 1} = [pAllocateArray]
    local result::CUresult = cuArray3DCreate(handle_array, allocatearray_array)
    @assert (result == CUDA_SUCCESS) ("cuArray3DCreate() error: " * cuGetErrorString(result))
    return pop!(handle_array)
end

function cuArray3DGetDescriptor(hArray::CUarray)::CUDA_ARRAY3D_DESCRIPTOR
    local array3d_descriptor_array::Array{CUDA_ARRAY3D_DESCRIPTOR, 1} = zeros(CUDA_ARRAY3D_DESCRIPTOR, 1)
    local result::CUresult = cuArray3DGetDescriptor(array3d_descriptor_array, hArray)
    @assert (result == CUDA_SUCCESS) ("cuArray3DGetDescriptor() error: " * cuGetErrorString(result))
    return pop!(array3d_descriptor_array)
end

function cuMipmappedArrayCreate(pMipmappedArrayDesc::CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels::Cuint)::CUmipmappedArray
    local mipmapped_array::Array{CUmipmappedArray, 1} = [C_NULL]
    local mm_array_desc::Array{CUDA_ARRAY3D_DESCRIPTOR, 1} = [pMipmappedArrayDesc]
    local result::CUresult = cuMipmappedArrayCreate(mipmapped_array, mm_array_desc, numMipmapLevels)
    @assert (result == CUDA_SUCCESS) ("cuMipmappedArrayCreate() error: " * cuGetErrorString(result))
    return pop!(mipmapped_array)
end

cuMipmappedArrayCreate(pMipmappedArrayDesc::CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels::Integer) = cuMipmappedArrayCreate(pMipmappedArrayDesc, Cuint(numMipmapLevels))

function cuMipmappedArrayGetLevel(hMipmappedArray::CUmipmappedArray, level::Cuint)::CUarray
    local mm_level_array::Array{CUarray, 1} = [C_NULL]
    local result::CUresult = cuMipmappedArrayGetLevel(mm_level_array, hMipmappedArray, level)
    @assert (result == CUDA_SUCCESS) ("cuMipmappedArrayGetLevel() error: " * cuGetErrorString(result))
    return pop!(mm_level_array)
end

cuMipmappedArrayGetLevel(hMipmappedArray::CUmipmappedArray, level::Integer) = cuMipmappedArrayGetLevel(hMipmappedArray, Cuint(level))

# unified addressing functions
if (CUDA_VERSION >= 9020)
    # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL available since CUDA 9.2
    function cuPointerGetAttribute(attribute::CUpointer_attribute, ptr::CUdeviceptr)
        local result::CUresult
        if (attribute == CU_POINTER_ATTRIBUTE_CONTEXT)
            local ctx_array::Array{CUcontext, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, ctx_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(ctx_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
            local memtype_array::Array{Cuint, 1} = zeros(Cuint, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, memtype_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(memtype_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
            local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, dev_ptr_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(dev_ptr_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER)
            local host_ptr_array::Array{Ptr{Nothing}, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, host_ptr_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(host_ptr_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_P2P_TOKENS)
            local p2p_tokens_array::Array{CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1} = zeros(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, p2p_tokens_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(p2p_tokens_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)
            local syncmem_array::Array{Bool, 1} = zeros(Bool, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, syncmem_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(syncmem_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_BUFFER_ID)
            local bufferid_array::Array{Culonglong, 1} = zeros(Culonglong, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, bufferid_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(bufferid_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED)
            local is_managed_array::Array{Bool, 1} = zeros(Bool, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, is_managed_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(is_managed_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
            local dev_ordinal_array::Array{CUdevice, 1} = zeros(CUdevice, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, dev_ordinal_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(dev_ordinal_array)
        else
            error("cuPointerGetAttribute(): error: Found unexpected CUpointer_attribute value, ", Int(attribute), ".")
        end
    end
else
    function cuPointerGetAttribute(attribute::CUpointer_attribute, ptr::CUdeviceptr)
        local result::CUresult
        if (attribute == CU_POINTER_ATTRIBUTE_CONTEXT)
            local ctx_array::Array{CUcontext, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, ctx_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(ctx_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
            local memtype_array::Array{Cuint, 1} = zeros(Cuint, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, memtype_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(memtype_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
            local dev_ptr_array::Array{CUdeviceptr, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, dev_ptr_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(dev_ptr_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER)
            local host_ptr_array::Array{Ptr{Nothing}, 1} = [C_NULL]
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, host_ptr_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(host_ptr_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_P2P_TOKENS)
            local p2p_tokens_array::Array{CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1} = zeros(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, p2p_tokens_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(p2p_tokens_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)
            local syncmem_array::Array{Bool, 1} = zeros(Bool, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, syncmem_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(syncmem_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_BUFFER_ID)
            local bufferid_array::Array{Culonglong, 1} = zeros(Culonglong, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, bufferid_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(bufferid_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED)
            local is_managed_array::Array{Bool, 1} = zeros(Bool, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, is_managed_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(is_managed_array)
        elseif (attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
            local dev_ordinal_array::Array{CUdevice, 1} = zeros(CUdevice, 1)
            result = cuPointerGetAttribute(Base.unsafe_convert(Ptr{Nothing}, dev_ordinal_array), attribute, ptr)
            @assert (result == CUDA_SUCCESS) ("cuPointerGetAttribute() error: " * cuGetErrorString(result))
            return pop!(dev_ordinal_array)
        else
            error("cuPointerGetAttribute(): error: Found unexpected CUpointer_attribute value, ", Int(attribute), ".")
        end
    end
end

function cuMemRangeGetAttribute(dataSize::Csize_t, attribute::CUmem_range_attribute, devPtr::CUdeviceptr, count::Csize_t)
    local result::CUresult

    if (attribute == CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY)
        @assert (dataSize == 4) ("cuMemRangeGetAttribute() error: 'dataSize' must be 4 when CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY is specified!")
        local read_mostly_array::Array{Cint, 1} = zeros(Cint, 1)
        result = cuMemRangeGetAttribute(Ptr{Nothing}(Base.unsafe_convert(Ptr{Cint}, read_mostly_array)), dataSize, attribute, devPtr, count)
        @assert (result == CUDA_SUCCESS) ("cuMemRangeGetAttribute() error: " * cuGetErrorString(result))
        return pop!(read_mostly_array)
    elseif (attribute == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION)
        @assert (dataSize == 4) ("cuMemRangeGetAttribute() error: 'dataSize' must be 4 when CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION is specified!")
        local preferred_location_array::Array{Cint, 1} = zeros(Cint, 1)
        result = cuMemRangeGetAttribute(Ptr{Nothing}(Base.unsafe_convert(Ptr{Cint}, preferred_location_array)), dataSize, attribute, devPtr, count)
        @assert (result == CUDA_SUCCESS) ("cuMemRangeGetAttribute() error: " * cuGetErrorString(result))
        return pop!(preferred_location_array)
    elseif (attribute == CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY)
        @assert ((dataSize > 0) && (dataSize % 4 == 0)) ("cuMemRangeGetAttribute() error: 'dataSize' must be a non-zero multiple of 4 when CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY is specified!")
        local accessed_by_array::Array{Cint, 1} = zeros(Cint, Int(dataSize * 0.25))
        result = cuMemRangeGetAttribute(Ptr{Nothing}(Base.unsafe_convert(Ptr{Cint}, accessed_by_array)), dataSize, attribute, devPtr, count)
        @assert (result == CUDA_SUCCESS) ("cuMemRangeGetAttribute() error: " * cuGetErrorString(result))
        return pop!(accessed_by_array)
    elseif (attribute == CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION)
        @assert (dataSize == 4) ("cuMemRangeGetAttribute() error: 'dataSize' must be 4 when CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION is specified!")
        local last_prefetch_array::Array{Cint, 1} = zeros(Cint, 1)
        result = cuMemRangeGetAttribute(Ptr{Nothing}(Base.unsafe_convert(Ptr{Cint}, last_prefetch_array)), dataSize, attribute, devPtr, count)
        @assert (result == CUDA_SUCCESS) ("cuMemRangeGetAttribute() error: " * cuGetErrorString(result))
        return pop!(last_prefetch_array)
    else
        error("cuMemRangeGetAttribute(): error: Found unexpected CUmem_range_attribute value, ", Int(attribute), ".")
    end
end

cuMemRangeGetAttribute(dataSize::Integer, attribute::CUmem_range_attribute, devPtr::CUdeviceptr, count::Integer) = cuMemRangeGetAttribute(Csize_t(dataSize), attribute, devPtr, Csize_t(count))

function cuMemRangeGetAttributes(dataSizes::Array{Csize_t, 1}, attributes::Array{CUmem_range_attribute, 1}, numAttributes::Csize_t, devPtr::CUdeviceptr, count::Csize_t)::Array{Array{Cint, 1}, 1}
    @assert (length(dataSizes) == length(attributes)) ("cuMemRangeGetAttributes() error: The length of 'dataSizes' and 'attributes' don't match!")
    local attributes_array::Array{Array{Cint, 1}, 1} = Array{Array{Cint, 1}, 1}()
    
    for (s, a) in zip(dataSizes, attributes)
        if (a == CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY)
            @assert (s == 4) ("cuMemRangeGetAttributes() error: The data size must be 4 when CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY is specified, got " * string(s) * "!")
            push!(attributes_array, zeros(Cint, 1))
        elseif (a == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION)
            @assert (s == 4) ("cuMemRangeGetAttributes() error: The data size must be 4 when CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION is specified, got " * string(s) * "!")
            push!(attributes_array, zeros(Cint, 1))
        elseif (a == CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY)
            @assert ((s > 0) && (s % 4 == 0)) ("cuMemRangeGetAttributes() error: The data size must be a non-zero multiple of 4 when CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY is specified, got " * string(s) * "!")
            push!(attributes_array, zeros(Cint, Int(a * 0.25)))
        elseif (a == CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION)
            @assert (s == 4) ("cuMemRangeGetAttributes() error: The data size must be 4 when CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION is specified, got " * string(s) * "!")
            push!(attributes_array, zeros(Cint, 1))
        else
            error("cuMemRangeGetAttributes(): error: Found unexpected CUmem_range_attribute value, ", Int(a), ".")
        end
    end

    local attributes_ptr_array::Array{Ptr{Nothing}, 1} = map(pointer, attributes_array)

    local result::CUresult = cuMemRangeGetAttributes(attributes_ptr_array, dataSizes, attributes, numAttributes, devPtr, count)
    @assert (result == CUDA_SUCCESS) ("cuMemRangeGetAttributes() error: " * cuGetErrorString(result))
    return attributes_array
end

cuMemRangeGetAttributes(dataSizes::Array{Csize_t, 1}, attributes::Array{CUmem_range_attribute, 1}, numAttributes::Integer, devPtr::CUdeviceptr, count::Integer) = cuMemRangeGetAttributes(dataSizes, attributes, Csize_t(numAttributes), devPtr, Csize_t(count))

if (CUDA_VERSION >= 9020)
    # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL available since CUDA 9.2
    function cuPointerGetAttributes(numAttributes::Cuint, attributes::Array{CUpointer_attribute, 1}, ptr::CUdeviceptr)::Array{Vector, 1}
        local data_array::Array{Vector, 1} = Array{Vector, 1}()
    
        for a in attributes
            if (a == CU_POINTER_ATTRIBUTE_CONTEXT)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
                push!(data_array, zeros(Cuint, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_HOST_POINTER)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_P2P_TOKENS)
                push!(data_array, zeros(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)
                push!(data_array, zeros(Bool, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_BUFFER_ID)
                push!(data_array, zeros(Culonglong, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_IS_MANAGED)
                push!(data_array, zeros(Bool, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
                push!(data_array, zeros(CUdevice, 1))
            else
                error("cuPointerGetAttributes(): error: Found unexpected CUpointer_attribute value, ", Int(a), ".")
            end
        end
    
        local data_ptr_array::Array{Ptr{Nothing}, 1} = map(pointer, data_array)
        local result::CUresult = cuPointerGetAttributes(numAttributes, attributes, data_ptr_array, ptr)
        @assert (result == CUDA_SUCCESS) ("cuPointerGetAttributes() error: " * cuGetErrorString(result))
        return data_array
    end
else
    function cuPointerGetAttributes(numAttributes::Cuint, attributes::Array{CUpointer_attribute, 1}, ptr::CUdeviceptr)::Array{Vector, 1}
        local data_array::Array{Vector, 1} = Array{Vector, 1}()
    
        for a in attributes
            if (a == CU_POINTER_ATTRIBUTE_CONTEXT)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
                push!(data_array, zeros(Cuint, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_DEVICE_POINTER)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_HOST_POINTER)
                push!(data_array, [C_NULL])
            elseif (a == CU_POINTER_ATTRIBUTE_P2P_TOKENS)
                push!(data_array, zeros(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS)
                push!(data_array, zeros(Bool, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_BUFFER_ID)
                push!(data_array, zeros(Culonglong, 1))
            elseif (a == CU_POINTER_ATTRIBUTE_IS_MANAGED)
                push!(data_array, zeros(Bool, 1))
            else
                error("cuPointerGetAttributes(): error: Found unexpected CUpointer_attribute value, ", Int(a), ".")
            end
        end
    
        local data_ptr_array::Array{Ptr{Nothing}, 1} = map(pointer, data_array)
        local result::CUresult = cuPointerGetAttributes(numAttributes, attributes, data_ptr_array, ptr)
        @assert (result == CUDA_SUCCESS) ("cuPointerGetAttributes() error: " * cuGetErrorString(result))
        return data_array
    end
end

cuPointerGetAttributes(numAttributes::Integer, attributes::Array{CUpointer_attribute, 1}, ptr::CUdeviceptr) = cuPointerGetAttributes(Cuint(numAttributes), attributes, ptr)

# stream management functions
function cuStreamCreate(Flags::CUstream_flags)::CUstream
    local stream_array::Array{CUstream, 1} = [C_NULL]
    local result::CUresult = cuStreamCreate(stream_array, Flags)
    @assert (result == CUDA_SUCCESS) ("cuStreamCreate() error: " * cuGetErrorString(result))
    return pop!(stream_array)
end

function cuStreamCreateWithPriority(Flags::CUstream_flags, priority::Cint)::CUstream
    local stream_array::Array{CUstream, 1} = [C_NULL]
    local result::CUresult = cuStreamCreateWithPriority(stream_array, Flags, priority)
    @assert (result == CUDA_SUCCESS) ("cuStreamCreateWithPriority() error: " * cuGetErrorString(result))
    return pop!(stream_array)
end

cuStreamCreateWithPriority(Flags::CUstream_flags, priority::Integer) = cuStreamCreateWithPriority(Flags, Cint(priority))

function cuStreamGetPriority(hStream::CUstream)::Cint
    local priority_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuStreamGetPriority(hStream, priority_array)
    @assert (result == CUDA_SUCCESS) ("cuStreamGetPriority() error: " * cuGetErrorString(result))
    return pop!(priority_array)
end

function cuStreamGetFlags(hStream::CUstream)::CUstream_flags
    local flags_array::Array{CUstream_flags, 1} = zeros(CUstream_flags, 1)
    local result::CUresult = cuStreamGetFlags(hStream, flags_array)
    @assert (result == CUDA_SUCCESS) ("cuStreamGetFlags() error: " * cuGetErrorString(result))
    return pop!(flags_array)
end

if (CUDA_VERSION >= 9020)
    # cuStreamGetCtx() available since CUDA 9.2
    function cuStreamGetCtx(hStream::CUstream)::CUcontext
        local context_array::Array{CUcontext, 1} = [C_NULL]
        local result::CUresult = cuStreamGetCtx(hStream, context_array)
        @assert (result == CUDA_SUCCESS) ("cuStreamGetCtx() error: " * cuGetErrorString(result))
        return pop!(context_array)
    end
end

function cuStreamWaitEvent(hStream::CUstream, hEvent::CUevent)::Nothing
    local result::CUresult = cuStreamWaitEvent(hStream, hEvent, Cuint(0))
    @assert (result == CUDA_SUCCESS) ("cuStreamWaitEvent() error: " * cuGetErrorString(result))
    nothing
end

# execution control functions
function cuFuncGetAttribute(attrib::CUfunction_attribute, hfunc::CUfunction)::Cint
    local attribute_value_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuFuncGetAttribute(attribute_value_array, attrib, hfunc)
    @assert (result == CUDA_SUCCESS) ("cuFuncGetAttribute() error: " * cuGetErrorString(result))
    return pop!(attribute_value_array)
end

# occupancy functions
function cuOccupancyMaxActiveBlocksPerMultiprocessor(func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t)::Cint
    local numblocks_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuOccupancyMaxActiveBlocksPerMultiprocessor(numblocks_array, func, blockSize, dynamicSMemSize)
    @assert (result == CUDA_SUCCESS) ("cuOccupancyMaxActiveBlocksPerMultiprocessor() error: " * cuGetErrorString(result))
    return pop!(numblocks_array)
end

cuOccupancyMaxActiveBlocksPerMultiprocessor(func::CUfunction, blockSize::Integer, dynamicSMemSize::Integer) = cuOccupancyMaxActiveBlocksPerMultiprocessor(func::CUfunction, Cint(blockSize), Csize_t(dynamicSMemSize))

function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t, flags::CUoccupancy_flags)::Cint
    local numblocks_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numblocks_array, func, blockSize, dynamicSMemSize, flags)
    @assert (result == CUDA_SUCCESS) ("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() error: " * cuGetErrorString(result))
    return pop!(numblocks_array)
end

cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func::CUfunction, blockSize::Integer, dynamicSMemSize::Integer, flags::CUoccupancy_flags) = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func, Cint(blockSize), Csize_t(dynamicSMemSize), flags)

function cuOccupancyMaxPotentialBlockSize(func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint)::Tuple{Cint, Cint}
    local mgs_array::Array{Cint, 1} = zeros(Cint, 1)
    local bs_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuOccupancyMaxPotentialBlockSize(mgs_array, bs_array, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
    @assert (result == CUDA_SUCCESS) ("cuOccupancyMaxPotentialBlockSize() error: " * cuGetErrorString(result))
    return (pop!(mgs_array), pop!(bs_array))
end

cuOccupancyMaxPotentialBlockSize(func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Integer, blockSizeLimit::Integer) = cuOccupancyMaxPotentialBlockSize(func, blockSizeToDynamicSMemSize, Csize_t(dynamicSMemSize), Cint(blockSizeLimit))

function cuOccupancyMaxPotentialBlockSizeWithFlags(func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint, flags::CUoccupancy_flags)::Tuple{Cint, Cint}
    local mgs_array::Array{Cint, 1} = zeros(Cint, 1)
    local bs_array::Array{Cint, 1} = zeros(Cint, 1)
    local result::CUresult = cuOccupancyMaxPotentialBlockSizeWithFlags(mgs_array, bs_array, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
    @assert (result == CUDA_SUCCESS) ("cuOccupancyMaxPotentialBlockSizeWithFlags() error: " * cuGetErrorString(result))
    return (pop!(mgs_array), pop!(bs_array))
end

cuOccupancyMaxPotentialBlockSizeWithFlags(func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Integer, blockSizeLimit::Integer, flags::CUoccupancy_flags) = cuOccupancyMaxPotentialBlockSizeWithFlags(func, blockSizeToDynamicSMemSize, Csize_t(dynamicSMemSize), Cint(blockSizeLimit), flags)

# CUDAArray functions
function _cast_cudaarray_args(x::T) where T
    if (T <: CUDAArray)
        return x.ptr
    else
        return x
    end
end

@inline function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Tuple{Vararg{DataType}}, args...; kwargs...)
    return cuLaunchKernel(func, grid, block, Tuple{types...}, map(_cast_cudaarray_args, args)...; kwargs...)
end

@generated function cuLaunchKernel(func::CUfunction, grid::dim3, block::dim3, types::Type, args...; stream::CUstream = CUstream(C_NULL), shmem::Integer = 0, kwargs...)
    local result_expr::Expr = Expr(:block)

    push!(result_expr.args, Base.@_inline_meta)

    args_types = types.parameters[1].parameters
    args_syms = Array{Symbol, 1}(undef, length(args))
    args_ptrs = Array{Symbol, 1}(undef, length(args))

    for i in 1:length(args)
        # assign safely referenced data to corresponding symbol
        args_syms[i] = gensym()
        push!(result_expr.args, :($(args_syms[i]) = Base.cconvert($(args_types[i]), args[$i])))
        
        # generate julia expressions to obtain 
        args_ptrs[i] = gensym()
        push!(result_expr.args, :($(args_ptrs[i]) = Base.unsafe_convert($(args_types[i]), $(args_syms[i]))))
    end
    
    append!(result_expr.args, (quote
        GC.@preserve $(args_syms...) begin
            arguments::Array{Any, 1} = [$(args_ptrs...)]
            result::CUresult = cuLaunchKernel(func,
                                grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                Cuint(shmem),
                                stream,
                                Base.unsafe_convert(Ptr{Ptr{Nothing}}, arguments),
                                Ptr{Ptr{Nothing}}(C_NULL))
            @assert (result == CUDA_SUCCESS) ("cuLaunchKernel() error: " * cuGetErrorName(result))
        end
    end).args)

    return result_expr
end

