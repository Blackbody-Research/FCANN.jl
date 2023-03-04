#=*
* CUDA API v10.1 functions
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

function cuGetErrorString(error::CUresult, pStr::Array{Ptr{UInt8}, 1})::CUresult
    return ccall((:cuGetErrorString, libcuda), CUresult, (CUresult, Ref{Ptr{UInt8}},), error, Base.cconvert(Ref{Ptr{UInt8}}, pStr))
end

function cuGetErrorString(error::CUresult, pStr::Ptr{Ptr{UInt8}})::CUresult
    return ccall((:cuGetErrorString, libcuda), CUresult, (CUresult, Ptr{Ptr{UInt8}},), error, pStr)
end

function cuGetErrorName(error::CUresult, pStr::Array{Ptr{UInt8}, 1})::CUresult
    return ccall((:cuGetErrorName, libcuda), CUresult, (CUresult, Ref{Ptr{UInt8}},), error, Base.cconvert(Ref{Ptr{UInt8}}, pStr))
end

function cuGetErrorName(error::CUresult, pStr::Ptr{Ptr{UInt8}})::CUresult
    return ccall((:cuGetErrorName, libcuda), CUresult, (CUresult, Ptr{Ptr{UInt8}},), error, pStr)
end

function cuInit(Flags::Cuint)::CUresult
    return ccall((:cuInit, libcuda), CUresult, (Cuint,), Flags)
end

function cuDriverGetVersion(driverVersion::Array{Cint, 1})::CUresult
    return ccall((:cuDriverGetVersion, libcuda), CUresult, (Ref{Cint},), Base.cconvert(Ref{Cint}, driverVersion))
end

function cuDriverGetVersion(driverVersion::Ptr{Cint})::CUresult
    return ccall((:cuDriverGetVersion, libcuda), CUresult, (Ptr{Cint},), driverVersion)
end

function cuDeviceGet(device::Array{CUdevice, 1}, ordinal::Cint)::CUresult
    return ccall((:cuDeviceGet, libcuda), CUresult, (Ref{CUdevice}, Cint,), Base.cconvert(Ref{CUdevice}, device), ordinal)
end

function cuDeviceGet(device::Ptr{CUdevice}, ordinal::Cint)::CUresult
    return ccall((:cuDeviceGet, libcuda), CUresult, (Ptr{CUdevice}, Cint,), device, ordinal)
end

function cuDeviceGetCount(count::Array{Cint, 1})::CUresult
    return ccall((:cuDeviceGetCount, libcuda), CUresult, (Ref{Cint},), Base.cconvert(Ref{Cint}, count))
end

function cuDeviceGetCount(count::Ptr{Cint})::CUresult
    return ccall((:cuDeviceGetCount, libcuda), CUresult, (Ptr{Cint},), count)
end

function cuDeviceGetName(name::Array{UInt8, 1}, len::Cint, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetName, libcuda), CUresult, (Ref{UInt8}, Cint, CUdevice,), Base.cconvert(Ref{UInt8}, name), len, dev)
end

function cuDeviceGetName(name::Ptr{UInt8}, len::Cint, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetName, libcuda), CUresult, (Ptr{UInt8}, Cint, CUdevice,), name, len, dev)
end

# cuDeviceGetUuid() available since CUDA 9.2
function cuDeviceGetUuid(uuid::Array{CUuuid, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetUuid, libcuda), CUresult, (Ref{CUuuid}, CUdevice,), Base.cconvert(Ref{CUuuid}, uuid), dev)
end

function cuDeviceGetUuid(uuid::Ptr{CUuuid}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetUuid, libcuda), CUresult, (Ptr{CUuuid}, CUdevice,), uuid, dev)
end

# cuDeviceGetLuid() available since CUDA 10.0
# cuDeviceGetLuid() exists only on Windows operating systems
function cuDeviceGetLuid(luid::Array{UInt8, 1}, deviceNodeMask::Array{Cuint, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetLuid, libcuda), CUresult, (Ref{UInt8}, Ref{Cuint}, CUdevice,), Base.cconvert(Ref{UInt8}, luid), Base.cconvert(Ref{Cuint}, deviceNodeMask), dev)
end

function cuDeviceGetLuid(luid::Ptr{UInt8}, deviceNodeMask::Ptr{Cuint}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetLuid, libcuda), CUresult, (Ptr{UInt8}, Ptr{Cuint}, CUdevice,), luid, deviceNodeMask, dev)
end

# cuDeviceTotalMem() available since CUDA 3.2
function cuDeviceTotalMem(bytes::Array{Csize_t, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceTotalMem, libcuda), CUresult, (Ref{Csize_t}, CUdevice,), Base.cconvert(Ref{Csize_t}, bytes), dev)
end

function cuDeviceTotalMem(bytes::Ptr{Csize_t}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceTotalMem, libcuda), CUresult, (Ptr{Csize_t}, CUdevice,), bytes, dev)
end

function cuDeviceGetAttribute(pi::Array{Cint, 1}, attrib::CUdevice_attribute, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetAttribute, libcuda), CUresult, (Ref{Cint}, CUdevice_attribute, CUdevice,), Base.cconvert(Ref{Cint}, pi), attrib, dev)
end

function cuDeviceGetAttribute(pi::Ptr{Cint}, attrib::CUdevice_attribute, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetAttribute, libcuda), CUresult, (Ptr{Cint}, CUdevice_attribute, CUdevice,), pi, attrib, dev)
end

function cuDeviceGetProperties(prop::Array{CUdevprop, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetProperties, libcuda), CUresult, (Ref{CUdevprop}, CUdevice,), Base.cconvert(Ref{CUdevprop}, prop), dev)
end

function cuDeviceGetProperties(prop::Ptr{CUdevprop}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetProperties, libcuda), CUresult, (Ptr{CUdevprop}, CUdevice,), prop, dev)
end

function cuDeviceComputeCapability(major::Array{Cint, 1}, minor::Array{Cint, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceComputeCapability, libcuda), CUresult, (Ref{Cint}, Ref{Cint}, CUdevice,), Base.cconvert(Ref{Cint}, major), Base.cconvert(Ref{Cint}, minor), dev)
end

function cuDeviceComputeCapability(major::Ptr{Cint}, minor::Ptr{Cint}, dev::CUdevice)::CUresult
    return ccall((:cuDeviceComputeCapability, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUdevice,), major, minor, dev)
end

# cuDevicePrimaryCtxRetain() available since CUDA 7.0
function cuDevicePrimaryCtxRetain(pctx::Array{CUcontext, 1}, dev::CUdevice)::CUresult
    return ccall((:cuDevicePrimaryCtxRetain, libcuda), CUresult, (Ref{CUcontext}, CUdevice,), Base.cconvert(Ref{CUcontext}, pctx), dev)
end

function cuDevicePrimaryCtxRetain(pctx::Ptr{CUcontext}, dev::CUdevice)::CUresult
    return ccall((:cuDevicePrimaryCtxRetain, libcuda), CUresult, (Ptr{CUcontext}, CUdevice,), pctx, dev)
end

# cuDevicePrimaryCtxRelease() available since CUDA 7.0
function cuDevicePrimaryCtxRelease(dev::CUdevice)::CUresult
    return ccall((:cuDevicePrimaryCtxRelease, libcuda), CUresult, (CUdevice,), dev)
end

# cuDevicePrimaryCtxSetFlags() available since CUDA 7.0
function cuDevicePrimaryCtxSetFlags(dev::CUdevice, flags::Cuint)::CUresult
    return ccall((:cuDevicePrimaryCtxSetFlags, libcuda), CUresult, (CUdevice, Cuint,), dev, flags)
end

# cuDevicePrimaryCtxGetState() available since CUDA 7.0
function cuDevicePrimaryCtxGetState(dev::CUdevice, flags::Array{Cuint, 1}, active::Array{Cint, 1})::CUresult
    return ccall((:cuDevicePrimaryCtxGetState, libcuda), CUresult, (CUdevice, Ref{Cuint}, Ref{Cint},), dev, Base.cconvert(Ref{Cuint}, flags), Base.cconvert(Ref{Cint}, active))
end

function cuDevicePrimaryCtxGetState(dev::CUdevice, flags::Ptr{Cuint}, active::Ptr{Cint})::CUresult
    return ccall((:cuDevicePrimaryCtxGetState, libcuda), CUresult, (CUdevice, Ptr{Cuint}, Ptr{Cint},), dev, flags, active)
end

# cuDevicePrimaryCtxReset() available since CUDA 7.0
function cuDevicePrimaryCtxReset(dev::CUdevice)::CUresult
    return ccall((:cuDevicePrimaryCtxReset, libcuda), CUresult, (CUdevice,), dev)
end

# cuCtxCreate() available since CUDA 3.2
function cuCtxCreate(pctx::Array{CUcontext, 1}, flags::Cuint, dev::CUdevice)::CUresult
    return ccall((:cuCtxCreate, libcuda), CUresult, (Ref{CUcontext}, Cuint, CUdevice,), Base.cconvert(Ref{CUcontext}, pctx), flags, dev)
end

function cuCtxCreate(pctx::Ptr{CUcontext}, flags::Cuint, dev::CUdevice)::CUresult
    return ccall((:cuCtxCreate, libcuda), CUresult, (Ptr{CUcontext}, Cuint, CUdevice,), pctx, flags, dev)
end

# cuCtxDestroy() available since CUDA 4.0
function cuCtxDestroy(ctx::CUcontext)::CUresult
    return ccall((:cuCtxDestroy, libcuda), CUresult, (CUcontext,), ctx)
end

# cuCtxPushCurrent() available since CUDA 4.0
function cuCtxPushCurrent(ctx::CUcontext)::CUresult
    return ccall((:cuCtxPushCurrent, libcuda), CUresult, (CUcontext,), ctx)
end

# cuCtxPopCurrent() available since CUDA 4.0
function cuCtxPopCurrent(pctx::Array{CUcontext, 1})::CUresult
    return ccall((:cuCtxPopCurrent, libcuda), CUresult, (Ref{CUcontext},), Base.cconvert(Ref{CUcontext}, pctx))
end

function cuCtxPopCurrent(pctx::Ptr{CUcontext})::CUresult
    return ccall((:cuCtxPopCurrent, libcuda), CUresult, (Ptr{CUcontext},), pctx)
end

# cuCtxSetCurrent() available since CUDA 4.0
function cuCtxSetCurrent(ctx::CUcontext)::CUresult
    return ccall((:cuCtxSetCurrent, libcuda), CUresult, (CUcontext,), ctx)
end

# cuCtxGetCurrent() available since CUDA 4.0
function cuCtxGetCurrent(pctx::Array{CUcontext, 1})::CUresult
    return ccall((:cuCtxGetCurrent, libcuda), CUresult, (Ref{CUcontext},), Base.cconvert(Ref{CUcontext}, pctx))
end

function cuCtxGetCurrent(pctx::Ptr{CUcontext})::CUresult
    return ccall((:cuCtxGetCurrent, libcuda), CUresult, (Ptr{CUcontext},), pctx)
end

function cuCtxGetDevice(device::Array{CUdevice, 1})::CUresult
    return ccall((:cuCtxGetDevice, libcuda), CUresult, (Ref{CUdevice},), Base.cconvert(Ref{CUdevice}, device))
end

function cuCtxGetDevice(device::Ptr{CUdevice})::CUresult
    return ccall((:cuCtxGetDevice, libcuda), CUresult, (Ptr{CUdevice},), device)
end

# cuCtxGetFlags() available since CUDA 7.0
function cuCtxGetFlags(flags::Array{Cuint, 1})::CUresult
    return ccall((:cuCtxGetFlags, libcuda), CUresult, (Ref{Cuint},), Base.cconvert(Ref{Cuint}, flags))
end

function cuCtxGetFlags(flags::Ptr{Cuint})::CUresult
    return ccall((:cuCtxGetFlags, libcuda), CUresult, (Ptr{Cuint},), flags)
end

function cuCtxSynchronize()::CUresult
    return ccall((:cuCtxSynchronize, libcuda), CUresult, (),)
end

function cuCtxSetLimit(limit::CUlimit, value::Csize_t)::CUresult
    return ccall((:cuCtxSetLimit, libcuda), CUresult, (CUlimit, Csize_t,), limit, value)
end

function cuCtxGetLimit(pvalue::Array{Csize_t, 1}, limit::CUlimit)::CUresult
    return ccall((:cuCtxGetLimit, libcuda), CUresult, (Ref{Csize_t}, CUlimit,), Base.cconvert(Ref{Csize_t}, pvalue), limit)
end

function cuCtxGetLimit(pvalue::Ptr{Csize_t}, limit::CUlimit)::CUresult
    return ccall((:cuCtxGetLimit, libcuda), CUresult, (Ptr{Csize_t}, CUlimit,), pvalue, limit)
end

function cuCtxGetCacheConfig(pconfig::Array{CUfunc_cache, 1})::CUresult
    return ccall((:cuCtxGetCacheConfig, libcuda), CUresult, (Ref{CUfunc_cache},), Base.cconvert(Ref{CUfunc_cache}, pconfig))
end

function cuCtxGetCacheConfig(pconfig::Ptr{CUfunc_cache})::CUresult
    return ccall((:cuCtxGetCacheConfig, libcuda), CUresult, (Ptr{CUfunc_cache},), pconfig)
end

function cuCtxSetCacheConfig(config::CUfunc_cache)::CUresult
    return ccall((:cuCtxSetCacheConfig, libcuda), CUresult, (CUfunc_cache,), config)
end

function cuCtxGetSharedMemConfig(pConfig::Array{CUsharedconfig, 1})::CUresult
    return ccall((:cuCtxGetSharedMemConfig, libcuda), CUresult, (Ref{CUsharedconfig},), Base.cconvert(Ref{CUsharedconfig}, pConfig))
end

# cuCtxGetSharedMemConfig() available since CUDA 4.2
function cuCtxGetSharedMemConfig(pConfig::Ptr{CUsharedconfig})::CUresult
    return ccall((:cuCtxGetSharedMemConfig, libcuda), CUresult, (Ptr{CUsharedconfig},), pConfig)
end

# cuCtxSetSharedMemConfig() available since CUDA 4.2
function cuCtxSetSharedMemConfig(config::CUsharedconfig)::CUresult
    return ccall((:cuCtxSetSharedMemConfig, libcuda), CUresult, (CUsharedconfig,), config)
end

function cuCtxGetApiVersion(ctx::CUcontext, version::Array{Cuint, 1})::CUresult
    return ccall((:cuCtxGetApiVersion, libcuda), CUresult, (CUcontext, Ref{Cuint},), ctx, Base.cconvert(Ref{Cuint}, version))
end

function cuCtxGetApiVersion(ctx::CUcontext, version::Ptr{Cuint})::CUresult
    return ccall((:cuCtxGetApiVersion, libcuda), CUresult, (CUcontext, Ptr{Cuint},), ctx, version)
end

function cuCtxGetStreamPriorityRange(leastPriority::Array{Cint, 1}, greatestPriority::Array{Cint, 1})::CUresult
    return ccall((:cuCtxGetStreamPriorityRange, libcuda), CUresult, (Ref{Cint}, Ref{Cint},), Base.cconvert(Ref{Cint}, leastPriority), Base.cconvert(Ref{Cint}, greatestPriority))
end

function cuCtxGetStreamPriorityRange(leastPriority::Ptr{Cint}, greatestPriority::Ptr{Cint})::CUresult
    return ccall((:cuCtxGetStreamPriorityRange, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint},), leastPriority, greatestPriority)
end

function cuCtxAttach(pctx::Array{CUcontext, 1}, flags::Cuint)::CUresult
    return ccall((:cuCtxAttach, libcuda), CUresult, (Ref{CUcontext}, Cuint,), Base.cconvert(Ref{CUcontext}, pctx), flags)
end

function cuCtxAttach(pctx::Ptr{CUcontext}, flags::Cuint)::CUresult
    return ccall((:cuCtxAttach, libcuda), CUresult, (Ptr{CUcontext}, Cuint,), pctx, flags)
end

function cuCtxDetach(ctx::CUcontext)::CUresult
    return ccall((:cuCtxDetach, libcuda), CUresult, (CUcontext,), ctx)
end

function cuModuleLoad(pmodule::Array{CUmodule, 1}, fname::Array{UInt8, 1})::CUresult
    return ccall((:cuModuleLoad, libcuda), CUresult, (Ref{CUmodule}, Ref{UInt8},), Base.cconvert(Ref{CUmodule}, pmodule), Base.cconvert(Ref{UInt8}, fname))
end

function cuModuleLoad(pmodule::Ptr{CUmodule}, fname::Ptr{UInt8})::CUresult
    return ccall((:cuModuleLoad, libcuda), CUresult, (Ptr{CUmodule}, Ptr{UInt8},), pmodule, fname)
end

function cuModuleLoadData(pmodule::Array{CUmodule, 1}, image::Ptr{Cvoid})::CUresult
    return ccall((:cuModuleLoadData, libcuda), CUresult, (Ref{CUmodule}, Ptr{Cvoid},), Base.cconvert(Ref{CUmodule}, pmodule), image)
end

function cuModuleLoadData(pmodule::Ptr{CUmodule}, image::Ptr{Cvoid})::CUresult
    return ccall((:cuModuleLoadData, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid},), pmodule, image)
end

function cuModuleLoadDataEx(pmodule::Array{CUmodule, 1}, image::Ptr{Cvoid}, numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1})::CUresult
    return ccall((:cuModuleLoadDataEx, libcuda),
                CUresult,
                (Ref{CUmodule}, Ptr{Cvoid}, Cuint, Ref{CUjit_option}, Ref{Ptr{Cvoid}},),
                Base.cconvert(Ref{CUmodule}, pmodule),
                image,
                numOptions,
                Base.cconvert(Ref{CUjit_option}, options),
                Base.cconvert(Ref{Ptr{Cvoid}}, optionValues))
end

function cuModuleLoadDataEx(pmodule::Ptr{CUmodule}, image::Ptr{Cvoid}, numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}})::CUresult
    return ccall((:cuModuleLoadDataEx, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}},), pmodule, image, numOptions, options, optionValues)
end

function cuModuleLoadFatBinary(pmodule::Array{CUmodule, 1}, fatCubin::Ptr{Cvoid})::CUresult
    return ccall((:cuModuleLoadFatBinary, libcuda), CUresult, (Ref{CUmodule}, Ptr{Cvoid},), Base.cconvert(Ref{CUmodule}, pmodule), fatCubin)
end

function cuModuleLoadFatBinary(pmodule::Ptr{CUmodule}, fatCubin::Ptr{Cvoid})::CUresult
    return ccall((:cuModuleLoadFatBinary, libcuda), CUresult, (Ptr{CUmodule}, Ptr{Cvoid},), pmodule, fatCubin)
end

function cuModuleUnload(hmod::CUmodule)::CUresult
    return ccall((:cuModuleUnload, libcuda), CUresult, (CUmodule,), hmod)
end

function cuModuleGetFunction(hfunc::Array{CUfunction, 1}, hmod::CUmodule, name::Array{UInt8, 1})::CUresult
    return ccall((:cuModuleGetFunction, libcuda), CUresult, (Ref{CUfunction}, CUmodule, Ref{UInt8},), Base.cconvert(Ref{CUfunction}, hfunc), hmod, Base.cconvert(Ref{UInt8}, name))
end

function cuModuleGetFunction(hfunc::Ptr{CUfunction}, hmod::CUmodule, name::Ptr{UInt8})::CUresult
    return ccall((:cuModuleGetFunction, libcuda), CUresult, (Ptr{CUfunction}, CUmodule, Ptr{UInt8},), hfunc, hmod, name)
end

# cuModuleGetGlobal() available since CUDA 3.2
function cuModuleGetGlobal(dptr::Array{CUdeviceptr, 1}, bytes::Array{Csize_t, 1}, hmod::CUmodule, name::Array{UInt8, 1})::CUresult
    return ccall((:cuModuleGetGlobal, libcuda), CUresult, (Ref{CUdeviceptr}, Ref{Csize_t}, CUmodule, Ref{UInt8},), Base.cconvert(Ref{CUdeviceptr}, dptr), Base.cconvert(Ref{Csize_t}, bytes), hmod, Base.cconvert(Ref{UInt8}, name))
end

function cuModuleGetGlobal(dptr::Ptr{CUdeviceptr}, bytes::Ptr{Csize_t}, hmod::CUmodule, name::Ptr{UInt8})::CUresult
    return ccall((:cuModuleGetGlobal, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUmodule, Ptr{UInt8},), dptr, bytes, hmod, name)
end

function cuModuleGetTexRef(pTexRef::Array{CUtexref, 1}, hmod::CUmodule, name::Array{UInt8, 1})::CUresult
    return ccall((:cuModuleGetTexRef, libcuda), CUresult, (Ref{CUtexref}, CUmodule, Ref{UInt8},), Base.cconvert(Ref{CUtexref}, pTexRef), hmod, Base.cconvert(Ref{UInt8}, name))
end

function cuModuleGetTexRef(pTexRef::Ptr{CUtexref}, hmod::CUmodule, name::Ptr{UInt8})::CUresult
    return ccall((:cuModuleGetTexRef, libcuda), CUresult, (Ptr{CUtexref}, CUmodule, Ptr{UInt8},), pTexRef, hmod, name)
end

function cuModuleGetSurfRef(pSurfRef::Array{CUsurfref, 1}, hmod::CUmodule, name::Array{UInt8, 1})::CUresult
    return ccall((:cuModuleGetSurfRef, libcuda), CUresult, (Ref{CUsurfref}, CUmodule, Ref{UInt8},), Base.cconvert(Ref{CUsurfref}, pSurfRef), hmod, Base.cconvert(Ref{UInt8}, name))
end

function cuModuleGetSurfRef(pSurfRef::Ptr{CUsurfref}, hmod::CUmodule, name::Ptr{UInt8})::CUresult
    return ccall((:cuModuleGetSurfRef, libcuda), CUresult, (Ptr{CUsurfref}, CUmodule, Ptr{UInt8},), pSurfRef, hmod, name)
end

# cuLinkCreate() available since CUDA 5.5
function cuLinkCreate(numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1}, stateOut::Array{CUlinkState, 1})::CUresult
    return ccall((:cuLinkCreate, libcuda), CUresult, (Cuint, Ref{CUjit_option}, Ref{Ptr{Cvoid}}, Ref{CUlinkState},), numOptions, Base.cconvert(Ref{CUjit_option}, options), Base.cconvert(Ref{Ptr{Cvoid}}, optionValues), Base.cconvert(Ref{CUlinkState}, stateOut))
end

function cuLinkCreate(numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}}, stateOut::Ptr{CUlinkState})::CUresult
    return ccall((:cuLinkCreate, libcuda), CUresult, (Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}, Ptr{CUlinkState},), numOptions, options, optionValues, stateOut)
end

# cuLinkAddData() available since CUDA 5.5
function cuLinkAddData(state::CUlinkState, type::CUjitInputType, data::Ptr{Cvoid}, size::Csize_t, name::Array{UInt8, 1}, numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1})::CUresult
    return ccall((:cuLinkAddData, libcuda), CUresult, (CUlinkState, CUjitInputType, Ptr{Cvoid}, Csize_t, Ref{UInt8}, Cuint, Ref{CUjit_option}, Ref{Ptr{Cvoid}},), state, type, data, size, Base.cconvert(Ref{UInt8}, name), numOptions, Base.cconvert(Ref{CUjit_option}, options), Base.cconvert(Ref{Ptr{Cvoid}}, optionValues))
end

function cuLinkAddData(state::CUlinkState, type::CUjitInputType, data::Ptr{Cvoid}, size::Csize_t, name::Ptr{UInt8}, numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}})::CUresult
    return ccall((:cuLinkAddData, libcuda), CUresult, (CUlinkState, CUjitInputType, Ptr{Cvoid}, Csize_t, Ptr{UInt8}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}},), state, type, data, size, name, numOptions, options, optionValues)
end

# cuLinkAddFile() available since CUDA 5.5
function cuLinkAddFile(state::CUlinkState, type::CUjitInputType, path::Array{UInt8, 1}, numOptions::Cuint, options::Array{CUjit_option, 1}, optionValues::Array{Ptr{Cvoid}, 1})::CUresult
    return ccall((:cuLinkAddFile, libcuda), CUresult, (CUlinkState, CUjitInputType, Ref{UInt8}, Cuint, Ref{CUjit_option}, Ref{Ptr{Cvoid}},), state, type, Base.cconvert(Ref{UInt8}, path), numOptions, Base.cconvert(Ref{CUjit_option}, options), Base.cconvert(Ref{Ptr{Cvoid}}, optionValues))
end

function cuLinkAddFile(state::CUlinkState, type::CUjitInputType, path::Ptr{UInt8}, numOptions::Cuint, options::Ptr{CUjit_option}, optionValues::Ptr{Ptr{Cvoid}})::CUresult
    return ccall((:cuLinkAddFile, libcuda), CUresult, (CUlinkState, CUjitInputType, Ptr{UInt8}, Cuint, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}},), state, type, path, numOptions, options, optionValues)
end

# cuLinkComplete() available since CUDA 5.5
function cuLinkComplete(state::CUlinkState, cubinOut::Array{Ptr{Cvoid}, 1}, sizeOut::Array{Csize_t, 1})::CUresult
    return ccall((:cuLinkComplete, libcuda), CUresult, (CUlinkState, Ref{Ptr{Cvoid}}, Ref{Csize_t},), state, Base.cconvert(Ref{Ptr{Cvoid}}, cubinOut), Base.cconvert(Ref{Csize_t}, sizeOut))
end

function cuLinkComplete(state::CUlinkState, cubinOut::Ptr{Ptr{Cvoid}}, sizeOut::Ptr{Csize_t})::CUresult
    return ccall((:cuLinkComplete, libcuda), CUresult, (CUlinkState, Ptr{Ptr{Cvoid}}, Ptr{Csize_t},), state, cubinOut, sizeOut)
end

# cuLinkDestroy() available since CUDA 5.5
function cuLinkDestroy(state::CUlinkState)::CUresult
    return ccall((:cuLinkDestroy, libcuda), CUresult, (CUlinkState,), state)
end

# cuMemGetInfo() available since CUDA 3.2
function cuMemGetInfo(free::Array{Csize_t, 1}, total::Array{Csize_t, 1})::CUresult
    return ccall((:cuMemGetInfo, libcuda), CUresult, (Ref{Csize_t}, Ref{Csize_t},), Base.cconvert(Ref{Csize_t}, free), Base.cconvert(Ref{Csize_t}, total))
end

function cuMemGetInfo(free::Ptr{Csize_t}, total::Ptr{Csize_t})::CUresult
    return ccall((:cuMemGetInfo, libcuda), CUresult, (Ptr{Csize_t}, Ptr{Csize_t},), free, total)
end

# cuMemAlloc() available since CUDA 3.2
function cuMemAlloc(dptr::Array{CUdeviceptr, 1}, bytesize::Csize_t)::CUresult
    return ccall((:cuMemAlloc, libcuda), CUresult, (Ref{CUdeviceptr}, Csize_t,), Base.cconvert(Ref{CUdeviceptr}, dptr), bytesize)
end

function cuMemAlloc(dptr::Ptr{CUdeviceptr}, bytesize::Csize_t)::CUresult
    return ccall((:cuMemAlloc, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t,), dptr, bytesize)
end

# cuMemAllocPitch() available since CUDA 3.2
function cuMemAllocPitch(dptr::Array{CUdeviceptr, 1}, pPitch::Array{Csize_t, 1}, WidthInBytes::Csize_t, Height::Csize_t, ElementSizeBytes::Cuint)::CUresult
    return ccall((:cuMemAllocPitch, libcuda), CUresult, (Ref{CUdeviceptr}, Ref{Csize_t}, Csize_t, Csize_t, Cuint,), Base.cconvert(Ref{CUdeviceptr}, dptr), Base.cconvert(Ref{Csize_t}, pPitch), WidthInBytes, Height, ElementSizeBytes)
end

function cuMemAllocPitch(dptr::Ptr{CUdeviceptr}, pPitch::Ptr{Csize_t}, WidthInBytes::Csize_t, Height::Csize_t, ElementSizeBytes::Cuint)::CUresult
    return ccall((:cuMemAllocPitch, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, Csize_t, Csize_t, Cuint,), dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
end

# cuMemFree() available since CUDA 3.2
function cuMemFree(dptr::CUdeviceptr)::CUresult
    return ccall((:cuMemFree, libcuda), CUresult, (CUdeviceptr,), dptr)
end

# cuMemGetAddressRange() available since CUDA 3.2
function cuMemGetAddressRange(pbase::Array{CUdeviceptr, 1}, psize::Array{Csize_t, 1}, dptr::CUdeviceptr)::CUresult
    return ccall((:cuMemGetAddressRange, libcuda), CUresult, (Ref{CUdeviceptr}, Ref{Csize_t}, CUdeviceptr,), Base.cconvert(Ref{CUdeviceptr}, pbase), Base.cconvert(Ref{Csize_t}, psize), dptr)
end

function cuMemGetAddressRange(pbase::Ptr{CUdeviceptr}, psize::Ptr{Csize_t}, dptr::CUdeviceptr)::CUresult
    return ccall((:cuMemGetAddressRange, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUdeviceptr,), pbase, psize, dptr)
end

# cuMemAllocHost() available since CUDA 3.2
function cuMemAllocHost(pp::Array{Ptr{Cvoid}, 1}, bytesize::Csize_t)::CUresult
    return ccall((:cuMemAllocHost, libcuda), CUresult, (Ref{Ptr{Cvoid}}, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, pp), bytesize)
end

function cuMemAllocHost(pp::Ptr{Ptr{Cvoid}}, bytesize::Csize_t)::CUresult
    return ccall((:cuMemAllocHost, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Csize_t,), pp, bytesize)
end

function cuMemFreeHost(p::Ptr{Cvoid})::CUresult
    return ccall((:cuMemFreeHost, libcuda), CUresult, (Ptr{Cvoid},), p)
end

function cuMemHostAlloc(pp::Array{Ptr{Cvoid}, 1}, bytesize::Csize_t, Flags::Cuint)::CUresult
    return ccall((:cuMemHostAlloc, libcuda), CUresult, (Ref{Ptr{Cvoid}}, Csize_t, Cuint,), Base.cconvert(Ref{Ptr{Cvoid}}, pp), bytesize, Flags)
end

function cuMemHostAlloc(pp::Ptr{Ptr{Cvoid}}, bytesize::Csize_t, Flags::Cuint)::CUresult
    return ccall((:cuMemHostAlloc, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Csize_t, Cuint,), pp, bytesize, Flags)
end

# cuMemHostGetDevicePointer() available since CUDA 3.2
function cuMemHostGetDevicePointer(pdptr::Array{CUdeviceptr, 1}, p::Ptr{Cvoid}, Flags::Cuint)::CUresult
    return ccall((:cuMemHostGetDevicePointer, libcuda), CUresult, (Ref{CUdeviceptr}, Ptr{Cvoid}, Cuint,), Base.cconvert(Ref{CUdeviceptr}, pdptr), p, Flags)
end

function cuMemHostGetDevicePointer(pdptr::Ptr{CUdeviceptr}, p::Ptr{Cvoid}, Flags::Cuint)::CUresult
    return ccall((:cuMemHostGetDevicePointer, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Cvoid}, Cuint,), pdptr, p, Flags)
end

function cuMemHostGetFlags(pFlags::Array{Cuint, 1}, p::Ptr{Cvoid})::CUresult
    return ccall((:cuMemHostGetFlags, libcuda), CUresult, (Ref{Cuint}, Ptr{Cvoid},), Base.cconvert(Ref{Cuint}, pFlags), p)
end

function cuMemHostGetFlags(pFlags::Ptr{Cuint}, p::Ptr{Cvoid})::CUresult
    return ccall((:cuMemHostGetFlags, libcuda), CUresult, (Ptr{Cuint}, Ptr{Cvoid},), pFlags, p)
end

# cuMemAllocManaged() available since CUDA 6.0
function cuMemAllocManaged(dptr::Array{CUdeviceptr, 1}, bytesize::Csize_t, flags::Cuint)::CUresult
    return ccall((:cuMemAllocManaged, libcuda), CUresult, (Ref{CUdeviceptr}, Csize_t, Cuint,), Base.cconvert(Ref{CUdeviceptr}, dptr), bytesize, flags)
end

function cuMemAllocManaged(dptr::Ptr{CUdeviceptr}, bytesize::Csize_t, flags::Cuint)::CUresult
    return ccall((:cuMemAllocManaged, libcuda), CUresult, (Ptr{CUdeviceptr}, Csize_t, Cuint,), dptr, bytesize, flags)
end

# cuDeviceGetByPCIBusId() available since CUDA 4.1
function cuDeviceGetByPCIBusId(dev::Array{CUdevice, 1}, pciBusId::Array{UInt8, 1})::CUresult
    return ccall((:cuDeviceGetByPCIBusId, libcuda), CUresult, (Ref{CUdevice}, Ref{UInt8},), Base.cconvert(Ref{CUdevice}, dev), Base.cconvert(Ref{UInt8}, pciBusId))
end

function cuDeviceGetByPCIBusId(dev::Ptr{CUdevice}, pciBusId::Ptr{UInt8})::CUresult
    return ccall((:cuDeviceGetByPCIBusId, libcuda), CUresult, (Ptr{CUdevice}, Ptr{UInt8},), dev, pciBusId)
end

# cuDeviceGetPCIBusId() available since CUDA 4.1
function cuDeviceGetPCIBusId(pciBusId::Array{UInt8, 1}, len::Cint, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetPCIBusId, libcuda), CUresult, (Ref{UInt8}, Cint, CUdevice,), Base.cconvert(Ref{UInt8}, pciBusId), len, dev)
end

function cuDeviceGetPCIBusId(pciBusId::Ptr{UInt8}, len::Cint, dev::CUdevice)::CUresult
    return ccall((:cuDeviceGetPCIBusId, libcuda), CUresult, (Ptr{UInt8}, Cint, CUdevice,), pciBusId, len, dev)
end

# cuIpcGetEventHandle() available since CUDA 4.1
function cuIpcGetEventHandle(pHandle::Array{CUipcEventHandle, 1}, event::CUevent)::CUresult
    return ccall((:cuIpcGetEventHandle, libcuda), CUresult, (Ref{CUipcEventHandle}, CUevent,), Base.cconvert(Ref{CUipcEventHandle}, pHandle), event)
end

function cuIpcGetEventHandle(pHandle::Ptr{CUipcEventHandle}, event::CUevent)::CUresult
    return ccall((:cuIpcGetEventHandle, libcuda), CUresult, (Ptr{CUipcEventHandle}, CUevent,), pHandle, event)
end

# cuIpcOpenEventHandle() available since CUDA 4.1
function cuIpcOpenEventHandle(phEvent::Array{CUevent, 1}, handle::CUipcEventHandle)::CUresult
    return ccall((:cuIpcOpenEventHandle, libcuda), CUresult, (Ref{CUevent}, CUipcEventHandle,), Base.cconvert(Ref{CUevent}, phEvent), handle)
end

function cuIpcOpenEventHandle(phEvent::Ptr{CUevent}, handle::CUipcEventHandle)::CUresult
    return ccall((:cuIpcOpenEventHandle, libcuda), CUresult, (Ptr{CUevent}, CUipcEventHandle,), phEvent, handle)
end

# cuIpcGetMemHandle() available since CUDA 4.1
function cuIpcGetMemHandle(pHandle::Array{CUipcMemHandle, 1}, dptr::CUdeviceptr)::CUresult
    return ccall((:cuIpcGetMemHandle, libcuda), CUresult, (Ref{CUipcMemHandle}, CUdeviceptr,), Base.cconvert(Ref{CUipcMemHandle}, pHandle), dptr)
end

function cuIpcGetMemHandle(pHandle::Ptr{CUipcMemHandle}, dptr::CUdeviceptr)::CUresult
    return ccall((:cuIpcGetMemHandle, libcuda), CUresult, (Ptr{CUipcMemHandle}, CUdeviceptr,), pHandle, dptr)
end

# cuIpcOpenMemHandle() available since CUDA 4.1
function cuIpcOpenMemHandle(pdptr::Array{CUdeviceptr, 1}, handle::CUipcMemHandle, Flags::Cuint)::CUresult
    return ccall((:cuIpcOpenMemHandle, libcuda), CUresult, (Ref{CUdeviceptr}, CUipcMemHandle, Cuint,), Base.cconvert(Ref{CUdeviceptr}, pdptr), handle, Flags)
end

function cuIpcOpenMemHandle(pdptr::Ptr{CUdeviceptr}, handle::CUipcMemHandle, Flags::Cuint)::CUresult
    return ccall((:cuIpcOpenMemHandle, libcuda), CUresult, (Ptr{CUdeviceptr}, CUipcMemHandle, Cuint,), pdptr, handle, Flags)
end

# cuIpcCloseMemHandle() available since CUDA 4.1
function cuIpcCloseMemHandle(dptr::CUdeviceptr)::CUresult
    return ccall((:cuIpcCloseMemHandle, libcuda), CUresult, (CUdeviceptr,), dptr)
end

# cuMemHostRegister() available since CUDA 4.0
function cuMemHostRegister(p::Ptr{Cvoid}, bytesize::Csize_t, Flags::Cuint)::CUresult
    return ccall((:cuMemHostRegister, libcuda), CUresult, (Ptr{Cvoid}, Csize_t, Cuint,), p, bytesize, Flags)
end

# cuMemHostUnregister() available since CUDA 4.0
function cuMemHostUnregister(p::Ptr{Cvoid})::CUresult
    return ccall((:cuMemHostUnregister, libcuda), CUresult, (Ptr{Cvoid},), p)
end

# cuMemcpy() available since CUDA 4.0
function cuMemcpy(dst::CUdeviceptr, src::CUdeviceptr, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpy, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t,), dst, src, ByteCount)
end

# cuMemcpyPeer() available since CUDA 4.0
function cuMemcpyPeer(dstDevice::CUdeviceptr, dstContext::CUcontext, srcDevice::CUdeviceptr, srcContext::CUcontext, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyPeer, libcuda), CUresult, (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t,), dstDevice, dstContext, srcDevice, srcContext, ByteCount)
end

# cuMemcpyHtoD() available since CUDA 3.2
function cuMemcpyHtoD(dstDevice::CUdeviceptr, srcHost::Ptr{Cvoid}, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyHtoD, libcuda), CUresult, (CUdeviceptr, Ptr{Cvoid}, Csize_t,), dstDevice, srcHost, ByteCount)
end

# cuMemcpyDtoH() available since CUDA 3.2
function cuMemcpyDtoH(dstHost::Ptr{Cvoid}, srcDevice::CUdeviceptr, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyDtoH, libcuda), CUresult, (Ptr{Cvoid}, CUdeviceptr, Csize_t,), dstHost, srcDevice, ByteCount)
end

# cuMemcpyDtoD() available since CUDA 3.2
function cuMemcpyDtoD(dstDevice::CUdeviceptr, srcDevice::CUdeviceptr, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyDtoD, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t,), dstDevice, srcDevice, ByteCount)
end

# cuMemcpyDtoA() available since CUDA 3.2
function cuMemcpyDtoA(dstArray::CUarray, dstOffset::Csize_t, srcDevice::CUdeviceptr, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyDtoA, libcuda), CUresult, (CUarray, Csize_t, CUdeviceptr, Csize_t,), dstArray, dstOffset, srcDevice, ByteCount)
end

# cuMemcpyAtoD() available since CUDA 3.2
function cuMemcpyAtoD(dstDevice::CUdeviceptr, srcArray::CUarray, srcOffset::Csize_t, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyAtoD, libcuda), CUresult, (CUdeviceptr, CUarray, Csize_t, Csize_t,), dstDevice, srcArray, srcOffset, ByteCount)
end

# cuMemcpyHtoA() available since CUDA 3.2
function cuMemcpyHtoA(dstArray::CUarray, dstOffset::Csize_t, srcHost::Ptr{Cvoid}, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyHtoA, libcuda), CUresult, (CUarray, Csize_t, Ptr{Cvoid}, Csize_t,), dstArray, dstOffset, srcHost, ByteCount)
end

# cuMemcpyAtoH() available since CUDA 3.2
function cuMemcpyAtoH(dstHost::Ptr{Cvoid}, srcArray::CUarray, srcOffset::Csize_t, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyAtoH, libcuda), CUresult, (Ptr{Cvoid}, CUarray, Csize_t, Csize_t,), dstHost, srcArray, srcOffset, ByteCount)
end

# cuMemcpyAtoA() available since CUDA 3.2
function cuMemcpyAtoA(dstArray::CUarray, dstOffset::Csize_t, srcArray::CUarray, srcOffset::Csize_t, ByteCount::Csize_t)::CUresult
    return ccall((:cuMemcpyAtoA, libcuda), CUresult, (CUarray, Csize_t, CUarray, Csize_t, Csize_t,), dstArray, dstOffset, srcArray, srcOffset, ByteCount)
end

# cuMemcpy2D() available since CUDA 3.2
function cuMemcpy2D(pCopy::Array{CUDA_MEMCPY2D, 1})::CUresult
    return ccall((:cuMemcpy2D, libcuda), CUresult, (Ref{CUDA_MEMCPY2D},), Base.cconvert(Ref{CUDA_MEMCPY2D}, pCopy))
end

function cuMemcpy2D(pCopy::Ptr{CUDA_MEMCPY2D})::CUresult
    return ccall((:cuMemcpy2D, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D},), pCopy)
end

# cuMemcpy2DUnaligned() available since CUDA 3.2
function cuMemcpy2DUnaligned(pCopy::Array{CUDA_MEMCPY2D, 1})::CUresult
    return ccall((:cuMemcpy2DUnaligned, libcuda), CUresult, (Ref{CUDA_MEMCPY2D},), Base.cconvert(Ref{CUDA_MEMCPY2D}, pCopy))
end

function cuMemcpy2DUnaligned(pCopy::Ptr{CUDA_MEMCPY2D})::CUresult
    return ccall((:cuMemcpy2DUnaligned, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D},), pCopy)
end

# cuMemcpy3D() available since CUDA 3.2
function cuMemcpy3D(pCopy::Array{CUDA_MEMCPY3D, 1})::CUresult
    return ccall((:cuMemcpy3D, libcuda), CUresult, (Ref{CUDA_MEMCPY3D},), Base.cconvert(Ref{CUDA_MEMCPY3D}, pCopy))
end

function cuMemcpy3D(pCopy::Ptr{CUDA_MEMCPY3D})::CUresult
    return ccall((:cuMemcpy3D, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D},), pCopy)
end

# cuMemcpy3DPeer() available since CUDA 4.0
function cuMemcpy3DPeer(pCopy::Array{CUDA_MEMCPY3D_PEER, 1})::CUresult
    return ccall((:cuMemcpy3DPeer, libcuda), CUresult, (Ref{CUDA_MEMCPY3D_PEER},), Base.cconvert(Ref{CUDA_MEMCPY3D_PEER}, pCopy))
end

function cuMemcpy3DPeer(pCopy::Ptr{CUDA_MEMCPY3D_PEER})::CUresult
    return ccall((:cuMemcpy3DPeer, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D_PEER},), pCopy)
end

# cuMemcpyAsync() available since CUDA 4.0
function cuMemcpyAsync(dst::CUdeviceptr, src::CUdeviceptr, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyAsync, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t, CUstream,), dst, src, ByteCount, hStream)
end

# cuMemcpyPeerAsync() available since CUDA 4.0
function cuMemcpyPeerAsync(dstDevice::CUdeviceptr, dstContext::CUcontext, srcDevice::CUdeviceptr, srcContext::CUcontext, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyPeerAsync, libcuda), CUresult, (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t, CUstream,), dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
end

# cuMemcpyHtoDAsync() available since CUDA 3.2
function cuMemcpyHtoDAsync(dstDevice::CUdeviceptr, srcHost::Ptr{Cvoid}, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyHtoDAsync, libcuda), CUresult, (CUdeviceptr, Ptr{Cvoid}, Csize_t, CUstream,), dstDevice, srcHost, ByteCount, hStream)
end

# cuMemcpyDtoHAsync() available since CUDA 3.2
function cuMemcpyDtoHAsync(dstHost::Ptr{Cvoid}, srcDevice::CUdeviceptr, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyDtoHAsync, libcuda), CUresult, (Ptr{Cvoid}, CUdeviceptr, Csize_t, CUstream,), dstHost, srcDevice, ByteCount, hStream)
end

# cuMemcpyDtoDAsync() available since CUDA 3.2
function cuMemcpyDtoDAsync(dstDevice::CUdeviceptr, srcDevice::CUdeviceptr, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyDtoDAsync, libcuda), CUresult, (CUdeviceptr, CUdeviceptr, Csize_t, CUstream,), dstDevice, srcDevice, ByteCount, hStream)
end

# cuMemcpyHtoAAsync() available since CUDA 3.2
function cuMemcpyHtoAAsync(dstArray::CUarray, dstOffset::Csize_t, srcHost::Ptr{Cvoid}, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyHtoAAsync, libcuda), CUresult, (CUarray, Csize_t, Ptr{Cvoid}, Csize_t, CUstream,), dstArray, dstOffset, srcHost, ByteCount, hStream)
end

# cuMemcpyAtoHAsync() available since CUDA 3.2
function cuMemcpyAtoHAsync(dstHost::Ptr{Cvoid}, srcArray::CUarray, srcOffset::Csize_t, ByteCount::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemcpyAtoHAsync, libcuda), CUresult, (Ptr{Cvoid}, CUarray, Csize_t, Csize_t, CUstream,), dstHost, srcArray, srcOffset, ByteCount, hStream)
end

# cuMemcpy2DAsync() available since CUDA 3.2
function cuMemcpy2DAsync(pCopy::Array{CUDA_MEMCPY2D, 1}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy2DAsync, libcuda), CUresult, (Ref{CUDA_MEMCPY2D}, CUstream,), Base.cconvert(Ref{CUDA_MEMCPY2D}, pCopy), hStream)
end

function cuMemcpy2DAsync(pCopy::Ptr{CUDA_MEMCPY2D}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy2DAsync, libcuda), CUresult, (Ptr{CUDA_MEMCPY2D}, CUstream,), pCopy, hStream)
end

# cuMemcpy3DAsync() available since CUDA 3.2
function cuMemcpy3DAsync(pCopy::Array{CUDA_MEMCPY3D, 1}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy3DAsync, libcuda), CUresult, (Ref{CUDA_MEMCPY3D}, CUstream,), Base.cconvert(Ref{CUDA_MEMCPY3D}, pCopy), hStream)
end

function cuMemcpy3DAsync(pCopy::Ptr{CUDA_MEMCPY3D}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy3DAsync, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D}, CUstream,), pCopy, hStream)
end

# cuMemcpy3DPeerAsync() available since CUDA 4.0
function cuMemcpy3DPeerAsync(pCopy::Array{CUDA_MEMCPY3D_PEER, 1}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy3DPeerAsync, libcuda), CUresult, (Ref{CUDA_MEMCPY3D_PEER}, CUstream,), Base.cconvert(Ref{CUDA_MEMCPY3D_PEER}, pCopy), hStream)
end

function cuMemcpy3DPeerAsync(pCopy::Ptr{CUDA_MEMCPY3D_PEER}, hStream::CUstream)::CUresult
    return ccall((:cuMemcpy3DPeerAsync, libcuda), CUresult, (Ptr{CUDA_MEMCPY3D_PEER}, CUstream,), pCopy, hStream)
end

# cuMemsetD8() available since CUDA 3.2
function cuMemsetD8(dstDevice::CUdeviceptr, uc::UInt8, N::Csize_t)::CUresult
    return ccall((:cuMemsetD8, libcuda), CUresult, (CUdeviceptr, UInt8, Csize_t,), dstDevice, uc, N)
end

# cuMemsetD16() available since CUDA 3.2
function cuMemsetD16(dstDevice::CUdeviceptr, us::UInt16, N::Csize_t)::CUresult
    return ccall((:cuMemsetD16, libcuda), CUresult, (CUdeviceptr, UInt16, Csize_t,), dstDevice, us, N)
end

# cuMemsetD32() available since CUDA 3.2
function cuMemsetD32(dstDevice::CUdeviceptr, ui::Cuint, N::Csize_t)::CUresult
    return ccall((:cuMemsetD32, libcuda), CUresult, (CUdeviceptr, Cuint, Csize_t,), dstDevice, ui, N)
end

# cuMemsetD2D8() available since CUDA 3.2
function cuMemsetD2D8(dstDevice::CUdeviceptr, dstPitch::Csize_t, uc::UInt8, Width::Csize_t, Height::Csize_t)::CUresult
    return ccall((:cuMemsetD2D8, libcuda), CUresult, (CUdeviceptr, Csize_t, UInt8, Csize_t, Csize_t,), dstDevice, dstPitch, uc, Width, Height)
end

# cuMemsetD2D16() available since CUDA 3.2
function cuMemsetD2D16(dstDevice::CUdeviceptr, dstPitch::Csize_t, us::UInt16, Width::Csize_t, Height::Csize_t)::CUresult
    return ccall((:cuMemsetD2D16, libcuda), CUresult, (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t,), dstDevice, dstPitch, us, Width, Height)
end

# cuMemsetD2D16() available since CUDA 3.2
function cuMemsetD2D32(dstDevice::CUdeviceptr, dstPitch::Csize_t, ui::Cuint, Width::Csize_t, Height::Csize_t)::CUresult
    return ccall((:cuMemsetD2D32, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuint, Csize_t, Csize_t,), dstDevice, dstPitch, ui, Width, Height)
end

# cuMemsetD8Async() available since CUDA 3.2
function cuMemsetD8Async(dstDevice::CUdeviceptr, uc::UInt8, N::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD8Async, libcuda), CUresult, (CUdeviceptr, UInt8, Csize_t, CUstream,), dstDevice, uc, N, hStream)
end

# cuMemsetD16Async() available since CUDA 3.2
function cuMemsetD16Async(dstDevice::CUdeviceptr, us::UInt16, N::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD16Async, libcuda), CUresult, (CUdeviceptr, UInt16, Csize_t, CUstream,), dstDevice, us, N, hStream)
end

# cuMemsetD32Async() available since CUDA 3.2
function cuMemsetD32Async(dstDevice::CUdeviceptr, ui::Cuint, N::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD32Async, libcuda), CUresult, (CUdeviceptr, Cuint, Csize_t, CUstream,), dstDevice, ui, N, hStream)
end

# cuMemsetD2D8Async() available since CUDA 3.2
function cuMemsetD2D8Async(dstDevice::CUdeviceptr, dstPitch::Csize_t, uc::UInt8, Width::Csize_t, Height::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD2D8Async, libcuda), CUresult, (CUdeviceptr, Csize_t, UInt8, Csize_t, Csize_t, CUstream,), dstDevice, dstPitch, uc, Width, Height, hStream)
end

# cuMemsetD2D16Async() available since CUDA 3.2
function cuMemsetD2D16Async(dstDevice::CUdeviceptr, dstPitch::Csize_t, us::UInt16, Width::Csize_t, Height::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD2D16Async, libcuda), CUresult, (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t, CUstream,), dstDevice, dstPitch, us, Width, Height, hStream)
end

# cuMemsetD2D32Async() available since CUDA 3.2
function cuMemsetD2D32Async(dstDevice::CUdeviceptr, dstPitch::Csize_t, ui::Cuint, Width::Csize_t, Height::Csize_t, hStream::CUstream)::CUresult
    return ccall((:cuMemsetD2D32Async, libcuda), CUresult, (CUdeviceptr, Csize_t, Cuint, Csize_t, Csize_t, CUstream,), dstDevice, dstPitch, ui, Width, Height, hStream)
end

# cuArrayCreate() available since CUDA 3.2
function cuArrayCreate(pHandle::Array{CUarray, 1}, pAllocateArray::Array{CUDA_ARRAY_DESCRIPTOR, 1})::CUresult
    return ccall((:cuArrayCreate, libcuda), CUresult, (Ref{CUarray}, Ref{CUDA_ARRAY_DESCRIPTOR},), Base.cconvert(Ref{CUarray}, pHandle), Base.cconvert(Ref{CUDA_ARRAY_DESCRIPTOR}, pAllocateArray))
end

function cuArrayCreate(pHandle::Ptr{CUarray}, pAllocateArray::Ptr{CUDA_ARRAY_DESCRIPTOR})::CUresult
    return ccall((:cuArrayCreate, libcuda), CUresult, (Ptr{CUarray}, Ptr{CUDA_ARRAY_DESCRIPTOR},), pHandle, pAllocateArray)
end

# cuArrayGetDescriptor() available since CUDA 3.2
function cuArrayGetDescriptor(pArrayDescriptor::Array{CUDA_ARRAY_DESCRIPTOR, 1}, hArray::CUarray)::CUresult
    return ccall((:cuArrayGetDescriptor, libcuda), CUresult, (Ref{CUDA_ARRAY_DESCRIPTOR}, CUarray,), Base.cconvert(Ref{CUDA_ARRAY_DESCRIPTOR}, pArrayDescriptor), hArray)
end

function cuArrayGetDescriptor(pArrayDescriptor::Ptr{CUDA_ARRAY_DESCRIPTOR}, hArray::CUarray)::CUresult
    return ccall((:cuArrayGetDescriptor, libcuda), CUresult, (Ptr{CUDA_ARRAY_DESCRIPTOR}, CUarray,), pArrayDescriptor, hArray)
end

function cuArrayDestroy(hArray::CUarray)::CUresult
    return ccall((:cuArrayDestroy, libcuda), CUresult, (CUarray,), hArray)
end

# cuArray3DCreate() available since CUDA 3.2
function cuArray3DCreate(pHandle::Array{CUarray, 1}, pAllocateArray::Array{CUDA_ARRAY3D_DESCRIPTOR, 1})::CUresult
    return ccall((:cuArray3DCreate, libcuda), CUresult, (Ref{CUarray}, Ref{CUDA_ARRAY3D_DESCRIPTOR},), Base.cconvert(Ref{CUarray}, pHandle), Base.cconvert(Ref{CUDA_ARRAY3D_DESCRIPTOR}, pAllocateArray))
end

function cuArray3DCreate(pHandle::Ptr{CUarray}, pAllocateArray::Ptr{CUDA_ARRAY3D_DESCRIPTOR})::CUresult
    return ccall((:cuArray3DCreate, libcuda), CUresult, (Ptr{CUarray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR},), pHandle, pAllocateArray)
end

# cuArray3DGetDescriptor() available since CUDA 3.2
function cuArray3DGetDescriptor(pArrayDescriptor::Array{CUDA_ARRAY3D_DESCRIPTOR, 1}, hArray::CUarray)::CUresult
    return ccall((:cuArray3DGetDescriptor, libcuda), CUresult, (Ref{CUDA_ARRAY3D_DESCRIPTOR}, CUarray,), Base.cconvert(Ref{CUDA_ARRAY3D_DESCRIPTOR}, pArrayDescriptor), hArray)
end

function cuArray3DGetDescriptor(pArrayDescriptor::Ptr{CUDA_ARRAY3D_DESCRIPTOR}, hArray::CUarray)::CUresult
    return ccall((:cuArray3DGetDescriptor, libcuda), CUresult, (Ptr{CUDA_ARRAY3D_DESCRIPTOR}, CUarray,), pArrayDescriptor, hArray)
end

# cuMipmappedArrayCreate() available since CUDA 5.0
function cuMipmappedArrayCreate(pHandle::Array{CUmipmappedArray, 1}, pMipmappedArrayDesc::Array{CUDA_ARRAY3D_DESCRIPTOR, 1}, numMipmapLevels::Cuint)::CUresult
    return ccall((:cuMipmappedArrayCreate, libcuda), CUresult, (Ref{CUmipmappedArray}, Ref{CUDA_ARRAY3D_DESCRIPTOR}, Cuint,), Base.cconvert(Ref{CUmipmappedArray}, pHandle), Base.cconvert(Ref{CUDA_ARRAY3D_DESCRIPTOR}, pMipmappedArrayDesc), numMipmapLevels)
end

function cuMipmappedArrayCreate(pHandle::Ptr{CUmipmappedArray}, pMipmappedArrayDesc::Ptr{CUDA_ARRAY3D_DESCRIPTOR}, numMipmapLevels::Cuint)::CUresult
    return ccall((:cuMipmappedArrayCreate, libcuda), CUresult, (Ptr{CUmipmappedArray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}, Cuint,), pHandle, pMipmappedArrayDesc, numMipmapLevels)
end

# cuMipmappedArrayGetLevel() available since CUDA 5.0
function cuMipmappedArrayGetLevel(pLevelArray::Array{CUarray, 1}, hMipmappedArray::CUmipmappedArray, level::Cuint)::CUresult
    return ccall((:cuMipmappedArrayGetLevel, libcuda), CUresult, (Ref{CUarray}, CUmipmappedArray, Cuint,), Base.cconvert(Ref{CUarray}, pLevelArray), hMipmappedArray, level)
end

function cuMipmappedArrayGetLevel(pLevelArray::Ptr{CUarray}, hMipmappedArray::CUmipmappedArray, level::Cuint)::CUresult
    return ccall((:cuMipmappedArrayGetLevel, libcuda), CUresult, (Ptr{CUarray}, CUmipmappedArray, Cuint,), pLevelArray, hMipmappedArray, level)
end

# cuMipmappedArrayDestroy() available since CUDA 5.0
function cuMipmappedArrayDestroy(hMipmappedArray::CUmipmappedArray)::CUresult
    return ccall((:cuMipmappedArrayDestroy, libcuda), CUresult, (CUmipmappedArray,), hMipmappedArray)
end

# cuPointerGetAttribute() available since CUDA 4.0
function cuPointerGetAttribute(data::Ptr{Cvoid}, attribute::CUpointer_attribute, ptr::CUdeviceptr)::CUresult
    return ccall((:cuPointerGetAttribute, libcuda), CUresult, (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr,), data, attribute, ptr)
end

# cuMemPrefetchAsync() available since CUDA 8.0
function cuMemPrefetchAsync(devPtr::CUdeviceptr, count::Csize_t, dstDevice::CUdevice, hStream::CUstream)::CUresult
    return ccall((:cuMemPrefetchAsync, libcuda), CUresult, (CUdeviceptr, Csize_t, CUdevice, CUstream,), devPtr, count, dstDevice, hStream)
end

# cuMemAdvise() available since CUDA 8.0
function cuMemAdvise(devPtr::CUdeviceptr, count::Csize_t, advice::CUmem_advise, device::CUdevice)::CUresult
    return ccall((:cuMemAdvise, libcuda), CUresult, (CUdeviceptr, Csize_t, CUmem_advise, CUdevice,), devPtr, count, advice, device)
end

# cuMemRangeGetAttribute() available since CUDA 8.0
function cuMemRangeGetAttribute(data::Ptr{Cvoid}, dataSize::Csize_t, attribute::CUmem_range_attribute, devPtr::CUdeviceptr, count::Csize_t)::CUresult
    return ccall((:cuMemRangeGetAttribute, libcuda), CUresult, (Ptr{Cvoid}, Csize_t, CUmem_range_attribute, CUdeviceptr, Csize_t,), data, dataSize, attribute, devPtr, count)
end

# cuMemRangeGetAttributes() available since CUDA 8.0
function cuMemRangeGetAttributes(data::Array{Ptr{Cvoid}, 1}, dataSizes::Array{Csize_t, 1}, attributes::Array{CUmem_range_attribute, 1}, numAttributes::Csize_t, devPtr::CUdeviceptr, count::Csize_t)::CUresult
    return ccall((:cuMemRangeGetAttributes, libcuda), CUresult, (Ref{Ptr{Cvoid}}, Ref{Csize_t}, Ref{CUmem_range_attribute}, Csize_t, CUdeviceptr, Csize_t,), Base.cconvert(Ref{Ptr{Cvoid}}, data), Base.cconvert(Ref{Csize_t}, dataSizes), Base.cconvert(Ref{CUmem_range_attribute}, attributes), numAttributes, devPtr, count)
end

function cuMemRangeGetAttributes(data::Ptr{Ptr{Cvoid}}, dataSizes::Ptr{Csize_t}, attributes::Ptr{CUmem_range_attribute}, numAttributes::Csize_t, devPtr::CUdeviceptr, count::Csize_t)::CUresult
    return ccall((:cuMemRangeGetAttributes, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{CUmem_range_attribute}, Csize_t, CUdeviceptr, Csize_t,), data, dataSizes, attributes, numAttributes, devPtr, count)
end

# cuPointerSetAttribute() available since CUDA 6.0
function cuPointerSetAttribute(value::Ptr{Cvoid}, attribute::CUpointer_attribute, ptr::CUdeviceptr)::CUresult
    return ccall((:cuPointerSetAttribute, libcuda), CUresult, (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr,), value, attribute, ptr)
end

# cuPointerGetAttributes() available since CUDA 7.0
function cuPointerGetAttributes(numAttributes::Cuint, attributes::Array{CUpointer_attribute, 1}, data::Array{Ptr{Cvoid}, 1}, ptr::CUdeviceptr)::CUresult
    return ccall((:cuPointerGetAttributes, libcuda), CUresult, (Cuint, Ref{CUpointer_attribute}, Ref{Ptr{Cvoid}}, CUdeviceptr,), numAttributes, Base.cconvert(Ref{CUpointer_attribute}, attributes), Base.cconvert(Ref{Ptr{Cvoid}}, data), ptr)
end

function cuPointerGetAttributes(numAttributes::Cuint, attributes::Ptr{CUpointer_attribute}, data::Ptr{Ptr{Cvoid}}, ptr::CUdeviceptr)::CUresult
    return ccall((:cuPointerGetAttributes, libcuda), CUresult, (Cuint, Ptr{CUpointer_attribute}, Ptr{Ptr{Cvoid}}, CUdeviceptr,), numAttributes, attributes, data, ptr)
end

function cuStreamCreate(phStream::Array{CUstream, 1}, Flags::Cuint)::CUresult
    return ccall((:cuStreamCreate, libcuda), CUresult, (Ref{CUstream}, Cuint,), Base.cconvert(Ref{CUstream}, phStream), Flags)
end

function cuStreamCreate(phStream::Ptr{CUstream}, Flags::Cuint)::CUresult
    return ccall((:cuStreamCreate, libcuda), CUresult, (Ptr{CUstream}, Cuint,), phStream, Flags)
end

function cuStreamCreateWithPriority(phStream::Array{CUstream, 1}, flags::Cuint, priority::Cint)::CUresult
    return ccall((:cuStreamCreateWithPriority, libcuda), CUresult, (Ref{CUstream}, Cuint, Cint,), Base.cconvert(Ref{CUstream}, phStream), flags, priority)
end

function cuStreamCreateWithPriority(phStream::Ptr{CUstream}, flags::Cuint, priority::Cint)::CUresult
    return ccall((:cuStreamCreateWithPriority, libcuda), CUresult, (Ptr{CUstream}, Cuint, Cint,), phStream, flags, priority)
end

function cuStreamGetPriority(hStream::CUstream, priority::Array{Cint, 1})::CUresult
    return ccall((:cuStreamGetPriority, libcuda), CUresult, (CUstream, Ref{Cint},), hStream, Base.cconvert(Ref{Cint}, priority))
end

function cuStreamGetPriority(hStream::CUstream, priority::Ptr{Cint})::CUresult
    return ccall((:cuStreamGetPriority, libcuda), CUresult, (CUstream, Ptr{Cint},), hStream, priority)
end

function cuStreamGetFlags(hStream::CUstream, flags::Array{Cuint, 1})::CUresult
    return ccall((:cuStreamGetFlags, libcuda), CUresult, (CUstream, Ref{Cuint},), hStream, Base.cconvert(Ref{Cuint}, flags))
end

function cuStreamGetFlags(hStream::CUstream, flags::Ptr{Cuint})::CUresult
    return ccall((:cuStreamGetFlags, libcuda), CUresult, (CUstream, Ptr{Cuint},), hStream, flags)
end

# cuStreamGetCtx() available since CUDA 9.2
function cuStreamGetCtx(hStream::CUstream, pctx::Array{CUcontext, 1})::CUresult
    return ccall((:cuStreamGetCtx, libcuda), CUresult, (CUstream, Ref{CUcontext},), hStream, Base.cconvert(Ref{CUcontext}, pctx))
end

function cuStreamGetCtx(hStream::CUstream, pctx::Ptr{CUcontext})::CUresult
    return ccall((:cuStreamGetCtx, libcuda), CUresult, (CUstream, Ptr{CUcontext},), hStream, pctx)
end

function cuStreamWaitEvent(hStream::CUstream, hEvent::CUevent, Flags::Cuint)::CUresult
    return ccall((:cuStreamWaitEvent, libcuda), CUresult, (CUstream, CUevent, Cuint,), hStream, hEvent, Flags)
end

function cuStreamAddCallback(hStream::CUstream, callback::CUstreamCallback, userData::Ptr{Cvoid}, flags::Cuint)::CUresult
    return ccall((:cuStreamAddCallback, libcuda), CUresult, (CUstream, CUstreamCallback, Ptr{Cvoid}, Cuint,), hStream, callback, userData, flags)
end

# cuStreamBeginCapture() available since CUDA 10.0
function cuStreamBeginCapture(hStream::CUstream)::CUresult
    return ccall((:cuStreamBeginCapture, libcuda), CUresult, (CUstream,), hStream)
end

# cuThreadExchangeStreamCaptureMode() available since CUDA 10.1
function cuThreadExchangeStreamCaptureMode(mode::Array{CUstreamCaptureMode, 1})::CUresult
    return ccall((:cuThreadExchangeStreamCaptureMode, libcuda), CUresult, (Ref{CUstreamCaptureMode},), Base.cconvert(Ref{CUstreamCaptureMode}, mode))
end

function cuThreadExchangeStreamCaptureMode(mode::Ptr{CUstreamCaptureMode})::CUresult
    return ccall((:cuThreadExchangeStreamCaptureMode, libcuda), CUresult, (Ptr{CUstreamCaptureMode},), mode)
end

# cuStreamEndCapture() available since CUDA 10.0
function cuStreamEndCapture(hStream::CUstream, phGraph::Array{CUgraph, 1})::CUresult
    return ccall((:cuStreamEndCapture, libcuda), CUresult, (CUstream, Ref{CUgraph},), hStream, Base.cconvert(Ref{CUgraph}, phGraph))
end

function cuStreamEndCapture(hStream::CUstream, phGraph::Ptr{CUgraph})::CUresult
    return ccall((:cuStreamEndCapture, libcuda), CUresult, (CUstream, Ptr{CUgraph},), hStream, phGraph)
end

# cuStreamIsCapturing() available since CUDA 10.0
function cuStreamIsCapturing(hStream::CUstream, captureStatus::Array{CUstreamCaptureStatus, 1})::CUresult
    return ccall((:cuStreamIsCapturing, libcuda), CUresult, (CUstream, Ref{CUstreamCaptureStatus},), hStream, Base.cconvert(Ref{CUstreamCaptureStatus}, captureStatus))
end

function cuStreamIsCapturing(hStream::CUstream, captureStatus::Ptr{CUstreamCaptureStatus})::CUresult
    return ccall((:cuStreamIsCapturing, libcuda), CUresult, (CUstream, Ptr{CUstreamCaptureStatus},), hStream, captureStatus)
end

# cuStreamGetCaptureInfo() available since CUDA 10.1
function cuStreamGetCaptureInfo(hStream::CUstream, captureStatus::Array{CUstreamCaptureStatus, 1}, id::Array{cuuint64_t, 1})::CUresult
    return ccall((:cuStreamGetCaptureInfo, libcuda), CUresult, (CUstream, Ref{CUstreamCaptureStatus}, Ref{cuuint64_t},), hStream, Base.cconvert(Ref{CUstreamCaptureStatus}, captureStatus), Base.cconvert(Ref{cuuint64_t}, id))
end

function cuStreamGetCaptureInfo(hStream::CUstream, captureStatus::Ptr{CUstreamCaptureStatus}, id::Ptr{cuuint64_t})::CUresult
    return ccall((:cuStreamGetCaptureInfo, libcuda), CUresult, (CUstream, Ptr{CUstreamCaptureStatus}, Ptr{cuuint64_t},), hStream, captureStatus, id)
end

# cuStreamAttachMemAsync() available since CUDA 6.0
function cuStreamAttachMemAsync(hStream::CUstream, dptr::CUdeviceptr, length::Csize_t, flags::Cuint)::CUresult
    return ccall((:cuStreamAttachMemAsync, libcuda), CUresult, (CUstream, CUdeviceptr, Csize_t, Cuint,), hStream, dptr, length, flags)
end

function cuStreamQuery(hStream::CUstream)::CUresult
    return ccall((:cuStreamQuery, libcuda), CUresult, (CUstream,), hStream)
end

function cuStreamSynchronize(hStream::CUstream)::CUresult
    return ccall((:cuStreamSynchronize, libcuda), CUresult, (CUstream,), hStream)
end

# cuStreamDestroy() available since CUDA 4.0
function cuStreamDestroy(hStream::CUstream)::CUresult
    return ccall((:cuStreamDestroy, libcuda), CUresult, (CUstream,), hStream)
end

function cuEventCreate(phEvent::Array{CUevent, 1}, Flags::Cuint)::CUresult
    return ccall((:cuEventCreate, libcuda), CUresult, (Ref{CUevent}, Cuint,), Base.cconvert(Ref{CUevent}, phEvent), Flags)
end

function cuEventCreate(phEvent::Ptr{CUevent}, Flags::Cuint)::CUresult
    return ccall((:cuEventCreate, libcuda), CUresult, (Ptr{CUevent}, Cuint,), phEvent, Flags)
end

function cuEventRecord(hEvent::CUevent, hStream::CUstream)::CUresult
    return ccall((:cuEventRecord, libcuda), CUresult, (CUevent, CUstream,), hEvent, hStream)
end

function cuEventQuery(hEvent::CUevent)::CUresult
    return ccall((:cuEventQuery, libcuda), CUresult, (CUevent,), hEvent)
end

function cuEventSynchronize(hEvent::CUevent)::CUresult
    return ccall((:cuEventSynchronize, libcuda), CUresult, (CUevent,), hEvent)
end

# cuEventDestroy() available since CUDA 4.0
function cuEventDestroy(hEvent::CUevent)::CUresult
    return ccall((:cuEventDestroy, libcuda), CUresult, (CUevent,), hEvent)
end

function cuEventElapsedTime(pMilliseconds::Array{Cfloat, 1}, hStart::CUevent, hEnd::CUevent)::CUresult
    return ccall((:cuEventElapsedTime, libcuda), CUresult, (Ref{Cfloat}, CUevent, CUevent,), Base.cconvert(Ref{Cfloat}, pMilliseconds), hStart, hEnd)
end

function cuEventElapsedTime(pMilliseconds::Ptr{Cfloat}, hStart::CUevent, hEnd::CUevent)::CUresult
    return ccall((:cuEventElapsedTime, libcuda), CUresult, (Ptr{Cfloat}, CUevent, CUevent,), pMilliseconds, hStart, hEnd)
end

# cuImportExternalMemory() available since CUDA 10.0
function cuImportExternalMemory(extMem_out::Array{CUexternalMemory, 1}, memHandleDesc::Array{CUDA_EXTERNAL_MEMORY_HANDLE_DESC, 1})::CUresult
    return ccall((:cuImportExternalMemory, libcuda), CUresult, (Ref{CUexternalMemory}, Ref{CUDA_EXTERNAL_MEMORY_HANDLE_DESC},), Base.cconvert(Ref{CUexternalMemory}, extMem_out), Base.cconvert(Ref{CUDA_EXTERNAL_MEMORY_HANDLE_DESC}, memHandleDesc))
end

function cuImportExternalMemory(extMem_out::Ptr{CUexternalMemory}, memHandleDesc::Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC})::CUresult
    return ccall((:cuImportExternalMemory, libcuda), CUresult, (Ptr{CUexternalMemory}, Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC},), extMem_out, memHandleDesc)
end

# cuExternalMemoryGetMappedBuffer() available since CUDA 10.0
function cuExternalMemoryGetMappedBuffer(devPtr::Array{CUdeviceptr, 1}, extMem::CUexternalMemory, bufferDesc::Array{CUDA_EXTERNAL_MEMORY_BUFFER_DESC, 1})::CUresult
    return ccall((:cuExternalMemoryGetMappedBuffer, libcuda), CUresult, (Ref{CUdeviceptr}, CUexternalMemory, Ref{CUDA_EXTERNAL_MEMORY_BUFFER_DESC},), Base.cconvert(Ref{CUdeviceptr}, devPtr), extMem, Base.cconvert(Ref{CUDA_EXTERNAL_MEMORY_BUFFER_DESC}, bufferDesc))
end

function cuExternalMemoryGetMappedBuffer(devPtr::Ptr{CUdeviceptr}, extMem::CUexternalMemory, bufferDesc::Ptr{CUDA_EXTERNAL_MEMORY_BUFFER_DESC})::CUresult
    return ccall((:cuExternalMemoryGetMappedBuffer, libcuda), CUresult, (Ptr{CUdeviceptr}, CUexternalMemory, Ptr{CUDA_EXTERNAL_MEMORY_BUFFER_DESC},), devPtr, extMem, bufferDesc)
end

# cuExternalMemoryGetMappedMipmappedArray() available since CUDA 10.0
function cuExternalMemoryGetMappedMipmappedArray(mipmap::Array{CUmipmappedArray, 1}, extMem::CUexternalMemory, mipmapDesc::Array{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, 1})::CUresult
    return ccall((:cuExternalMemoryGetMappedMipmappedArray, libcuda), CUresult, (Ref{CUmipmappedArray}, CUexternalMemory, Ref{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC},), Base.cconvert(Ref{CUmipmappedArray}, mipmap), extMem, Base.cconvert(Ref{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC}, mipmapDesc))
end

function cuExternalMemoryGetMappedMipmappedArray(mipmap::Ptr{CUmipmappedArray}, extMem::CUexternalMemory, mipmapDesc::Ptr{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC})::CUresult
    return ccall((:cuExternalMemoryGetMappedMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUexternalMemory, Ptr{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC},), mipmap, extMem, mipmapDesc)
end

# cuDestroyExternalMemory() available since CUDA 10.0
function cuDestroyExternalMemory(extMem::CUexternalMemory)::CUresult
    return ccall((:cuDestroyExternalMemory, libcuda), CUresult, (CUexternalMemory,), extMem)
end

# cuImportExternalSemaphore() available since CUDA 10.0
function cuImportExternalSemaphore(extSem_out::Array{CUexternalSemaphore, 1}, semHandleDesc::Array{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, 1})::CUresult
    return ccall((:cuImportExternalSemaphore, libcuda), CUresult, (Ref{CUexternalSemaphore}, Ref{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC},), Base.cconvert(Ref{CUexternalSemaphore}, extSem_out), Base.cconvert(Ref{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC}, semHandleDesc))
end

function cuImportExternalSemaphore(extSem_out::Ptr{CUexternalSemaphore}, semHandleDesc::Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC})::CUresult
    return ccall((:cuImportExternalSemaphore, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC},), extSem_out, semHandleDesc)
end

# cuSignalExternalSemaphoresAsync() available since CUDA 10.0
function cuSignalExternalSemaphoresAsync(extSemArray::Array{CUexternalSemaphore, 1}, paramsArray::Array{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, 1}, numExtSems::Cuint, stream::CUstream)::CUresult
    return ccall((:cuSignalExternalSemaphoresAsync, libcuda), CUresult, (Ref{CUexternalSemaphore}, Ref{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}, Cuint, CUstream,), Base.cconvert(Ref{CUexternalSemaphore}, extSemArray), Base.cconvert(Ref{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}, paramsArray), numExtSems, stream)
end

function cuSignalExternalSemaphoresAsync(extSemArray::Ptr{CUexternalSemaphore}, paramsArray::Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}, numExtSems::Cuint, stream::CUstream)::CUresult
    return ccall((:cuSignalExternalSemaphoresAsync, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS}, Cuint, CUstream,), extSemArray, paramsArray, numExtSems, stream)
end

# cuWaitExternalSemaphoresAsync() available since CUDA 10.0
function cuWaitExternalSemaphoresAsync(extSemArray::Array{CUexternalSemaphore, 1}, paramsArray::Array{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, 1}, numExtSems::Cuint, stream::CUstream)::CUresult
    return ccall((:cuWaitExternalSemaphoresAsync, libcuda), CUresult, (Ref{CUexternalSemaphore}, Ref{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}, Cuint, CUstream,), Base.cconvert(Ref{CUexternalSemaphore}, extSemArray), Base.cconvert(Ref{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}, paramsArray), numExtSems, stream)
end

function cuWaitExternalSemaphoresAsync(extSemArray::Ptr{CUexternalSemaphore}, paramsArray::Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}, numExtSems::Cuint, stream::CUstream)::CUresult
    return ccall((:cuWaitExternalSemaphoresAsync, libcuda), CUresult, (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS}, Cuint, CUstream,), extSemArray, paramsArray, numExtSems, stream)
end

# cuDestroyExternalSemaphore() available since CUDA 10.0
function cuDestroyExternalSemaphore(extSem::CUexternalSemaphore)::CUresult
    return ccall((:cuDestroyExternalSemaphore, libcuda), CUresult, (CUexternalSemaphore,), extSem)
end

# cuStreamWaitValue32() available since CUDA 8.0
function cuStreamWaitValue32(stream::CUstream, addr::CUdeviceptr, value::cuuint32_t, flags::Cuint)::CUresult
    return ccall((:cuStreamWaitValue32, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint,), stream, addr, value, flags)
end

# cuStreamWaitValue64() available since CUDA 8.0
function cuStreamWaitValue64(stream::CUstream, addr::CUdeviceptr, value::cuuint64_t, flags::Cuint)::CUresult
    return ccall((:cuStreamWaitValue64, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint,), stream, addr, value, flags)
end

# cuStreamWriteValue32() available since CUDA 8.0
function cuStreamWriteValue32(stream::CUstream, addr::CUdeviceptr, value::cuuint32_t, flags::Cuint)::CUresult
    return ccall((:cuStreamWriteValue32, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint32_t, Cuint,), stream, addr, value, flags)
end

# cuStreamWriteValue64() available since CUDA 8.0
function cuStreamWriteValue64(stream::CUstream, addr::CUdeviceptr, value::cuuint64_t, flags::Cuint)::CUresult
    return ccall((:cuStreamWriteValue64, libcuda), CUresult, (CUstream, CUdeviceptr, cuuint64_t, Cuint,), stream, addr, value, flags)
end

# cuStreamBatchMemOp() available since CUDA 8.0
function cuStreamBatchMemOp(stream::CUstream, count::Cuint, paramArray::Array{CUstreamBatchMemOpParams, 1}, flags::Cuint)::CUresult
    return ccall((:cuStreamBatchMemOp, libcuda), CUresult, (CUstream, Cuint, Ref{CUstreamBatchMemOpParams}, Cuint,), stream, count, Base.cconvert(Ref{CUstreamBatchMemOpParams}, paramArray), flags)
end

function cuStreamBatchMemOp(stream::CUstream, count::Cuint, paramArray::Ptr{CUstreamBatchMemOpParams}, flags::Cuint)::CUresult
    return ccall((:cuStreamBatchMemOp, libcuda), CUresult, (CUstream, Cuint, Ptr{CUstreamBatchMemOpParams}, Cuint,), stream, count, paramArray, flags)
end

function cuFuncGetAttribute(pi::Array{Cint, 1}, attrib::CUfunction_attribute, hfunc::CUfunction)::CUresult
    return ccall((:cuFuncGetAttribute, libcuda), CUresult, (Ref{Cint}, CUfunction_attribute, CUfunction,), Base.cconvert(Ref{Cint}, pi), attrib, hfunc)
end

function cuFuncGetAttribute(pi::Ptr{Cint}, attrib::CUfunction_attribute, hfunc::CUfunction)::CUresult
    return ccall((:cuFuncGetAttribute, libcuda), CUresult, (Ptr{Cint}, CUfunction_attribute, CUfunction,), pi, attrib, hfunc)
end

# cuFuncSetAttribute() available since CUDA 9.0
function cuFuncSetAttribute(hfunc::CUfunction, attrib::CUfunction_attribute, value::Cint)::CUresult
    return ccall((:cuFuncSetAttribute, libcuda), CUresult, (CUfunction, CUfunction_attribute, Cint,), hfunc, attrib, value)
end

function cuFuncSetCacheConfig(hfunc::CUfunction, config::CUfunc_cache)::CUresult
    return ccall((:cuFuncSetCacheConfig, libcuda), CUresult, (CUfunction, CUfunc_cache,), hfunc, config)
end

# cuFuncSetSharedMemConfig() available since CUDA 4.2
function cuFuncSetSharedMemConfig(hfunc::CUfunction, config::CUsharedconfig)::CUresult
    return ccall((:cuFuncSetSharedMemConfig, libcuda), CUresult, (CUfunction, CUsharedconfig,), hfunc, config)
end

# cuLaunchKernel() available since CUDA 4.0
function cuLaunchKernel(f::CUfunction, gridDimX::Cuint, gridDimY::Cuint, gridDimZ::Cuint, blockDimX::Cuint, blockDimY::Cuint, blockDimZ::Cuint, sharedMemBytes::Cuint, hStream::CUstream, kernelParams::Array{Ptr{Cvoid}, 1}, extra::Array{Ptr{Cvoid}, 1})::CUresult
    return ccall((:cuLaunchKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}},), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, Base.cconvert(Ref{Ptr{Cvoid}}, kernelParams), Base.cconvert(Ref{Ptr{Cvoid}}, extra))
end

function cuLaunchKernel(f::CUfunction, gridDimX::Cuint, gridDimY::Cuint, gridDimZ::Cuint, blockDimX::Cuint, blockDimY::Cuint, blockDimZ::Cuint, sharedMemBytes::Cuint, hStream::CUstream, kernelParams::Ptr{Ptr{Cvoid}}, extra::Ptr{Ptr{Cvoid}})::CUresult
    return ccall((:cuLaunchKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}},), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
end

# cuLaunchCooperativeKernel() available since CUDA 9.0
function cuLaunchCooperativeKernel(f::CUfunction, gridDimX::Cuint, gridDimY::Cuint, gridDimZ::Cuint, blockDimX::Cuint, blockDimY::Cuint, blockDimZ::Cuint, sharedMemBytes::Cuint, hStream::CUstream, kernelParams::Array{Ptr{Cvoid}, 1})::CUresult
    return ccall((:cuLaunchCooperativeKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ref{Ptr{Cvoid}},), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, Base.cconvert(Ref{Ptr{Cvoid}}, kernelParams))
end

function cuLaunchCooperativeKernel(f::CUfunction, gridDimX::Cuint, gridDimY::Cuint, gridDimZ::Cuint, blockDimX::Cuint, blockDimY::Cuint, blockDimZ::Cuint, sharedMemBytes::Cuint, hStream::CUstream, kernelParams::Ptr{Ptr{Cvoid}})::CUresult
    return ccall((:cuLaunchCooperativeKernel, libcuda), CUresult, (CUfunction, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, Cuint, CUstream, Ptr{Ptr{Cvoid}},), f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
end

# cuLaunchCooperativeKernelMultiDevice() available since CUDA 9.0
function cuLaunchCooperativeKernelMultiDevice(launchParamsList::Array{CUDA_LAUNCH_PARAMS, 1}, numDevices::Cuint, flags::Cuint)::CUresult
    return ccall((:cuLaunchCooperativeKernelMultiDevice, libcuda), CUresult, (Ref{CUDA_LAUNCH_PARAMS}, Cuint, Cuint,), Base.cconvert(Ref{CUDA_LAUNCH_PARAMS}, launchParamsList), numDevices, flags)
end

function cuLaunchCooperativeKernelMultiDevice(launchParamsList::Ptr{CUDA_LAUNCH_PARAMS}, numDevices::Cuint, flags::Cuint)::CUresult
    return ccall((:cuLaunchCooperativeKernelMultiDevice, libcuda), CUresult, (Ptr{CUDA_LAUNCH_PARAMS}, Cuint, Cuint,), launchParamsList, numDevices, flags)
end

# cuLaunchHostFunc() available since CUDA 10.0
function cuLaunchHostFunc(hStream::CUstream, fn::CUhostFn, userData::Ptr{Cvoid})::CUresult
    return ccall((:cuLaunchHostFunc, libcuda), CUresult, (CUstream, CUhostFn, Ptr{Cvoid},), hStream, fn, userData)
end

function cuFuncSetBlockShape(hfunc::CUfunction, x::Cint, y::Cint, z::Cint)::CUresult
    return ccall((:cuFuncSetBlockShape, libcuda), CUresult, (CUfunction, Cint, Cint, Cint,), hfunc, x, y, z)
end

function cuFuncSetSharedSize(hfunc::CUfunction, bytes::Cuint)::CUresult
    return ccall((:cuFuncSetSharedSize, libcuda), CUresult, (CUfunction, Cuint,), hfunc, bytes)
end

function cuParamSetSize(hfunc::CUfunction, numbytes::Cuint)::CUresult
    return ccall((:cuParamSetSize, libcuda), CUresult, (CUfunction, Cuint,), hfunc, numbytes)
end

function cuParamSeti(hfunc::CUfunction, offset::Cint, value::Cuint)::CUresult
    return ccall((:cuParamSeti, libcuda), CUresult, (CUfunction, Cint, Cuint,), hfunc, offset, value)
end

function cuParamSetf(hfunc::CUfunction, offset::Cint, value::Cfloat)::CUresult
    return ccall((:cuParamSetf, libcuda), CUresult, (CUfunction, Cint, Cfloat,), hfunc, offset, value)
end

function cuParamSetv(hfunc::CUfunction, offset::Cint, ptr::Ptr{Cvoid}, numbytes::Cuint)::CUresult
    return ccall((:cuParamSetv, libcuda), CUresult, (CUfunction, Cint, Ptr{Cvoid}, Cuint,), hfunc, offset, ptr, numbytes)
end

function cuLaunch(f::CUfunction)::CUresult
    return ccall((:cuLaunch, libcuda), CUresult, (CUfunction,), f)
end

function cuLaunchGrid(f::CUfunction, grid_width::Cint, grid_height::Cint)::CUresult
    return ccall((:cuLaunchGrid, libcuda), CUresult, (CUfunction, Cint, Cint,), f, grid_width, grid_height)
end

function cuLaunchGridAsync(f::CUfunction, grid_width::Cint, grid_height::Cint, hStream::CUstream)::CUresult
    return ccall((:cuLaunchGridAsync, libcuda), CUresult, (CUfunction, Cint, Cint, CUstream,), f, grid_width, grid_height, hStream)
end

function cuParamSetTexRef(hfunc::CUfunction, texunit::Cint, hTexRef::CUtexref)::CUresult
    return ccall((:cuParamSetTexRef, libcuda), CUresult, (CUfunction, Cint, CUtexref,), hfunc, texunit, hTexRef)
end

# cuGraphCreate() available since CUDA 10.0
function cuGraphCreate(phGraph::Array{CUgraph, 1}, flags::Cuint)::CUresult
    return ccall((:cuGraphCreate, libcuda), CUresult, (Ref{CUgraph}, Cuint,), Base.cconvert(Ref{CUgraph}, phGraph), flags)
end

function cuGraphCreate(phGraph::Ptr{CUgraph}, flags::Cuint)::CUresult
    return ccall((:cuGraphCreate, libcuda), CUresult, (Ptr{CUgraph}, Cuint,), phGraph, flags)
end

# cuGraphAddKernelNode() available since CUDA 10.0
function cuGraphAddKernelNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t, nodeParams::Array{CUDA_KERNEL_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphAddKernelNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t, Ref{CUDA_KERNEL_NODE_PARAMS},), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies, Base.cconvert(Ref{CUDA_KERNEL_NODE_PARAMS}, nodeParams))
end

function cuGraphAddKernelNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t, nodeParams::Ptr{CUDA_KERNEL_NODE_PARAMS})::CUresult
    return ccall((:cuGraphAddKernelNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_KERNEL_NODE_PARAMS},), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

# cuGraphKernelNodeGetParams() available since CUDA 10.0
function cuGraphKernelNodeGetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_KERNEL_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphKernelNodeGetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_KERNEL_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_KERNEL_NODE_PARAMS}, nodeParams))
end

function cuGraphKernelNodeGetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_KERNEL_NODE_PARAMS})::CUresult
    return ccall((:cuGraphKernelNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphKernelNodeSetParams() available since CUDA 10.0
function cuGraphKernelNodeSetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_KERNEL_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphKernelNodeSetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_KERNEL_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_KERNEL_NODE_PARAMS}, nodeParams))
end

function cuGraphKernelNodeSetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_KERNEL_NODE_PARAMS})::CUresult
    return ccall((:cuGraphKernelNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphAddMemcpyNode() available since CUDA 10.0
function cuGraphAddMemcpyNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t, copyParams::Array{CUDA_MEMCPY3D, 1}, ctx::CUcontext)::CUresult
    return ccall((:cuGraphAddMemcpyNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t, Ref{CUDA_MEMCPY3D}, CUcontext,), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies, Base.cconvert(Ref{CUDA_MEMCPY3D}, copyParams), ctx)
end

function cuGraphAddMemcpyNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t, copyParams::Ptr{CUDA_MEMCPY3D}, ctx::CUcontext)::CUresult
    return ccall((:cuGraphAddMemcpyNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_MEMCPY3D}, CUcontext,), phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
end

# cuGraphMemcpyNodeGetParams() available since CUDA 10.0
function cuGraphMemcpyNodeGetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_MEMCPY3D, 1})::CUresult
    return ccall((:cuGraphMemcpyNodeGetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_MEMCPY3D},), hNode, Base.cconvert(Ref{CUDA_MEMCPY3D}, nodeParams))
end

function cuGraphMemcpyNodeGetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_MEMCPY3D})::CUresult
    return ccall((:cuGraphMemcpyNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMCPY3D},), hNode, nodeParams)
end

# cuGraphMemcpyNodeSetParams() available since CUDA 10.0
function cuGraphMemcpyNodeSetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_MEMCPY3D, 1})::CUresult
    return ccall((:cuGraphMemcpyNodeSetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_MEMCPY3D},), hNode, Base.cconvert(Ref{CUDA_MEMCPY3D}, nodeParams))
end

function cuGraphMemcpyNodeSetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_MEMCPY3D})::CUresult
    return ccall((:cuGraphMemcpyNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMCPY3D},), hNode, nodeParams)
end

# cuGraphAddMemsetNode() available since CUDA 10.0
function cuGraphAddMemsetNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t, memsetParams::Array{CUDA_MEMSET_NODE_PARAMS, 1}, ctx::CUcontext)::CUresult
    return ccall((:cuGraphAddMemsetNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t, Ref{CUDA_MEMSET_NODE_PARAMS}, CUcontext,), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies, Base.cconvert(Ref{CUDA_MEMSET_NODE_PARAMS}, memsetParams), ctx)
end

function cuGraphAddMemsetNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t, memsetParams::Ptr{CUDA_MEMSET_NODE_PARAMS}, ctx::CUcontext)::CUresult
    return ccall((:cuGraphAddMemsetNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext,), phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
end

# cuGraphMemsetNodeGetParams() available since CUDA 10.0
function cuGraphMemsetNodeGetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_MEMSET_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphMemsetNodeGetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_MEMSET_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_MEMSET_NODE_PARAMS}, nodeParams))
end

function cuGraphMemsetNodeGetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_MEMSET_NODE_PARAMS})::CUresult
    return ccall((:cuGraphMemsetNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphMemsetNodeSetParams() available since CUDA 10.0
function cuGraphMemsetNodeSetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_MEMSET_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphMemsetNodeSetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_MEMSET_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_MEMSET_NODE_PARAMS}, nodeParams))
end

function cuGraphMemsetNodeSetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_MEMSET_NODE_PARAMS})::CUresult
    return ccall((:cuGraphMemsetNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphAddHostNode() available since CUDA 10.0
function cuGraphAddHostNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t, nodeParams::Array{CUDA_HOST_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphAddHostNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t, Ref{CUDA_HOST_NODE_PARAMS},), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies, Base.cconvert(Ref{CUDA_HOST_NODE_PARAMS}, nodeParams))
end

function cuGraphAddHostNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t, nodeParams::Ptr{CUDA_HOST_NODE_PARAMS})::CUresult
    return ccall((:cuGraphAddHostNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, Ptr{CUDA_HOST_NODE_PARAMS},), phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

# cuGraphHostNodeGetParams() available since CUDA 10.0
function cuGraphHostNodeGetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_HOST_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphHostNodeGetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_HOST_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_HOST_NODE_PARAMS}, nodeParams))
end

function cuGraphHostNodeGetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_HOST_NODE_PARAMS})::CUresult
    return ccall((:cuGraphHostNodeGetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphHostNodeSetParams() available since CUDA 10.0
function cuGraphHostNodeSetParams(hNode::CUgraphNode, nodeParams::Array{CUDA_HOST_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphHostNodeSetParams, libcuda), CUresult, (CUgraphNode, Ref{CUDA_HOST_NODE_PARAMS},), hNode, Base.cconvert(Ref{CUDA_HOST_NODE_PARAMS}, nodeParams))
end

function cuGraphHostNodeSetParams(hNode::CUgraphNode, nodeParams::Ptr{CUDA_HOST_NODE_PARAMS})::CUresult
    return ccall((:cuGraphHostNodeSetParams, libcuda), CUresult, (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS},), hNode, nodeParams)
end

# cuGraphAddChildGraphNode() available since CUDA 10.0
function cuGraphAddChildGraphNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t, childGraph::CUgraph)::CUresult
    return ccall((:cuGraphAddChildGraphNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t, CUgraph,), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies, childGraph)
end

function cuGraphAddChildGraphNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t, childGraph::CUgraph)::CUresult
    return ccall((:cuGraphAddChildGraphNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUgraph,), phGraphNode, hGraph, dependencies, numDependencies, childGraph)
end

# cuGraphChildGraphNodeGetGraph() available since CUDA 10.0
function cuGraphChildGraphNodeGetGraph(hNode::CUgraphNode, phGraph::Array{CUgraph, 1})::CUresult
    return ccall((:cuGraphChildGraphNodeGetGraph, libcuda), CUresult, (CUgraphNode, Ref{CUgraph},), hNode, Base.cconvert(Ref{CUgraph}, phGraph))
end

function cuGraphChildGraphNodeGetGraph(hNode::CUgraphNode, phGraph::Ptr{CUgraph})::CUresult
    return ccall((:cuGraphChildGraphNodeGetGraph, libcuda), CUresult, (CUgraphNode, Ptr{CUgraph},), hNode, phGraph)
end

# cuGraphAddEmptyNode() available since CUDA 10.0
function cuGraphAddEmptyNode(phGraphNode::Array{CUgraphNode, 1}, hGraph::CUgraph, dependencies::Array{CUgraphNode, 1}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphAddEmptyNode, libcuda), CUresult, (Ref{CUgraphNode}, CUgraph, Ref{CUgraphNode}, Csize_t,), Base.cconvert(Ref{CUgraphNode}, phGraphNode), hGraph, Base.cconvert(Ref{CUgraphNode}, dependencies), numDependencies)
end

function cuGraphAddEmptyNode(phGraphNode::Ptr{CUgraphNode}, hGraph::CUgraph, dependencies::Ptr{CUgraphNode}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphAddEmptyNode, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,), phGraphNode, hGraph, dependencies, numDependencies)
end

# cuGraphClone() available since CUDA 10.0
function cuGraphClone(phGraphClone::Array{CUgraph, 1}, originalGraph::CUgraph)::CUresult
    return ccall((:cuGraphClone, libcuda), CUresult, (Ref{CUgraph}, CUgraph,), Base.cconvert(Ref{CUgraph}, phGraphClone), originalGraph)
end

function cuGraphClone(phGraphClone::Ptr{CUgraph}, originalGraph::CUgraph)::CUresult
    return ccall((:cuGraphClone, libcuda), CUresult, (Ptr{CUgraph}, CUgraph,), phGraphClone, originalGraph)
end

# cuGraphNodeFindInClone() available since CUDA 10.0
function cuGraphNodeFindInClone(phNode::Array{CUgraphNode, 1}, hOriginalNode::CUgraphNode, hClonedGraph::CUgraph)::CUresult
    return ccall((:cuGraphNodeFindInClone, libcuda), CUresult, (Ref{CUgraphNode}, CUgraphNode, CUgraph,), Base.cconvert(Ref{CUgraphNode}, phNode), hOriginalNode, hClonedGraph)
end

function cuGraphNodeFindInClone(phNode::Ptr{CUgraphNode}, hOriginalNode::CUgraphNode, hClonedGraph::CUgraph)::CUresult
    return ccall((:cuGraphNodeFindInClone, libcuda), CUresult, (Ptr{CUgraphNode}, CUgraphNode, CUgraph,), phNode, hOriginalNode, hClonedGraph)
end

# cuGraphNodeGetType() available since CUDA 10.0
function cuGraphNodeGetType(hNode::CUgraphNode, type::Array{CUgraphNodeType, 1})::CUresult
    return ccall((:cuGraphNodeGetType, libcuda), CUresult, (CUgraphNode, Ref{CUgraphNodeType},), hNode, Base.cconvert(Ref{CUgraphNodeType}, type))
end

function cuGraphNodeGetType(hNode::CUgraphNode, type::Ptr{CUgraphNodeType})::CUresult
    return ccall((:cuGraphNodeGetType, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNodeType},), hNode, type)
end

# cuGraphGetNodes() available since CUDA 10.0
function cuGraphGetNodes(hGraph::CUgraph, nodes::Array{CUgraphNode, 1}, numNodes::Array{Csize_t, 1})::CUresult
    return ccall((:cuGraphGetNodes, libcuda), CUresult, (CUgraph, Ref{CUgraphNode}, Ref{Csize_t},), hGraph, Base.cconvert(Ref{CUgraphNode}, nodes), Base.cconvert(Ref{Csize_t}, numNodes))
end

function cuGraphGetNodes(hGraph::CUgraph, nodes::Ptr{CUgraphNode}, numNodes::Ptr{Csize_t})::CUresult
    return ccall((:cuGraphGetNodes, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t},), hGraph, nodes, numNodes)
end

# cuGraphGetRootNodes() available since CUDA 10.0
function cuGraphGetRootNodes(hGraph::CUgraph, rootNodes::Array{CUgraphNode, 1}, numRootNodes::Array{Csize_t, 1})::CUresult
    return ccall((:cuGraphGetRootNodes, libcuda), CUresult, (CUgraph, Ref{CUgraphNode}, Ref{Csize_t},), hGraph, Base.cconvert(Ref{CUgraphNode}, rootNodes), Base.cconvert(Ref{Csize_t}, numRootNodes))
end

function cuGraphGetRootNodes(hGraph::CUgraph, rootNodes::Ptr{CUgraphNode}, numRootNodes::Ptr{Csize_t})::CUresult
    return ccall((:cuGraphGetRootNodes, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t},), hGraph, rootNodes, numRootNodes)
end

# cuGraphGetEdges() available since CUDA 10.0
function cuGraphGetEdges(hGraph::CUgraph, from::Array{CUgraphNode, 1}, to::Array{CUgraphNode, 1}, numEdges::Array{Csize_t, 1})::CUresult
    return ccall((:cuGraphGetEdges, libcuda), CUresult, (CUgraph, Ref{CUgraphNode}, Ref{CUgraphNode}, Ref{Csize_t},), hGraph, Base.cconvert(Ref{CUgraphNode}, from), Base.cconvert(Ref{CUgraphNode}, to), Base.cconvert(Ref{Csize_t}, numEdges))
end

function cuGraphGetEdges(hGraph::CUgraph, from::Ptr{CUgraphNode}, to::Ptr{CUgraphNode}, numEdges::Ptr{Csize_t})::CUresult
    return ccall((:cuGraphGetEdges, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Ptr{Csize_t},), hGraph, from, to, numEdges)
end

# cuGraphNodeGetDependencies() available since CUDA 10.0
function cuGraphNodeGetDependencies(hNode::CUgraphNode, dependencies::Array{CUgraphNode, 1}, numDependencies::Array{Csize_t, 1})::CUresult
    return ccall((:cuGraphNodeGetDependencies, libcuda), CUresult, (CUgraphNode, Ref{CUgraphNode}, Ref{Csize_t},), hNode, Base.cconvert(Ref{CUgraphNode}, dependencies), Base.cconvert(Ref{Csize_t}, numDependencies))
end

function cuGraphNodeGetDependencies(hNode::CUgraphNode, dependencies::Ptr{CUgraphNode}, numDependencies::Ptr{Csize_t})::CUresult
    return ccall((:cuGraphNodeGetDependencies, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t},), hNode, dependencies, numDependencies)
end

# cuGraphNodeGetDependentNodes() available since CUDA 10.0
function cuGraphNodeGetDependentNodes(hNode::CUgraphNode, dependentNodes::Array{CUgraphNode, 1}, numDependentNodes::Array{Csize_t, 1})::CUresult
    return ccall((:cuGraphNodeGetDependentNodes, libcuda), CUresult, (CUgraphNode, Ref{CUgraphNode}, Ref{Csize_t},), hNode, Base.cconvert(Ref{CUgraphNode}, dependentNodes), Base.cconvert(Ref{Csize_t}, numDependentNodes))
end

function cuGraphNodeGetDependentNodes(hNode::CUgraphNode, dependentNodes::Ptr{CUgraphNode}, numDependentNodes::Ptr{Csize_t})::CUresult
    return ccall((:cuGraphNodeGetDependentNodes, libcuda), CUresult, (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t},), hNode, dependentNodes, numDependentNodes)
end

# cuGraphAddDependencies() available since CUDA 10.0
function cuGraphAddDependencies(hGraph::CUgraph, from::Array{CUgraphNode, 1}, to::Array{CUgraphNode, 1}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphAddDependencies, libcuda), CUresult, (CUgraph, Ref{CUgraphNode}, Ref{CUgraphNode}, Csize_t,), hGraph, Base.cconvert(Ref{CUgraphNode}, from), Base.cconvert(Ref{CUgraphNode}, to), numDependencies)
end

function cuGraphAddDependencies(hGraph::CUgraph, from::Ptr{CUgraphNode}, to::Ptr{CUgraphNode}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphAddDependencies, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t,), hGraph, from, to, numDependencies)
end

# cuGraphRemoveDependencies() available since CUDA 10.0
function cuGraphRemoveDependencies(hGraph::CUgraph, from::Array{CUgraphNode, 1}, to::Array{CUgraphNode, 1}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphRemoveDependencies, libcuda), CUresult, (CUgraph, Ref{CUgraphNode}, Ref{CUgraphNode}, Csize_t,), hGraph, Base.cconvert(Ref{CUgraphNode}, from), Base.cconvert(Ref{CUgraphNode}, to), numDependencies)
end

function cuGraphRemoveDependencies(hGraph::CUgraph, from::Ptr{CUgraphNode}, to::Ptr{CUgraphNode}, numDependencies::Csize_t)::CUresult
    return ccall((:cuGraphRemoveDependencies, libcuda), CUresult, (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t,), hGraph, from, to, numDependencies)
end

# cuGraphDestroyNode() available since CUDA 10.0
function cuGraphDestroyNode(hNode::CUgraphNode)::CUresult
    return ccall((:cuGraphDestroyNode, libcuda), CUresult, (CUgraphNode,), hNode)
end

# cuGraphInstantiate() available since CUDA 10.0
function cuGraphInstantiate(phGraphExec::Array{CUgraphExec, 1}, hGraph::CUgraph, phErrorNode::Array{CUgraphNode, 1}, logBuffer::Array{UInt8, 1}, bufferSize::Csize_t)::CUresult
    return ccall((:cuGraphInstantiate, libcuda), CUresult, (Ref{CUgraphExec}, CUgraph, Ref{CUgraphNode}, Ref{UInt8}, Csize_t,), Base.cconvert(Ref{CUgraphExec}, phGraphExec), hGraph, Base.cconvert(Ref{CUgraphNode}, phErrorNode), Base.cconvert(Ref{UInt8}, logBuffer), bufferSize)
end

function cuGraphInstantiate(phGraphExec::Ptr{CUgraphExec}, hGraph::CUgraph, phErrorNode::Ptr{CUgraphNode}, logBuffer::Ptr{UInt8}, bufferSize::Csize_t)::CUresult
    return ccall((:cuGraphInstantiate, libcuda), CUresult, (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Ptr{UInt8}, Csize_t,), phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end

# cuGraphExecKernelNodeSetParams() available since CUDA 10.1
function cuGraphExecKernelNodeSetParams(hGraphExec::CUgraphExec, hNode::CUgraphNode, nodeParams::Array{CUDA_KERNEL_NODE_PARAMS, 1})::CUresult
    return ccall((:cuGraphExecKernelNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ref{CUDA_KERNEL_NODE_PARAMS},), hGraphExec, hNode, Base.cconvert(Ref{CUDA_KERNEL_NODE_PARAMS}, nodeParams))
end

function cuGraphExecKernelNodeSetParams(hGraphExec::CUgraphExec, hNode::CUgraphNode, nodeParams::Ptr{CUDA_KERNEL_NODE_PARAMS})::CUresult
    return ccall((:cuGraphExecKernelNodeSetParams, libcuda), CUresult, (CUgraphExec, CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS},), hGraphExec, hNode, nodeParams)
end

# cuGraphLaunch() available since CUDA 10.0
function cuGraphLaunch(hGraphExec::CUgraphExec, hStream::CUstream)::CUresult
    return ccall((:cuGraphLaunch, libcuda), CUresult, (CUgraphExec, CUstream,), hGraphExec, hStream)
end

# cuGraphExecDestroy() available since CUDA 10.0
function cuGraphExecDestroy(hGraphExec::CUgraphExec)::CUresult
    return ccall((:cuGraphExecDestroy, libcuda), CUresult, (CUgraphExec,), hGraphExec)
end

# cuGraphDestroy() available since CUDA 10.0
function cuGraphDestroy(hGraph::CUgraph)::CUresult
    return ccall((:cuGraphDestroy, libcuda), CUresult, (CUgraph,), hGraph)
end

# cuOccupancyMaxActiveBlocksPerMultiprocessor() available since CUDA 6.5
function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Array{Cint, 1}, func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t)::CUresult
    return ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor, libcuda), CUresult, (Ref{Cint}, CUfunction, Cint, Csize_t,), Base.cconvert(Ref{Cint}, numBlocks), func, blockSize, dynamicSMemSize)
end

function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks::Ptr{Cint}, func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t)::CUresult
    return ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor, libcuda), CUresult, (Ptr{Cint}, CUfunction, Cint, Csize_t,), numBlocks, func, blockSize, dynamicSMemSize)
end

# cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() available since CUDA 6.5
function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Array{Cint, 1}, func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t, flags::Cuint)::CUresult
    return ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcuda), CUresult, (Ref{Cint}, CUfunction, Cint, Csize_t, Cuint,), Base.cconvert(Ref{Cint}, numBlocks), func, blockSize, dynamicSMemSize, flags)
end

function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks::Ptr{Cint}, func::CUfunction, blockSize::Cint, dynamicSMemSize::Csize_t, flags::Cuint)::CUresult
    return ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcuda), CUresult, (Ptr{Cint}, CUfunction, Cint, Csize_t, Cuint,), numBlocks, func, blockSize, dynamicSMemSize, flags)
end

# cuOccupancyMaxPotentialBlockSize() available since CUDA 6.5
function cuOccupancyMaxPotentialBlockSize(minGridSize::Array{Cint, 1}, blockSize::Array{Cint, 1}, func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint)::CUresult
    return ccall((:cuOccupancyMaxPotentialBlockSize, libcuda), CUresult, (Ref{Cint}, Ref{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint,), Base.cconvert(Ref{Cint}, minGridSize), Base.cconvert(Ref{Cint}, blockSize), func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
end

function cuOccupancyMaxPotentialBlockSize(minGridSize::Ptr{Cint}, blockSize::Ptr{Cint}, func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint)::CUresult
    return ccall((:cuOccupancyMaxPotentialBlockSize, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint,), minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
end

# cuOccupancyMaxPotentialBlockSizeWithFlags() available since CUDA 6.5
function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize::Array{Cint, 1}, blockSize::Array{Cint, 1}, func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint, flags::Cuint)::CUresult
    return ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags, libcuda), CUresult, (Ref{Cint}, Ref{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint, Cuint,), Base.cconvert(Ref{Cint}, minGridSize), Base.cconvert(Ref{Cint}, blockSize), func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
end

function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize::Ptr{Cint}, blockSize::Ptr{Cint}, func::CUfunction, blockSizeToDynamicSMemSize::CUoccupancyB2DSize, dynamicSMemSize::Csize_t, blockSizeLimit::Cint, flags::Cuint)::CUresult
    return ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags, libcuda), CUresult, (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint, Cuint,), minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
end

function cuTexRefSetArray(hTexRef::CUtexref, hArray::CUarray, Flags::Cuint)::CUresult
    return ccall((:cuTexRefSetArray, libcuda), CUresult, (CUtexref, CUarray, Cuint,), hTexRef, hArray, Flags)
end

function cuTexRefSetMipmappedArray(hTexRef::CUtexref, hMipmappedArray::CUmipmappedArray, Flags::Cuint)::CUresult
    return ccall((:cuTexRefSetMipmappedArray, libcuda), CUresult, (CUtexref, CUmipmappedArray, Cuint,), hTexRef, hMipmappedArray, Flags)
end

# cuTexRefSetAddress() available since CUDA 3.2
function cuTexRefSetAddress(ByteOffset::Array{Csize_t, 1}, hTexRef::CUtexref, dptr::CUdeviceptr, bytes::Csize_t)::CUresult
    return ccall((:cuTexRefSetAddress, libcuda), CUresult, (Ref{Csize_t}, CUtexref, CUdeviceptr, Csize_t,), Base.cconvert(Ref{Csize_t}, ByteOffset), hTexRef, dptr, bytes)
end

function cuTexRefSetAddress(ByteOffset::Ptr{Csize_t}, hTexRef::CUtexref, dptr::CUdeviceptr, bytes::Csize_t)::CUresult
    return ccall((:cuTexRefSetAddress, libcuda), CUresult, (Ptr{Csize_t}, CUtexref, CUdeviceptr, Csize_t,), ByteOffset, hTexRef, dptr, bytes)
end

# cuTexRefSetAddress2D() available since CUDA 3.2
function cuTexRefSetAddress2D(hTexRef::CUtexref, desc::Array{CUDA_ARRAY_DESCRIPTOR, 1}, dptr::CUdeviceptr, Pitch::Csize_t)::CUresult
    return ccall((:cuTexRefSetAddress2D, libcuda), CUresult, (CUtexref, Ref{CUDA_ARRAY_DESCRIPTOR}, CUdeviceptr, Csize_t,), hTexRef, Base.cconvert(Ref{CUDA_ARRAY_DESCRIPTOR}, desc), dptr, Pitch)
end

function cuTexRefSetAddress2D(hTexRef::CUtexref, desc::Ptr{CUDA_ARRAY_DESCRIPTOR}, dptr::CUdeviceptr, Pitch::Csize_t)::CUresult
    return ccall((:cuTexRefSetAddress2D, libcuda), CUresult, (CUtexref, Ptr{CUDA_ARRAY_DESCRIPTOR}, CUdeviceptr, Csize_t,), hTexRef, desc, dptr, Pitch)
end

function cuTexRefSetFormat(hTexRef::CUtexref, fmt::CUarray_format, NumPackedComponents::Cint)::CUresult
    return ccall((:cuTexRefSetFormat, libcuda), CUresult, (CUtexref, CUarray_format, Cint,), hTexRef, fmt, NumPackedComponents)
end

function cuTexRefSetAddressMode(hTexRef::CUtexref, dim::Cint, am::CUaddress_mode)::CUresult
    return ccall((:cuTexRefSetAddressMode, libcuda), CUresult, (CUtexref, Cint, CUaddress_mode,), hTexRef, dim, am)
end

function cuTexRefSetFilterMode(hTexRef::CUtexref, fm::CUfilter_mode)::CUresult
    return ccall((:cuTexRefSetFilterMode, libcuda), CUresult, (CUtexref, CUfilter_mode,), hTexRef, fm)
end

function cuTexRefSetMipmapFilterMode(hTexRef::CUtexref, fm::CUfilter_mode)::CUresult
    return ccall((:cuTexRefSetMipmapFilterMode, libcuda), CUresult, (CUtexref, CUfilter_mode,), hTexRef, fm)
end

function cuTexRefSetMipmapLevelBias(hTexRef::CUtexref, bias::Cfloat)::CUresult
    return ccall((:cuTexRefSetMipmapLevelBias, libcuda), CUresult, (CUtexref, Cfloat,), hTexRef, bias)
end

function cuTexRefSetMipmapLevelClamp(hTexRef::CUtexref, minMipmapLevelClamp::Cfloat, maxMipmapLevelClamp::Cfloat)::CUresult
    return ccall((:cuTexRefSetMipmapLevelClamp, libcuda), CUresult, (CUtexref, Cfloat, Cfloat,), hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
end

function cuTexRefSetMaxAnisotropy(hTexRef::CUtexref, maxAniso::Cuint)::CUresult
    return ccall((:cuTexRefSetMaxAnisotropy, libcuda), CUresult, (CUtexref, Cuint,), hTexRef, maxAniso)
end

function cuTexRefSetBorderColor(hTexRef::CUtexref, pBorderColor::Array{Cfloat, 1})::CUresult
    return ccall((:cuTexRefSetBorderColor, libcuda), CUresult, (CUtexref, Ref{Cfloat},), hTexRef, Base.cconvert(Ref{Cfloat}, pBorderColor))
end

function cuTexRefSetBorderColor(hTexRef::CUtexref, pBorderColor::Ptr{Cfloat})::CUresult
    return ccall((:cuTexRefSetBorderColor, libcuda), CUresult, (CUtexref, Ptr{Cfloat},), hTexRef, pBorderColor)
end

function cuTexRefSetFlags(hTexRef::CUtexref, Flags::Cuint)::CUresult
    return ccall((:cuTexRefSetFlags, libcuda), CUresult, (CUtexref, Cuint,), hTexRef, Flags)
end

# cuTexRefGetAddress() available since CUDA 3.2
function cuTexRefGetAddress(pdptr::Array{CUdeviceptr, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetAddress, libcuda), CUresult, (Ref{CUdeviceptr}, CUtexref,), Base.cconvert(Ref{CUdeviceptr}, pdptr), hTexRef)
end

function cuTexRefGetAddress(pdptr::Ptr{CUdeviceptr}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetAddress, libcuda), CUresult, (Ptr{CUdeviceptr}, CUtexref,), pdptr, hTexRef)
end

function cuTexRefGetArray(phArray::Array{CUarray, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetArray, libcuda), CUresult, (Ref{CUarray}, CUtexref,), Base.cconvert(Ref{CUarray}, phArray), hTexRef)
end

function cuTexRefGetArray(phArray::Ptr{CUarray}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetArray, libcuda), CUresult, (Ptr{CUarray}, CUtexref,), phArray, hTexRef)
end

function cuTexRefGetMipmappedArray(phMipmappedArray::Array{CUmipmappedArray, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmappedArray, libcuda), CUresult, (Ref{CUmipmappedArray}, CUtexref,), Base.cconvert(Ref{CUmipmappedArray}, phMipmappedArray), hTexRef)
end

function cuTexRefGetMipmappedArray(phMipmappedArray::Ptr{CUmipmappedArray}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUtexref,), phMipmappedArray, hTexRef)
end

function cuTexRefGetAddressMode(pam::Array{CUaddress_mode, 1}, hTexRef::CUtexref, dim::Cint)::CUresult
    return ccall((:cuTexRefGetAddressMode, libcuda), CUresult, (Ref{CUaddress_mode}, CUtexref, Cint,), Base.cconvert(Ref{CUaddress_mode}, pam), hTexRef, dim)
end

function cuTexRefGetAddressMode(pam::Ptr{CUaddress_mode}, hTexRef::CUtexref, dim::Cint)::CUresult
    return ccall((:cuTexRefGetAddressMode, libcuda), CUresult, (Ptr{CUaddress_mode}, CUtexref, Cint,), pam, hTexRef, dim)
end

function cuTexRefGetFilterMode(pfm::Array{CUfilter_mode, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFilterMode, libcuda), CUresult, (Ref{CUfilter_mode}, CUtexref,), Base.cconvert(Ref{CUfilter_mode}, pfm), hTexRef)
end

function cuTexRefGetFilterMode(pfm::Ptr{CUfilter_mode}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFilterMode, libcuda), CUresult, (Ptr{CUfilter_mode}, CUtexref,), pfm, hTexRef)
end

function cuTexRefGetFormat(pFormat::Array{CUarray_format, 1}, pNumChannels::Array{Cint, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFormat, libcuda), CUresult, (Ref{CUarray_format}, Ref{Cint}, CUtexref,), Base.cconvert(Ref{CUarray_format}, pFormat), Base.cconvert(Ref{Cint}, pNumChannels), hTexRef)
end

function cuTexRefGetFormat(pFormat::Ptr{CUarray_format}, pNumChannels::Ptr{Cint}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFormat, libcuda), CUresult, (Ptr{CUarray_format}, Ptr{Cint}, CUtexref,), pFormat, pNumChannels, hTexRef)
end

function cuTexRefGetMipmapFilterMode(pfm::Array{CUfilter_mode, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapFilterMode, libcuda), CUresult, (Ref{CUfilter_mode}, CUtexref,), Base.cconvert(Ref{CUfilter_mode}, pfm), hTexRef)
end

function cuTexRefGetMipmapFilterMode(pfm::Ptr{CUfilter_mode}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapFilterMode, libcuda), CUresult, (Ptr{CUfilter_mode}, CUtexref,), pfm, hTexRef)
end

function cuTexRefGetMipmapLevelBias(pbias::Array{Cfloat, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapLevelBias, libcuda), CUresult, (Ref{Cfloat}, CUtexref,), Base.cconvert(Ref{Cfloat}, pbias), hTexRef)
end

function cuTexRefGetMipmapLevelBias(pbias::Ptr{Cfloat}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapLevelBias, libcuda), CUresult, (Ptr{Cfloat}, CUtexref,), pbias, hTexRef)
end

function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp::Array{Cfloat, 1}, pmaxMipmapLevelClamp::Array{Cfloat, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapLevelClamp, libcuda), CUresult, (Ref{Cfloat}, Ref{Cfloat}, CUtexref,), Base.cconvert(Ref{Cfloat}, pminMipmapLevelClamp), Base.cconvert(Ref{Cfloat}, pmaxMipmapLevelClamp), hTexRef)
end

function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp::Ptr{Cfloat}, pmaxMipmapLevelClamp::Ptr{Cfloat}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMipmapLevelClamp, libcuda), CUresult, (Ptr{Cfloat}, Ptr{Cfloat}, CUtexref,), pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
end

function cuTexRefGetMaxAnisotropy(pmaxAniso::Array{Cint, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMaxAnisotropy, libcuda), CUresult, (Ref{Cint}, CUtexref,), Base.cconvert(Ref{Cint}, pmaxAniso), hTexRef)
end

function cuTexRefGetMaxAnisotropy(pmaxAniso::Ptr{Cint}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetMaxAnisotropy, libcuda), CUresult, (Ptr{Cint}, CUtexref,), pmaxAniso, hTexRef)
end

function cuTexRefGetBorderColor(pBorderColor::Array{Cfloat, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetBorderColor, libcuda), CUresult, (Ref{Cfloat}, CUtexref,), Base.cconvert(Ref{Cfloat}, pBorderColor), hTexRef)
end

function cuTexRefGetBorderColor(pBorderColor::Ptr{Cfloat}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetBorderColor, libcuda), CUresult, (Ptr{Cfloat}, CUtexref,), pBorderColor, hTexRef)
end

function cuTexRefGetFlags(pFlags::Array{Cuint, 1}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFlags, libcuda), CUresult, (Ref{Cuint}, CUtexref,), Base.cconvert(Ref{Cuint}, pFlags), hTexRef)
end

function cuTexRefGetFlags(pFlags::Ptr{Cuint}, hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefGetFlags, libcuda), CUresult, (Ptr{Cuint}, CUtexref,), pFlags, hTexRef)
end

function cuTexRefCreate(pTexRef::Array{CUtexref, 1})::CUresult
    return ccall((:cuTexRefCreate, libcuda), CUresult, (Ref{CUtexref},), Base.cconvert(Ref{CUtexref}, pTexRef))
end

function cuTexRefCreate(pTexRef::Ptr{CUtexref})::CUresult
    return ccall((:cuTexRefCreate, libcuda), CUresult, (Ptr{CUtexref},), pTexRef)
end

function cuTexRefDestroy(hTexRef::CUtexref)::CUresult
    return ccall((:cuTexRefDestroy, libcuda), CUresult, (CUtexref,), hTexRef)
end

function cuSurfRefSetArray(hSurfRef::CUsurfref, hArray::CUarray, Flags::Cuint)::CUresult
    return ccall((:cuSurfRefSetArray, libcuda), CUresult, (CUsurfref, CUarray, Cuint,), hSurfRef, hArray, Flags)
end

function cuSurfRefGetArray(phArray::Array{CUarray, 1}, hSurfRef::CUsurfref)::CUresult
    return ccall((:cuSurfRefGetArray, libcuda), CUresult, (Ref{CUarray}, CUsurfref,), Base.cconvert(Ref{CUarray}, phArray), hSurfRef)
end

function cuSurfRefGetArray(phArray::Ptr{CUarray}, hSurfRef::CUsurfref)::CUresult
    return ccall((:cuSurfRefGetArray, libcuda), CUresult, (Ptr{CUarray}, CUsurfref,), phArray, hSurfRef)
end

# cuTexObjectCreate() available since CUDA 5.0
function cuTexObjectCreate(pTexObject::Array{CUtexObject, 1}, pResDesc::Array{CUDA_RESOURCE_DESC, 1}, pTexDesc::Array{CUDA_TEXTURE_DESC, 1}, pResViewDesc::Array{CUDA_RESOURCE_VIEW_DESC, 1})::CUresult
    return ccall((:cuTexObjectCreate, libcuda), CUresult, (Ref{CUtexObject}, Ref{CUDA_RESOURCE_DESC}, Ref{CUDA_TEXTURE_DESC}, Ref{CUDA_RESOURCE_VIEW_DESC},), Base.cconvert(Ref{CUtexObject}, pTexObject), Base.cconvert(Ref{CUDA_RESOURCE_DESC}, pResDesc), Base.cconvert(Ref{CUDA_TEXTURE_DESC}, pTexDesc), Base.cconvert(Ref{CUDA_RESOURCE_VIEW_DESC}, pResViewDesc))
end

function cuTexObjectCreate(pTexObject::Ptr{CUtexObject}, pResDesc::Ptr{CUDA_RESOURCE_DESC}, pTexDesc::Ptr{CUDA_TEXTURE_DESC}, pResViewDesc::Ptr{CUDA_RESOURCE_VIEW_DESC})::CUresult
    return ccall((:cuTexObjectCreate, libcuda), CUresult, (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC}, Ptr{CUDA_RESOURCE_VIEW_DESC},), pTexObject, pResDesc, pTexDesc, pResViewDesc)
end

# cuTexObjectDestroy() available since CUDA 5.0
function cuTexObjectDestroy(texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectDestroy, libcuda), CUresult, (CUtexObject,), texObject)
end

# cuTexObjectGetResourceDesc() available since CUDA 5.0
function cuTexObjectGetResourceDesc(pResDesc::Array{CUDA_RESOURCE_DESC, 1}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetResourceDesc, libcuda), CUresult, (Ref{CUDA_RESOURCE_DESC}, CUtexObject,), Base.cconvert(Ref{CUDA_RESOURCE_DESC}, pResDesc), texObject)
end

function cuTexObjectGetResourceDesc(pResDesc::Ptr{CUDA_RESOURCE_DESC}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetResourceDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_DESC}, CUtexObject,), pResDesc, texObject)
end

# cuTexObjectGetTextureDesc() available since CUDA 5.0
function cuTexObjectGetTextureDesc(pTexDesc::Array{CUDA_TEXTURE_DESC, 1}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetTextureDesc, libcuda), CUresult, (Ref{CUDA_TEXTURE_DESC}, CUtexObject,), Base.cconvert(Ref{CUDA_TEXTURE_DESC}, pTexDesc), texObject)
end

function cuTexObjectGetTextureDesc(pTexDesc::Ptr{CUDA_TEXTURE_DESC}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetTextureDesc, libcuda), CUresult, (Ptr{CUDA_TEXTURE_DESC}, CUtexObject,), pTexDesc, texObject)
end

# cuTexObjectGetResourceViewDesc() available since CUDA 5.0
function cuTexObjectGetResourceViewDesc(pResViewDesc::Array{CUDA_RESOURCE_VIEW_DESC, 1}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetResourceViewDesc, libcuda), CUresult, (Ref{CUDA_RESOURCE_VIEW_DESC}, CUtexObject,), Base.cconvert(Ref{CUDA_RESOURCE_VIEW_DESC}, pResViewDesc), texObject)
end

function cuTexObjectGetResourceViewDesc(pResViewDesc::Ptr{CUDA_RESOURCE_VIEW_DESC}, texObject::CUtexObject)::CUresult
    return ccall((:cuTexObjectGetResourceViewDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_VIEW_DESC}, CUtexObject,), pResViewDesc, texObject)
end

# cuSurfObjectCreate() available since CUDA 5.0
function cuSurfObjectCreate(pSurfObject::Array{CUsurfObject, 1}, pResDesc::Array{CUDA_RESOURCE_DESC, 1})::CUresult
    return ccall((:cuSurfObjectCreate, libcuda), CUresult, (Ref{CUsurfObject}, Ref{CUDA_RESOURCE_DESC},), Base.cconvert(Ref{CUsurfObject}, pSurfObject), Base.cconvert(Ref{CUDA_RESOURCE_DESC}, pResDesc))
end

function cuSurfObjectCreate(pSurfObject::Ptr{CUsurfObject}, pResDesc::Ptr{CUDA_RESOURCE_DESC})::CUresult
    return ccall((:cuSurfObjectCreate, libcuda), CUresult, (Ptr{CUsurfObject}, Ptr{CUDA_RESOURCE_DESC},), pSurfObject, pResDesc)
end

# cuSurfObjectDestroy() available since CUDA 5.0
function cuSurfObjectDestroy(surfObject::CUsurfObject)::CUresult
    return ccall((:cuSurfObjectDestroy, libcuda), CUresult, (CUsurfObject,), surfObject)
end

# cuSurfObjectGetResourceDesc() available since CUDA 5.0
function cuSurfObjectGetResourceDesc(pResDesc::Array{CUDA_RESOURCE_DESC, 1}, surfObject::CUsurfObject)::CUresult
    return ccall((:cuSurfObjectGetResourceDesc, libcuda), CUresult, (Ref{CUDA_RESOURCE_DESC}, CUsurfObject,), Base.cconvert(Ref{CUDA_RESOURCE_DESC}, pResDesc), surfObject)
end

function cuSurfObjectGetResourceDesc(pResDesc::Ptr{CUDA_RESOURCE_DESC}, surfObject::CUsurfObject)::CUresult
    return ccall((:cuSurfObjectGetResourceDesc, libcuda), CUresult, (Ptr{CUDA_RESOURCE_DESC}, CUsurfObject,), pResDesc, surfObject)
end

# cuDeviceCanAccessPeer() available since CUDA 4.0
function cuDeviceCanAccessPeer(canAccessPeer::Array{Cint, 1}, dev::CUdevice, peerDev::CUdevice)::CUresult
    return ccall((:cuDeviceCanAccessPeer, libcuda), CUresult, (Ref{Cint}, CUdevice, CUdevice,), Base.cconvert(Ref{Cint}, canAccessPeer), dev, peerDev)
end

function cuDeviceCanAccessPeer(canAccessPeer::Ptr{Cint}, dev::CUdevice, peerDev::CUdevice)::CUresult
    return ccall((:cuDeviceCanAccessPeer, libcuda), CUresult, (Ptr{Cint}, CUdevice, CUdevice,), canAccessPeer, dev, peerDev)
end

# cuCtxEnablePeerAccess() available since CUDA 4.0
function cuCtxEnablePeerAccess(peerContext::CUcontext, Flags::Cuint)::CUresult
    return ccall((:cuCtxEnablePeerAccess, libcuda), CUresult, (CUcontext, Cuint,), peerContext, Flags)
end

# cuCtxDisablePeerAccess() available since CUDA 4.0
function cuCtxDisablePeerAccess(peerContext::CUcontext)::CUresult
    return ccall((:cuCtxDisablePeerAccess, libcuda), CUresult, (CUcontext,), peerContext)
end

# cuDeviceGetP2PAttribute() available since CUDA 8.0
function cuDeviceGetP2PAttribute(value::Array{Cint, 1}, attrib::CUdevice_P2PAttribute, srcDevice::CUdevice, dstDevice::CUdevice)::CUresult
    return ccall((:cuDeviceGetP2PAttribute, libcuda), CUresult, (Ref{Cint}, CUdevice_P2PAttribute, CUdevice, CUdevice,), Base.cconvert(Ref{Cint}, value), attrib, srcDevice, dstDevice)
end

function cuDeviceGetP2PAttribute(value::Ptr{Cint}, attrib::CUdevice_P2PAttribute, srcDevice::CUdevice, dstDevice::CUdevice)::CUresult
    return ccall((:cuDeviceGetP2PAttribute, libcuda), CUresult, (Ptr{Cint}, CUdevice_P2PAttribute, CUdevice, CUdevice,), value, attrib, srcDevice, dstDevice)
end

function cuGraphicsUnregisterResource(resource::CUgraphicsResource)::CUresult
    return ccall((:cuGraphicsUnregisterResource, libcuda), CUresult, (CUgraphicsResource,), resource)
end

function cuGraphicsSubResourceGetMappedArray(pArray::Array{CUarray, 1}, resource::CUgraphicsResource, arrayIndex::Cuint, mipLevel::Cuint)::CUresult
    return ccall((:cuGraphicsSubResourceGetMappedArray, libcuda), CUresult, (Ref{CUarray}, CUgraphicsResource, Cuint, Cuint,), Base.cconvert(Ref{CUarray}, pArray), resource, arrayIndex, mipLevel)
end

function cuGraphicsSubResourceGetMappedArray(pArray::Ptr{CUarray}, resource::CUgraphicsResource, arrayIndex::Cuint, mipLevel::Cuint)::CUresult
    return ccall((:cuGraphicsSubResourceGetMappedArray, libcuda), CUresult, (Ptr{CUarray}, CUgraphicsResource, Cuint, Cuint,), pArray, resource, arrayIndex, mipLevel)
end

# cuGraphicsResourceGetMappedMipmappedArray() available since CUDA 5.0
function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray::Array{CUmipmappedArray, 1}, resource::CUgraphicsResource)::CUresult
    return ccall((:cuGraphicsResourceGetMappedMipmappedArray, libcuda), CUresult, (Ref{CUmipmappedArray}, CUgraphicsResource,), Base.cconvert(Ref{CUmipmappedArray}, pMipmappedArray), resource)
end

function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray::Ptr{CUmipmappedArray}, resource::CUgraphicsResource)::CUresult
    return ccall((:cuGraphicsResourceGetMappedMipmappedArray, libcuda), CUresult, (Ptr{CUmipmappedArray}, CUgraphicsResource,), pMipmappedArray, resource)
end

# cuGraphicsResourceGetMappedPointer() available since CUDA 3.2
function cuGraphicsResourceGetMappedPointer(pDevPtr::Array{CUdeviceptr, 1}, pSize::Array{Csize_t, 1}, resource::CUgraphicsResource)::CUresult
    return ccall((:cuGraphicsResourceGetMappedPointer, libcuda), CUresult, (Ref{CUdeviceptr}, Ref{Csize_t}, CUgraphicsResource,), Base.cconvert(Ref{CUdeviceptr}, pDevPtr), Base.cconvert(Ref{Csize_t}, pSize), resource)
end

function cuGraphicsResourceGetMappedPointer(pDevPtr::Ptr{CUdeviceptr}, pSize::Ptr{Csize_t}, resource::CUgraphicsResource)::CUresult
    return ccall((:cuGraphicsResourceGetMappedPointer, libcuda), CUresult, (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUgraphicsResource,), pDevPtr, pSize, resource)
end

function cuGraphicsResourceSetMapFlags(resource::CUgraphicsResource, flags::Cuint)::CUresult
    return ccall((:cuGraphicsResourceSetMapFlags, libcuda), CUresult, (CUgraphicsResource, Cuint,), resource, flags)
end

function cuGraphicsMapResources(count::Cuint, resources::Array{CUgraphicsResource, 1}, hStream::CUstream)::CUresult
    return ccall((:cuGraphicsMapResources, libcuda), CUresult, (Cuint, Ref{CUgraphicsResource}, CUstream,), count, Base.cconvert(Ref{CUgraphicsResource}, resources), hStream)
end

function cuGraphicsMapResources(count::Cuint, resources::Ptr{CUgraphicsResource}, hStream::CUstream)::CUresult
    return ccall((:cuGraphicsMapResources, libcuda), CUresult, (Cuint, Ptr{CUgraphicsResource}, CUstream,), count, resources, hStream)
end

function cuGraphicsUnmapResources(count::Cuint, resources::Array{CUgraphicsResource, 1}, hStream::CUstream)::CUresult
    return ccall((:cuGraphicsUnmapResources, libcuda), CUresult, (Cuint, Ref{CUgraphicsResource}, CUstream,), count, Base.cconvert(Ref{CUgraphicsResource}, resources), hStream)
end

function cuGraphicsUnmapResources(count::Cuint, resources::Ptr{CUgraphicsResource}, hStream::CUstream)::CUresult
    return ccall((:cuGraphicsUnmapResources, libcuda), CUresult, (Cuint, Ptr{CUgraphicsResource}, CUstream,), count, resources, hStream)
end

function cuGetExportTable(ppExportTable::Array{Ptr{Cvoid}, 1}, pExportTableId::Array{CUuuid, 1})::CUresult
    return ccall((:cuGetExportTable, libcuda), CUresult, (Ref{Ptr{Cvoid}}, Ref{CUuuid},), Base.cconvert(Ref{Ptr{Cvoid}}, ppExportTable), Base.cconvert(Ref{CUuuid}, pExportTableId))
end

function cuGetExportTable(ppExportTable::Ptr{Ptr{Cvoid}}, pExportTableId::Ptr{CUuuid})::CUresult
    return ccall((:cuGetExportTable, libcuda), CUresult, (Ptr{Ptr{Cvoid}}, Ptr{CUuuid},), ppExportTable, pExportTableId)
end
