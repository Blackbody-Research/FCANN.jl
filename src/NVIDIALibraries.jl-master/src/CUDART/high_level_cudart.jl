#=*
* High level CUDA runtime functions
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

function cudaGetDeviceCount()::Cint
    local device_count::Array{Cint, 1} = zeros(Cint, 1)
    local result::cudaError_t = cudaGetDeviceCount(device_count)
    @assert (result == cudaSuccess) ("cudaGetDeviceCount() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(device_count)
end

function cudaGetDevice()::Cint
    local device_count::Array{Cint, 1} = zeros(Cint, 1)
    local result::cudaError_t = cudaGetDevice(device_count)
    @assert (result == cudaSuccess) ("cudaGetDevice() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(device_count)
end

cudaSetDevice(device::Integer) = cudaSetDevice(Cint(device))

function cudaStreamCreate()::cudaStream_t
    local stream_array::Array{cudaStream_t, 1} = [C_NULL]
    local result::cudaError_t = cudaStreamCreate(stream_array)
    @assert (result == cudaSuccess) ("cudaStreamCreate() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(stream_array)
end

function cudaStreamCreateWithFlags(flags::Cuint)::cudaStream_t
    local stream_array::Array{cudaStream_t, 1} = [C_NULL]
    local result::cudaError_t = cudaStreamCreateWithFlags(stream_array, flags)
    @assert (result == cudaSuccess) ("cudaStreamCreateWithFlags() error: " * unsafe_string(cudaGetErrorString(result)))
    return pop!(stream_array)
end

cudaMalloc(devPtr::Array{Ptr{Cvoid}, 1}, size::Integer) = cudaMalloc(Base.unsafe_convert(Ptr{Ptr{Nothing}}, devPtr), Csize_t(size))
cudaMalloc(devPtr::Ptr{Ptr{Cvoid}}, size::Integer) = cudaMalloc(devPtr, Csize_t(size))

cudaMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, count::Integer, kind::Cuint) = cudaMemcpy(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, Csize_t(count), kind)
cudaMemcpy(dst::Ptr{Cvoid}, src::Array{T}, count::Integer, kind::Cuint) where {T} = cudaMemcpy(dst, Ptr{Cvoid}(Base.unsafe_convert(Ptr{T}, src)), Csize_t(count), kind)
cudaMemcpy(dst::Array{T}, src::Ptr{Cvoid}, count::Integer, kind::Cuint) where {T} = cudaMemcpy(Ptr{Cvoid}(Base.unsafe_convert(Ptr{T}, dst)), src, Csize_t(count), kind)

# CUDAArray functions
using ..DeviceArray
export cuda_allocate, deallocate!,
        unsafe_memcpy!, memcpy!

@inline function cudaLaunchKernel(func::Ptr{Nothing}, grid::dim3, block::dim3, types::Tuple{Vararg{DataType}}, args...; kwargs...)
    return cudaLaunchKernel(func, grid, block, Tuple{types...}, map(_cast_cudaarray_args, args)...; kwargs...)
end

@generated function cudaLaunchKernel(func::Ptr{Nothing}, grid::dim3, block::dim3, types::Type, args...; stream::cudaStream_t = cudaStream_t(C_NULL), shmem::Integer = 0, kwargs...)
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
            result::cudaError_t = cudaLaunchKernel(func,
                                grid,
                                block,
                                Base.unsafe_convert(Ptr{Ptr{Nothing}}, arguments),
                                Csize_t(shmem),
                                stream)
            @assert (result == cudaSuccess) ("cudaLaunchKernel() error: " * unsafe_string(cudaGetErrorName(result)))
        end
    end).args)

    return result_expr
end

function cuda_allocate(jl_array::Array{T})::CUDAArray where T
    local device_pointer_array::Array{Ptr{Nothing}, 1} = [C_NULL]
    local result::cudaError_t

    result = cudaMalloc(device_pointer_array, sizeof(jl_array))
    @assert (result === cudaSuccess) ("cudaMalloc() error: " * unsafe_string(cudaGetErrorName(result)))

    # allocate new array in device memory
    local devptr::Ptr{Nothing} = pop!(device_pointer_array)

    # copy data to the array in device memory
    result = cudaMemcpy(devptr, jl_array, sizeof(jl_array), cudaMemcpyHostToDevice)
    @assert (result === cudaSuccess) ("cudaMemcpy() error: " * unsafe_string(cudaGetErrorName(result)))

    local ca::CUDAArray = CUDAArray(devptr, size(jl_array), true, eltype(jl_array))
    finalizer(deallocate!, ca)
    return ca
end

# free pointers not yet deallocated
function deallocate!(ca::CUDAArray)
    if (!ca.freed && ca.is_device)
        cudaFree(ca.ptr)
        ca.freed = true
    end
    nothing
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_memcpy!(dst::Array{T}, doffset::Csize_t, src::CUDAArray, soffset::Csize_t, n::Integer)::Array where T
    if (src.is_device)
        cudaMemcpy(Ptr{Nothing}(pointer(dst, doffset + 1)), src.ptr + (soffset * sizeof(T)), sizeof(T) * n, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            Ptr{Nothing}(pointer(dst, doffset + 1)),
            src.ptr + (soffset * sizeof(T)),
            Csize_t(sizeof(T) * n))
    end
    return dst
end

# copy 'n' elements (offsets are zero indexed)
function unsafe_memcpy!(dst::CUDAArray, doffset::Csize_t, src::Array{T}, soffset::Csize_t, n::Integer)::CUDAArray where T
    if (dst.is_device)
        cudaMemcpy(dst.ptr + (doffset * sizeof(T)), Ptr{Nothing}(pointer(src, soffset + 1)), sizeof(T) * n, cudaMemcpyHostToDevice)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t),
            dst.ptr + (doffset * sizeof(T)),
            Ptr{Nothing}(pointer(src, soffset + 1)),
            Csize_t(sizeof(T) * n))
    end
    return dst
end

unsafe_memcpy!(dst::Array, doffset::Integer, src::CUDAArray, soffset::Integer, n::Integer) = unsafe_memcpy!(dst, Csize_t(doffset), src, Csize_t(soffset), n)
unsafe_memcpy!(dst::CUDAArray, doffset::Integer, src::Array, soffset::Integer, n::Integer) = unsafe_memcpy!(dst, Csize_t(doffset), src, Csize_t(soffset), n)

function unsafe_memcpy!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    local src_byte_size::Csize_t = sizeof(src.element_type) * reduce(*, src.size)
    if (src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToDevice)
    elseif (!src.is_device && dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyHostToDevice)
    elseif (src.is_device && !dst.is_device)
        cudaMemcpy(dst.ptr, src.ptr, src_byte_size, cudaMemcpyDeviceToHost)
    else
        ccall(:memcpy, Ptr{Nothing}, (Ptr{Nothing}, Ptr{Nothing}, Csize_t), dst.ptr, src.ptr, src_byte_size)
    end
    return dst
end

function memcpy!(dst::Array{T}, src::CUDAArray)::Array where T
    @assert (size(dst) == src.size)
    if (src.is_device)
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyDeviceToHost)
    else
        cudaMemcpy(dst, src.ptr, sizeof(dst), cudaMemcpyHostToHost)
    end
    return dst
end

function memcpy!(dst::CUDAArray, src::Array{T})::CUDAArray where T
    @assert (dst.size == size(src))
    if (dst.is_device)
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToDevice)
    else
        cudaMemcpy(dst.ptr, src, sizeof(src), cudaMemcpyHostToHost)
    end
    return dst
end

function memcpy!(dst::CUDAArray, src::CUDAArray)::CUDAArray
    @assert ((sizeof(dst.element_type) * reduce(*, dst.size))
            >= (sizeof(src.element_type) * reduce(*, src.size)))
    return unsafe_memcpy!(dst, src)
end
