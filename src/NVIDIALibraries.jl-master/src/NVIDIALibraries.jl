#=*
* Load NVIDIA library definitions and functions.
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

module NVIDIALibraries

export @using_nvidialib_settings, get_cuda_toolkit_versions

module DeviceArray
export CUDAArray

include("cuda_array.jl")

end # DeviceArray

module VectorTypes

include("cuda_vector_types.jl")

end # VectorTypes

module ComplexTypes
using ..VectorTypes

include("cuda_complex.jl")

end # ComplexTypes

module CUDA_8_0
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/8.0/libcuda_8.0_exports.jl")
    include("CUDA/8.0/libcuda_8.0_constants.jl")
    include("CUDA/8.0/libcuda_8.0_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 8.0

module CUDA_9_0
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/9.0/libcuda_9.0_exports.jl")
    include("CUDA/9.0/libcuda_9.0_constants.jl")
    include("CUDA/9.0/libcuda_9.0_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 9.0

module CUDA_9_1
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/9.1/libcuda_9.1_exports.jl")
    include("CUDA/9.1/libcuda_9.1_constants.jl")
    include("CUDA/9.1/libcuda_9.1_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 9.1

module CUDA_9_2
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/9.2/libcuda_9.2_exports.jl")
    include("CUDA/9.2/libcuda_9.2_constants.jl")
    include("CUDA/9.2/libcuda_9.2_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 9.2

module CUDA_10_0
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/10.0/libcuda_10.0_exports.jl")
    include("CUDA/10.0/libcuda_10.0_constants.jl")
    include("CUDA/10.0/libcuda_10.0_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 10.0

module CUDA_10_1
    using ..DeviceArray
    using ..VectorTypes

    include("load_cuda.jl")
    include("CUDA/10.1/libcuda_10.1_exports.jl")
    include("CUDA/10.1/libcuda_10.1_constants.jl")
    include("CUDA/10.1/libcuda_10.1_functions.jl")
    include("CUDA/high_level_cuda.jl")
end # CUDA 10.1

module CUDARuntime_8_0
using ..CUDA_8_0
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/8.0/load_cudart_8.0.jl")
include("CUDART/8.0/libcudart_8.0_exports.jl")
include("CUDART/8.0/libcudart_8.0_constants.jl")
include("CUDART/8.0/libcudart_8.0_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 8.0

module CUDARuntime_9_0
using ..CUDA_9_0
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/9.0/load_cudart_9.0.jl")
include("CUDART/9.0/libcudart_9.0_exports.jl")
include("CUDART/9.0/libcudart_9.0_constants.jl")
include("CUDART/9.0/libcudart_9.0_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 9.0

module CUDARuntime_9_1
using ..CUDA_9_1
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/9.1/load_cudart_9.1.jl")
include("CUDART/9.1/libcudart_9.1_exports.jl")
include("CUDART/9.1/libcudart_9.1_constants.jl")
include("CUDART/9.1/libcudart_9.1_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 9.1

module CUDARuntime_9_2
using ..CUDA_9_2
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/9.2/load_cudart_9.2.jl")
include("CUDART/9.2/libcudart_9.2_exports.jl")
include("CUDART/9.2/libcudart_9.2_constants.jl")
include("CUDART/9.2/libcudart_9.2_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 9.2

module CUDARuntime_10_0
using ..CUDA_10_0
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/10.0/load_cudart_10.0.jl")
include("CUDART/10.0/libcudart_10.0_exports.jl")
include("CUDART/10.0/libcudart_10.0_constants.jl")
include("CUDART/10.0/libcudart_10.0_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 10.0

module CUDARuntime_10_1
using ..CUDA_10_1
using ..VectorTypes
# CUDA runtime API is implemented over CUDA driver API
include("cuda_vector_types_exports.jl")
include("CUDART/10.1/load_cudart_10.1.jl")
include("CUDART/10.1/libcudart_10.1_exports.jl")
include("CUDART/10.1/libcudart_10.1_constants.jl")
include("CUDART/10.1/libcudart_10.1_functions.jl")
include("CUDART/high_level_cudart.jl")
end # CUDA Runtime 10.1

module CUBLAS_8_0
using ..CUDA_8_0
using ..CUDARuntime_8_0
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/8.0/load_cublas_8.0.jl")
include("CUBLAS/8.0/libcublas_8.0_exports.jl")
include("CUBLAS/8.0/libcublas_8.0_constants.jl")
include("CUBLAS/8.0/libcublas_8.0_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 8.0

module CUBLAS_9_0
using ..CUDA_9_0
using ..CUDARuntime_9_0
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/9.0/load_cublas_9.0.jl")
include("CUBLAS/9.0/libcublas_9.0_exports.jl")
include("CUBLAS/9.0/libcublas_9.0_constants.jl")
include("CUBLAS/9.0/libcublas_9.0_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 9.0

module CUBLAS_9_1
using ..CUDA_9_1
using ..CUDARuntime_9_1
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/9.1/load_cublas_9.1.jl")
include("CUBLAS/9.1/libcublas_9.1_exports.jl")
include("CUBLAS/9.1/libcublas_9.1_constants.jl")
include("CUBLAS/9.1/libcublas_9.1_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 9.1

module CUBLAS_9_2
using ..CUDA_9_2
using ..CUDARuntime_9_2
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/9.2/load_cublas_9.2.jl")
include("CUBLAS/9.2/libcublas_9.2_exports.jl")
include("CUBLAS/9.2/libcublas_9.2_constants.jl")
include("CUBLAS/9.2/libcublas_9.2_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 9.2

module CUBLAS_10_0
using ..CUDA_10_0
using ..CUDARuntime_10_0
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/10.0/load_cublas_10.0.jl")
include("CUBLAS/10.0/libcublas_10.0_exports.jl")
include("CUBLAS/10.0/libcublas_10.0_constants.jl")
include("CUBLAS/10.0/libcublas_10.0_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 10.0

module CUBLAS_10_1
using ..CUDA_10_1
using ..CUDARuntime_10_1
using ..VectorTypes
using ..ComplexTypes
using ..DeviceArray
# CUBLAS should be loaded after CUDA/CUDA Runtime definitions are loaded
# Export CUDA Complex and vector types from the CUBLAS module
include("cuda_vector_types_exports.jl")
include("cuda_complex_exports.jl")
include("cuda_library_types.jl")
include("CUBLAS/10.1/load_cublas_10.1.jl")
include("CUBLAS/10.1/libcublas_10.1_exports.jl")
include("CUBLAS/10.1/libcublas_10.1_constants.jl")
include("CUBLAS/10.1/libcublas_10.1_functions.jl")
include("CUBLAS/high_level_cublas.jl")
end # CUBLAS 10.1

using Printf

function set_default_nvlib_settings()
    local file::IOStream = open("nvlib_julia.conf", "w")
    write(file, ("# NVIDIALibraries.jl settings for importing submodules\n"
                * "# each line must have a colon delimiter\n\n"))
    local mods::Array{String, 1} = Array{String, 1}(["NVIDIALibraries.CUDA",
                                    "NVIDIALibraries.CUDARuntime",
                                    "NVIDIALibraries.CUBLAS"])
    # Determine the latest installed CUDA toolkit version
    local latest_cuda_version::VersionNumber
    local latest_cuda_version_string::String
    if (Sys.iswindows())
        latest_cuda_version = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    elseif (Sys.isapple())
        latest_cuda_version = reduce(max, map(VersionNumber, map((function(name::String)
                                                                        return name[6:end]
                                                                    end),
                                                                    readdir("/Developer/NVIDIA/"))))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    elseif (Sys.islinux())
        latest_cuda_version = reduce(max,
                                    map(VersionNumber,
                                        map((function(name::String)
                                                return name[6:end]
                                            end),
                                            collect(i for i in readdir("/usr/local/")
                                                    if occursin("cuda-", i)))))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    end
    for mod in mods
        write(file, (mod * ": " * latest_cuda_version_string * "\n"))
    end
    flush(file)
    close(file)
    return nothing
end

function set_default_nvlib_settings(fn::String)
    local file::IOStream = open(fn, "w")
    write(file, ("# NVIDIALibraries.jl settings for importing submodules\n"
                * "# each line must have a colon delimiter\n\n"))
    local mods::Array{String, 1} = Array{String, 1}(["NVIDIALibraries.CUDA",
                                    "NVIDIALibraries.CUDARuntime",
                                    "NVIDIALibraries.CUBLAS"])
    # Determine the latest installed CUDA toolkit version
    local latest_cuda_version::VersionNumber
    local latest_cuda_version_string::String
    if (Sys.iswindows())
        latest_cuda_version = reduce(max, map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    elseif (Sys.isapple())
        latest_cuda_version = reduce(max, map(VersionNumber, map((function(name::String)
                                                                        return name[6:end]
                                                                    end),
                                                                    readdir("/Developer/NVIDIA/"))))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    elseif (Sys.islinux())
        latest_cuda_version = reduce(max,
                                    map(VersionNumber,
                                        map((function(name::String)
                                                return name[6:end]
                                            end),
                                            collect(i for i in readdir("/usr/local/")
                                                    if occursin("cuda-", i)))))
        latest_cuda_version_string = @sprintf("%i.%i", latest_cuda_version.major, latest_cuda_version.minor)
    end
    for mod in mods
        write(file, (mod * ": " * latest_cuda_version_string * "\n"))
    end
    flush(file)
    close(file)
    return nothing
end

function get_nvlib_settings()::Array{String, 1}
    local file::IOStream = open("nvlib_julia.conf", "r")
    local mods::Array{String, 1} = Array{String, 1}()
    local lib_name::String
    local colon_count::Int
    local ver_num::VersionNumber
    local line_number::Int = 1
    while (!eof(file))
        local line::String = readline(file)
        if ((length(strip(line)) > 0) && (line[1] !== '#'))
            if (!occursin(",", line))
                if (occursin(":", line))
                    colon_count = count(i->(i === ':'), line)
                    @assert (colon_count == 1) ("get_nvlib_settings() error: found "
                                                * string(colon_count) * " colons in line "
                                                * string(line_number) * "!")
                    local mod_str_array::Array{String, 1} = map(String, split(line, ":"))
                    lib_name = strip(popfirst!(mod_str_array))
                    ver_num = VersionNumber(strip(popfirst!(mod_str_array)))
                    if (lib_name in ("NVIDIALibraries.CUDA",
                                    "NVIDIALibraries.CUDARuntime",
                                    "NVIDIALibraries.CUBLAS"))
                        push!(mods, (@sprintf("%s_%i_%i", lib_name, ver_num.major, ver_num.minor)))
                    end
                else
                    error("get_nvlib_settings() error: expected ':' in line " * string(line_number) * "!")
                end
            else
                error("get_nvlib_settings() error: found invalid character ',' in line " * string(line_number) * "!")
            end
        end
        line_number = line_number + 1
    end
    close(file)
    return mods
end

function get_nvlib_settings(fn::String)::Array{String, 1}
    @assert (fn !== "") ("get_nvlib_settings() error: filename can't be an empty string!")
    local file::IOStream = open("nvlib_julia.conf", "r")
    local mods::Array{String, 1} = Array{String, 1}()
    local lib_name::String
    local colon_count::Int
    local ver_num::VersionNumber
    local line_number::Int = 1
    while (!eof(file))
        local line::String = readline(file)
        if ((length(strip(line)) > 0) && (line[1] !== '#'))
            if (!occursin(",", line))
                if (occursin(":", line))
                    colon_count = count(i->(i === ':'), line)
                    @assert (colon_count == 1) ("get_nvlib_settings() error: found "
                                                * string(colon_count) * " colons in line "
                                                * string(line_number) * "!")
                    local mod_str_array::Array{String, 1} = map(String, split(line, ":"))
                    lib_name = strip(popfirst!(mod_str_array))
                    ver_num = VersionNumber(strip(popfirst!(mod_str_array)))
                    if (lib_name in ("NVIDIALibraries.CUDA",
                                    "NVIDIALibraries.CUDARuntime",
                                    "NVIDIALibraries.CUBLAS"))
                        push!(mods, (@sprintf("%s_%i_%i", lib_name, ver_num.major, ver_num.minor)))
                    end
                else
                    error("get_nvlib_settings() error: expected ':' in line " * string(line_number) * "!")
                end
            else
                error("get_nvlib_settings() error: found invalid character ',' in line " * string(line_number) * "!")
            end
        end
        line_number = line_number + 1
    end
    close(file)
    return mods
end

macro using_nvidialib_settings()
    if (!isfile("nvlib_julia.conf"))
        set_default_nvlib_settings()
    end
    return Expr(:block,
                collect(Expr(:using, Expr(:., i...))
                        for i in collect(map(Symbol, i)
                                        for i in collect(split(i, ".")
                                                        for i in get_nvlib_settings())))...)
end

macro using_nvidialib_settings(filename::String)
    if (!isfile(filename))
        set_default_nvlib_settings(filename)
    end
    return Expr(:block,
                collect(Expr(:using, Expr(:., i...))
                        for i in collect(map(Symbol, i)
                                        for i in collect(split(i, ".")
                                                        for i in get_nvlib_settings(filename))))...)
end

# return available CUDA Toolkit versions in no particular order
function get_cuda_toolkit_versions()::Array{VersionNumber, 1}
    # Determine installed CUDA toolkit versions
    if (Sys.iswindows())
        return map(VersionNumber, readdir("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"))
    elseif (Sys.isapple())
        return map(VersionNumber, map((function(name::String)
                                            return name[6:end]
                                        end),
                                        readdir("/Developer/NVIDIA/")))
    else
        error("get_cuda_toolkit_versions() error: unexpected operating system!")
    end
end

end
