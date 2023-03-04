#=*
* Run NVIDIALibraries.jl tests
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

using Test
using Printf

using NVIDIALibraries

@using_nvidialib_settings()

c_int_array = zeros(Cint, 1)
cuDriverGetVersion(c_int_array)
println("CUDA driver version: ", c_int_array[1])
cuda_driver_version = c_int_array[1]

# run CUDA library driver tests corresponding to the installed CUDA driver version
include((@sprintf("%i.%i",
                    Int(floor(0.001 * cuda_driver_version)),
                    Int(floor(0.1 * (cuda_driver_version % 1000)))))
        * "/libcuda_"
        * (@sprintf("%i.%i",
                    Int(floor(0.001 * cuda_driver_version)),
                    Int(floor(0.1 * (cuda_driver_version % 1000)))))
        * "_function_tests.jl")

let
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

    include(latest_cuda_version_string * "/libcudart_" * latest_cuda_version_string * "_function_tests.jl")
end

include("cuda_complex_tests.jl")

