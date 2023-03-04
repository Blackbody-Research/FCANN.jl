#=*
* CUDA Complex definitions and functions
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

include("cuda_complex_exports.jl")

const cuFloatComplex = float2

function cuCrealf(x::cuFloatComplex)::Float32
    return x.x
end

function cuCimagf(x::cuFloatComplex)::Float32
    return x.y
end

function make_cuFloatComplex(r::Float32, i::Float32)::cuFloatComplex
    return cuFloatComplex(r, i)
end

function cuConjf(x::cuFloatComplex)::cuFloatComplex
    return cuFloatComplex(x.x, -x.y)
end

function cuCaddf(x::cuFloatComplex, y::cuFloatComplex)::cuFloatComplex
    return cuFloatComplex(x.x + y.x, x.y + y.y)
end

function cuCsubf(x::cuFloatComplex, y::cuFloatComplex)::cuFloatComplex
    return cuFloatComplex(x.x - y.x, x.y - y.y)
end

function cuCmulf(x::cuFloatComplex, y::cuFloatComplex)::cuFloatComplex
    return cuFloatComplex(((x.x * y.x) - (x.y * y.y)), ((x.x * y.y) + (x.y * y.x)))
end

function cuCdivf(x::cuFloatComplex, y::cuFloatComplex)::cuFloatComplex
    local s::Float32 = abs(y.x) + abs(y.y)
    local oos::Float32 = 1.0f0 / s
    local ars::Float32 = x.x * oos
    local ais::Float32 = x.y * oos
    local brs::Float32 = y.x * oos
    local bis::Float32 = y.y * oos
    s = (brs * brs) + (bis * bis)
    oos = 1.0f0 / s
    return cuFloatComplex(((ars * brs) + (ais * bis)) * oos,
                            ((ais * brs) - (ars * bis)) * oos)
end

function cuCabsf(x::cuFloatComplex)::Float32
    local a::Float32 = x.x
    local b::Float32 = x.y
    local v::Float32, w::Float32, t::Float32
    a = abs(a)
    b = abs(b)
    if (a > b)
        v = a
        w = b
    else
        v = b
        w = a
    end
    t = w / v
    t = 1.0f0 + t * t
    t = v * sqrt(t)
    if ((v == 0.0f0) || (v > 3.402823466f38) || (w > 3.402823466f38))
        t = v + w
    end
    return t
end

const cuDoubleComplex = double2

function cuCreal(x::cuDoubleComplex)::Float64
    return x.x
end

function cuCimag(x::cuDoubleComplex)::Float64
    return x.y
end

function make_cuDoubleComplex(r::Float64, i::Float64)::cuDoubleComplex
    return cuDoubleComplex(r, i)
end

function cuConj(x::cuDoubleComplex)::cuDoubleComplex
    return cuDoubleComplex(x.x, -x.y)
end

function cuCadd(x::cuDoubleComplex, y::cuDoubleComplex)::cuDoubleComplex
    return cuDoubleComplex(x.x + y.x, x.y + y.y)
end

function cuCsub(x::cuDoubleComplex, y::cuDoubleComplex)::cuDoubleComplex
    return cuDoubleComplex(x.x - y.x, x.y - y.y)
end

function cuCmul(x::cuDoubleComplex, y::cuDoubleComplex)::cuDoubleComplex
    return cuDoubleComplex(((x.x * y.x) - (x.y * y.y)), ((x.x * y.y) + (x.y * y.x)))
end

function cuCdiv(x::cuDoubleComplex, y::cuDoubleComplex)::cuDoubleComplex
    local s::Float64 = abs(y.x) + abs(y.y)
    local oos::Float64 = 1.0 / s
    local ars::Float64 = x.x * oos
    local ais::Float64 = x.y * oos
    local brs::Float64 = y.x * oos
    local bis::Float64 = y.y * oos
    s = (brs * brs) + (bis * bis)
    oos = 1.0 / s
    return cuDoubleComplex(((ars * brs) + (ais * bis)) * oos,
                            ((ais * brs) - (ars * bis)) * oos)
end

function cuCabs(x::cuDoubleComplex)::Float64
    local a::Float64 = x.x
    local b::Float64 = x.y
    local v::Float64, w::Float64, t::Float64
    a = abs(a)
    b = abs(b)
    if (a > b)
        v = a
        w = b
    else
        v = b
        w = a
    end
    t = w / v
    t = 1.0 + t * t
    t = v * sqrt(t)
    if ((v == 0.0) || (v > 1.79769313486231570e308) || (w > 1.79769313486231570e308))
        t = v + w
    end
    return t
end

const cuComplex = cuFloatComplex

function make_cuComplex(x::Float32, y::Float32)::cuComplex
    return cuComplex(x, y)
end

function cuComplexFloatToDouble(c::cuFloatComplex)::cuDoubleComplex
    return cuDoubleComplex(Float64(c.x), Float64(c.y))
end

function cuComplexDoubleToFloat(c::cuDoubleComplex)::cuFloatComplex
    return cuFloatComplex(Float32(c.x), Float32(c.y))
end

function cuCfmaf(x::cuComplex, y::cuComplex, d::cuComplex)::cuComplex
    local real_res::Float32
    local imag_res::Float32

    real_res = (x.x * y.x) + d.x
    imag_res = (x.x * y.y) + d.y

    real_res = -(x.y * y.y) + real_res
    imag_res = (x.y * y.x) + imag_res

    return cuComplex(real_res, imag_res)
end

function cuCfma(x::cuDoubleComplex, y::cuDoubleComplex, d::cuDoubleComplex)::cuDoubleComplex
    local real_res::Float64
    local imag_res::Float64

    real_res = (x.x * y.x) + d.x
    imag_res = (x.x * y.y) + d.y

    real_res = -(x.y * y.y) + real_res
    imag_res = (x.y * y.x) + imag_res

    return cuDoubleComplex(real_res, imag_res)
end


