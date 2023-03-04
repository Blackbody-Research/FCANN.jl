#=*
* CUDA vector type definitions
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

include("cuda_vector_types_exports.jl")

# attempt to align C struct to replicate alignment CUDA uses
function align_struct(jltype::DataType, alignm::Int)::Cuint
    local alignm_bitfields::Cuint = unsafe_load(Ptr{Cuint}(jltype.layout) + 4)
    # the 'alignment' field is only 9 bits
    if (alignm > 0x01ff)
        error("align_struct(): alignment cannot exceed 0x01ff!")
    end
    local new_alignm_bitfields::Cuint = (((alignm_bitfields >> 9) << 9) | Cuint(alignm))
    unsafe_store!((Ptr{Cuint}(jltype.layout) + 4), new_alignm_bitfields)
    return new_alignm_bitfields
end

struct char1
    x::Cchar
end
Base.zero(::Type{char1}) = char1(Cchar(0))
Base.zero(x::char1) = zero(typeof(x))

struct uchar1
    x::Cuchar
end
Base.zero(::Type{uchar1}) = uchar1(Cuchar(0))
Base.zero(x::uchar1) = zero(typeof(x))

struct char2
    x::Cchar
    y::Cchar
end
char2(r_num::T) where {T <: Real} = char2(Cchar(r_num), Cchar(0))
char2(c_num::T) where {T <: Complex} = char2(Cchar(c_num.re), Cchar(c_num.im))

# char2 is aligned by 2 bytes
align_struct(char2, 2)

Base.zero(::Type{char2}) = char2(Cchar(0), Cchar(0))
Base.zero(x::char2) = zero(typeof(x))

Base.one(::Type{char2}) = char2(Cchar(1), Cchar(0))
Base.one(x::char2) = one(typeof(x))

struct uchar2
    x::Cuchar
    y::Cuchar
end
# uchar2 is aligned by 2 bytes
align_struct(uchar2, 2)

Base.zero(::Type{uchar2}) = uchar2(Cuchar(0), Cuchar(0))
Base.zero(x::uchar2) = zero(typeof(x))

struct char3
    x::Cchar
    y::Cchar
    z::Cchar
end
Base.zero(::Type{char3}) = char3(Cchar(0), Cchar(0), Cchar(0))
Base.zero(x::char3) = zero(typeof(x))

struct uchar3
    x::Cuchar
    y::Cuchar
    z::Cuchar
end
Base.zero(::Type{uchar3}) = uchar3(Cuchar(0), Cuchar(0), Cuchar(0))
Base.zero(x::uchar3) = zero(typeof(x))

struct char4
    x::Cchar
    y::Cchar
    z::Cchar
    w::Cchar
end
# char4 is aligned by 4 bytes
align_struct(char4, 4)

Base.zero(::Type{char4}) = char4(Cchar(0), Cchar(0), Cchar(0), Cchar(0))
Base.zero(x::char4) = zero(typeof(x))

struct uchar4
    x::Cuchar
    y::Cuchar
    z::Cuchar
    w::Cuchar
end
# uchar4 is aligned by 4 bytes
align_struct(uchar4, 4)

Base.zero(::Type{uchar4}) = uchar4(Cuchar(0), Cuchar(0), Cuchar(0), Cuchar(0))
Base.zero(x::uchar4) = zero(typeof(x))

struct short1
    x::Cshort
end
Base.zero(::Type{short1}) = short1(Cshort(0))
Base.zero(x::short1) = zero(typeof(x))

struct ushort1
    x::Cushort
end
Base.zero(::Type{ushort1}) = ushort1(Cushort(0))
Base.zero(x::ushort1) = zero(typeof(x))

struct short2
    x::Cshort
    y::Cshort
end
# short2 is aligned by 4 bytes
align_struct(short2, 4)

Base.zero(::Type{short2}) = short2(Cshort(0), Cshort(0))
Base.zero(x::short2) = zero(typeof(x))

struct ushort2
    x::Cushort
    y::Cushort
end
# ushort2 is aligned by 4 bytes
align_struct(ushort2, 4)

Base.zero(::Type{ushort2}) = ushort2(Cushort(0), Cushort(0))
Base.zero(x::ushort2) = zero(typeof(x))

struct short3
    x::Cshort
    y::Cshort
    z::Cshort
end
Base.zero(::Type{short3}) = short3(Cshort(0), Cshort(0), Cshort(0))
Base.zero(x::short3) = zero(typeof(x))

struct ushort3
    x::Cushort
    y::Cushort
    z::Cushort
end
Base.zero(::Type{ushort3}) = ushort3(Cushort(0), Cushort(0), Cushort(0))
Base.zero(x::ushort3) = zero(typeof(x))

struct short4
    x::Cshort
    y::Cshort
    z::Cshort
    w::Cshort
end
# short4 is aligned by 8 bytes
align_struct(short4, 8)

Base.zero(::Type{short4}) = short4(Cshort(0), Cshort(0), Cshort(0), Cshort(0))
Base.zero(x::short4) = zero(typeof(x))

struct ushort4
    x::Cushort
    y::Cushort
    z::Cushort
    w::Cushort
end
# ushort4 is aligned by 8 bytes
align_struct(ushort4, 8)

Base.zero(::Type{ushort4}) = ushort4(Cushort(0), Cushort(0), Cushort(0), Cushort(0))
Base.zero(x::ushort4) = zero(typeof(x))

struct int1
    x::Cint
end
Base.zero(::Type{int1}) = int1(Cint(0))
Base.zero(x::int1) = zero(typeof(x))

struct uint1
    x::Cuint
end
Base.zero(::Type{uint1}) = uint1(Cuint(0))
Base.zero(x::uint1) = zero(typeof(x))

struct int2
    x::Cint
    y::Cint
end
# int2 is aligned by 8 bytes
align_struct(int2, 8)

Base.zero(::Type{int2}) = int2(Cint(0), Cint(0))
Base.zero(x::int2) = zero(typeof(x))

struct uint2
    x::Cuint
    y::Cuint
end
# uint2 is aligned by 8 bytes
align_struct(uint2, 8)

Base.zero(::Type{uint2}) = uint2(Cuint(0), Cuint(0))
Base.zero(x::uint2) = zero(typeof(x))

struct int3
    x::Cint
    y::Cint
    z::Cint
end
Base.zero(::Type{int3}) = int3(Cint(0), Cint(0), Cint(0))
Base.zero(x::int3) = zero(typeof(x))

struct uint3
    x::Cuint
    y::Cuint
    z::Cuint
end
Base.zero(::Type{uint3}) = uint3(Cuint(0), Cuint(0), Cuint(0))
Base.zero(x::uint3) = zero(typeof(x))

struct int4
    x::Cint
    y::Cint
    z::Cint
    w::Cint
end
# int4 is aligned by 16 bytes
align_struct(int4, 16)

Base.zero(::Type{int4}) = int4(Cint(0), Cint(0), Cint(0), Cint(0))
Base.zero(x::int4) = zero(typeof(x))

struct uint4
    x::Cuint
    y::Cuint
    z::Cuint
    w::Cuint
end
# uint4 is aligned by 16 bytes
align_struct(uint4, 16)

Base.zero(::Type{uint4}) = uint4(Cuint(0), Cuint(0), Cuint(0), Cuint(0))
Base.zero(x::uint4) = zero(typeof(x))

struct long1
    x::Clong
end
Base.zero(::Type{long1}) = long1(Clong(0))
Base.zero(x::long1) = zero(typeof(x))

struct ulong1
    x::Culong
end
Base.zero(::Type{ulong1}) = ulong1(Culong(0))
Base.zero(x::ulong1) = zero(typeof(x))

struct long2
    x::Clong
    y::Clong
end
# long2 is aligned by 8 bytes on Windows, otherwise (2 * sizeof(Clong)) bytes
if (Sys.iswindows())
    align_struct(long2, 8)
else
    align_struct(long2, (2 * sizeof(Clong)))
end

Base.zero(::Type{long2}) = long2(Clong(0), Clong(0))
Base.zero(x::long2) = zero(typeof(x))

struct ulong2
    x::Culong
    y::Culong
end
# long2 is aligned by 8 bytes on Windows, otherwise (2 * sizeof(Clong)) bytes
if (Sys.iswindows())
    align_struct(ulong2, 8)
else
    align_struct(ulong2, (2 * sizeof(Culong)))
end

Base.zero(::Type{ulong2}) = ulong2(Culong(0), Culong(0))
Base.zero(x::ulong2) = zero(typeof(x))

struct long3
    x::Clong
    y::Clong
    z::Clong
end
Base.zero(::Type{long3}) = long3(Clong(0), Clong(0), Clong(0))
Base.zero(x::long3) = zero(typeof(x))

struct ulong3
    x::Culong
    y::Culong
    z::Culong
end
Base.zero(::Type{ulong3}) = ulong3(Culong(0), Culong(0), Culong(0))
Base.zero(x::ulong3) = zero(typeof(x))

struct long4
    x::Clong
    y::Clong
    z::Clong
    w::Clong
end
# long4 is aligned by 16 bytes
align_struct(long4, 16)

Base.zero(::Type{long4}) = long4(Clong(0), Clong(0), Clong(0), Clong(0))
Base.zero(x::long4) = zero(typeof(x))

struct ulong4
    x::Culong
    y::Culong
    z::Culong
    w::Culong
end
# ulong4 is aligned by 16 bytes
align_struct(ulong4, 16)

Base.zero(::Type{ulong4}) = ulong4(Culong(0), Culong(0), Culong(0), Culong(0))
Base.zero(x::ulong4) = zero(typeof(x))

struct float1
    x::Cfloat
end
Base.zero(::Type{float1}) = float1(Cfloat(0))
Base.zero(x::float1) = zero(typeof(x))

struct float2
    x::Cfloat
    y::Cfloat
end
float2(r_num::T) where {T <: Real} = float2(Cfloat(r_num), Cfloat(0))
float2(c_num::T) where {T <: Complex} = float2(Cfloat(c_num.re), Cfloat(c_num.im))

# float2 is aligned by 8 bytes
align_struct(float2, 8)

Base.zero(::Type{float2}) = float2(Cfloat(0), Cfloat(0))
Base.zero(x::float2) = zero(typeof(x))

Base.one(::Type{float2}) = float2(Cfloat(1), Cfloat(0))
Base.one(x::float2) = one(typeof(x))

struct float3
    x::Cfloat
    y::Cfloat
    z::Cfloat
end
Base.zero(::Type{float3}) = float3(Cfloat(0), Cfloat(0), Cfloat(0))
Base.zero(x::float3) = zero(typeof(x))

struct float4
    x::Cfloat
    y::Cfloat
    z::Cfloat
    w::Cfloat
end
# float4 is aligned by 16 bytes
align_struct(float4, 16)

Base.zero(::Type{float4}) = float4(Cfloat(0), Cfloat(0), Cfloat(0), Cfloat(0))
Base.zero(x::float4) = zero(typeof(x))

struct longlong1
    x::Clonglong
end
Base.zero(::Type{longlong1}) = longlong1(Clonglong(0))
Base.zero(x::longlong1) = zero(typeof(x))

struct ulonglong1
    x::Culonglong
end
Base.zero(::Type{ulonglong1}) = ulonglong1(Culonglong(0))
Base.zero(x::ulonglong1) = zero(typeof(x))

struct longlong2
    x::Clonglong
    y::Clonglong
end
# longlong2 is aligned by 16 bytes
align_struct(longlong2, 16)

Base.zero(::Type{longlong2}) = longlong2(Clonglong(0), Clonglong(0))
Base.zero(x::longlong2) = zero(typeof(x))

struct ulonglong2
    x::Culonglong
    y::Culonglong
end
# ulonglong2 is aligned by 16 bytes
align_struct(ulonglong2, 16)

Base.zero(::Type{ulonglong2}) = ulonglong2(Culonglong(0), Culonglong(0))
Base.zero(x::ulonglong2) = zero(typeof(x))

struct longlong3
    x::Clonglong
    y::Clonglong
    z::Clonglong
end
Base.zero(::Type{longlong3}) = longlong3(Clonglong(0), Clonglong(0), Clonglong(0))
Base.zero(x::longlong3) = zero(typeof(x))

struct ulonglong3
    x::Culonglong
    y::Culonglong
    z::Culonglong
end
Base.zero(::Type{ulonglong3}) = ulonglong3(Culonglong(0), Culonglong(0), Culonglong(0))
Base.zero(x::ulonglong3) = zero(typeof(x))

struct longlong4
    x::Clonglong
    y::Clonglong
    z::Clonglong
    w::Clonglong
end
# longlong4 is aligned by 16 bytes
align_struct(longlong4, 16)

Base.zero(::Type{longlong4}) = longlong4(Clonglong(0), Clonglong(0), Clonglong(0), Clonglong(0))
Base.zero(x::longlong4) = zero(typeof(x))

struct ulonglong4
    x::Culonglong
    y::Culonglong
    z::Culonglong
    w::Culonglong
end
# ulonglong4 is aligned by 16 bytes
align_struct(ulonglong4, 16)

Base.zero(::Type{ulonglong4}) = ulonglong4(Culonglong(0), Culonglong(0), Culonglong(0), Culonglong(0))
Base.zero(x::ulonglong4) = zero(typeof(x))

struct double1
    x::Cdouble
end
Base.zero(::Type{double1}) = double1(Cdouble(0))
Base.zero(x::double1) = zero(typeof(x))

struct double2
    x::Cdouble
    y::Cdouble
end
double2(r_num::T) where {T <: Real} = double2(Cdouble(r_num), Cdouble(0))
double2(c_num::T) where {T <: Complex} = double2(Cdouble(c_num.re), Cdouble(c_num.im))
# double2 is aligned by 16 bytes
align_struct(double2, 16)

Base.zero(::Type{double2}) = double2(Cdouble(0), Cdouble(0))
Base.zero(x::double2) = zero(typeof(x))

Base.one(::Type{double2}) = double2(Cdouble(1), Cdouble(0))
Base.one(x::double2) = one(typeof(x))

struct double3
    x::Cdouble
    y::Cdouble
    z::Cdouble
end
Base.zero(::Type{double3}) = double3(Cdouble(0), Cdouble(0), Cdouble(0))
Base.zero(x::double3) = zero(typeof(x))

struct double4
    x::Cdouble
    y::Cdouble
    z::Cdouble
    w::Cdouble
end
# double4 is aligned by 16 bytes
align_struct(double4, 16)

Base.zero(::Type{double4}) = double4(Cdouble(0), Cdouble(0), Cdouble(0), Cdouble(0))
Base.zero(x::double4) = zero(typeof(x))

struct dim3
    x::Cuint
    y::Cuint
    z::Cuint
end

Base.zero(::Type{dim3}) = dim3(Cuint(0), Cuint(0), Cuint(0))
Base.zero(x::dim3) = zero(typeof(x))

dim3(x::Integer) = dim3(Cuint(x), Cuint(1), Cuint(1))
dim3(x::Integer, y::Integer) = dim3(Cuint(x), Cuint(y), Cuint(1))
dim3(x::Integer, y::Integer, z::Integer) = dim3(Cuint(x), Cuint(y), Cuint(z))

# CUDA half precision vector types from 'cuda_fp16.hpp'

# type alias '__half' to 'Float16'
const __half = Float16

struct __half2
    x::Float16
    y::Float16
end
# __half2 is aligned by 4 bytes
align_struct(__half2, 4)

Base.zero(::Type{__half2}) = __half2(Float16(0), Float16(0))
Base.zero(x::__half2) = zero(typeof(x))

