#=*
* CUBLAS API v8.0 functions
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

# CUBLAS API functions from 'cublas_api.h'

function cublasCreate_v2(handle::Array{cublasHandle_t, 1})::cublasStatus_t
    return ccall((:cublasCreate_v2, libcublas), cublasStatus_t, (Ref{cublasHandle_t},), Base.cconvert(Ref{cublasHandle_t}, handle))
end

function cublasCreate_v2(handle::Ptr{cublasHandle_t})::cublasStatus_t
    return ccall((:cublasCreate_v2, libcublas), cublasStatus_t, (Ptr{cublasHandle_t},), handle)
end

function cublasDestroy_v2(handle::cublasHandle_t)::cublasStatus_t
    return ccall((:cublasDestroy_v2, libcublas), cublasStatus_t, (cublasHandle_t,), handle)
end

function cublasGetVersion_v2(handle::cublasHandle_t, version::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasGetVersion_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{Cint},), handle, Base.cconvert(Ref{Cint}, version))
end

function cublasGetVersion_v2(handle::cublasHandle_t, version::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasGetVersion_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cint},), handle, version)
end

function cublasGetProperty(lptype::libraryPropertyType, value::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasGetProperty, libcublas), cublasStatus_t, (libraryPropertyType, Ref{Cint},), lptype, Base.cconvert(Ref{Cint}, value))
end

function cublasGetProperty(lptype::libraryPropertyType, value::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasGetProperty, libcublas), cublasStatus_t, (libraryPropertyType, Ptr{Cint},), lptype, value)
end

function cublasSetStream_v2(handle::cublasHandle_t, streamId::cudaStream_t)::cublasStatus_t
    return ccall((:cublasSetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, cudaStream_t,), handle, streamId)
end

function cublasGetStream_v2(handle::cublasHandle_t, streamId::Array{cudaStream_t, 1})::cublasStatus_t
    return ccall((:cublasGetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{cudaStream_t},), handle, Base.cconvert(Ref{cudaStream_t}, streamId))
end

function cublasGetStream_v2(handle::cublasHandle_t, streamId::Ptr{cudaStream_t})::cublasStatus_t
    return ccall((:cublasGetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cudaStream_t},), handle, streamId)
end

function cublasGetPointerMode_v2(handle::cublasHandle_t, mode::Array{cublasPointerMode_t, 1})::cublasStatus_t
    return ccall((:cublasGetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{cublasPointerMode_t},), handle, Base.cconvert(Ref{cublasPointerMode_t}, mode))
end

function cublasGetPointerMode_v2(handle::cublasHandle_t, mode::Ptr{cublasPointerMode_t})::cublasStatus_t
    return ccall((:cublasGetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasPointerMode_t},), handle, mode)
end

function cublasSetPointerMode_v2(handle::cublasHandle_t, mode::cublasPointerMode_t)::cublasStatus_t
    return ccall((:cublasSetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasPointerMode_t,), handle, mode)
end

function cublasGetAtomicsMode(handle::cublasHandle_t, mode::Array{cublasAtomicsMode_t, 1})::cublasStatus_t
    return ccall((:cublasGetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, Ref{cublasAtomicsMode_t},), handle, Base.cconvert(Ref{cublasAtomicsMode_t}, mode))
end

function cublasGetAtomicsMode(handle::cublasHandle_t, mode::Ptr{cublasAtomicsMode_t})::cublasStatus_t
    return ccall((:cublasGetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasAtomicsMode_t},), handle, mode)
end

function cublasSetAtomicsMode(handle::cublasHandle_t, mode::cublasAtomicsMode_t)::cublasStatus_t
    return ccall((:cublasSetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, cublasAtomicsMode_t,), handle, mode)
end

function cublasSetVector(n::Cint, elemSize::Cint, x::Ptr{Cvoid}, incx::Cint, devicePtr::Ptr{Cvoid}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint,), n, elemSize, x, incx, devicePtr, incy)
end

function cublasGetVector(n::Cint, elemSize::Cint, x::Ptr{Cvoid}, incx::Cint, y::Ptr{Cvoid}, incy::Cint)::cublasStatus_t
    return ccall((:cublasGetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint,), n, elemSize, x, incx, y, incy)
end

function cublasSetMatrix(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasSetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint,), rows, cols, elemSize, A, lda, B, ldb)
end

function cublasGetMatrix(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasGetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint,), rows, cols, elemSize, A, lda, B, ldb)
end

function cublasSetVectorAsync(n::Cint, elemSize::Cint, hostPtr::Ptr{Cvoid}, incx::Cint, devicePtr::Ptr{Cvoid}, incy::Cint, stream::cudaStream_t)::cublasStatus_t
    return ccall((:cublasSetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, cudaStream_t,), n, elemSize, hostPtr, incx, devicePtr, incy, stream)
end

function cublasGetVectorAsync(n::Cint, elemSize::Cint, devicePtr::Ptr{Cvoid}, incx::Cint, hostPtr::Ptr{Cvoid}, incy::Cint, stream::cudaStream_t)::cublasStatus_t
    return ccall((:cublasGetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, cudaStream_t,), n, elemSize, devicePtr, incx, hostPtr, incy, stream)
end

function cublasSetMatrixAsync(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, stream::cudaStream_t)::cublasStatus_t
    return ccall((:cublasSetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, cudaStream_t,), rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasGetMatrixAsync(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{Cvoid}, lda::Cint, B::Ptr{Cvoid}, ldb::Cint, stream::cudaStream_t)::cublasStatus_t
    return ccall((:cublasGetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Cint, cudaStream_t,), rows, cols, elemSize, A, lda, B, ldb, stream)
end

function cublasXerbla(srName::Array{UInt8, 1}, info::Cint)::Nothing
    return ccall((:cublasXerbla, libcublas), Nothing, (Ref{UInt8}, Cint,), Base.cconvert(Ref{UInt8}, srName), info)
end

function cublasXerbla(srName::Ptr{UInt8}, info::Cint)::Nothing
    return ccall((:cublasXerbla, libcublas), Nothing, (Ptr{UInt8}, Cint,), srName, info)
end

# CUBLAS BLAS level 1 functions
function cublasNrm2Ex(handle::cublasHandle_t, n::Cint, x::Ptr{Cvoid}, xType::cudaDataType, incx::Cint, result::Ptr{Cvoid}, resultType::cudaDataType, executionType::cudaDataType)::cublasStatus_t
    return ccall((:cublasNrm2Ex, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, cudaDataType,), handle, n, x, xType, incx, result, resultType, executionType)
end

function cublasSnrm2_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, result::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, result))
end

function cublasSnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, n, x, incx, result)
end

function cublasDnrm2_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, result::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, result))
end

function cublasDnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, n, x, incx, result)
end

function cublasScnrm2_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, result::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasScnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{Cfloat}, result))
end

function cublasScnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasScnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat},), handle, n, x, incx, result)
end

function cublasDznrm2_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, result::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDznrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{Cdouble}, result))
end

function cublasDznrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDznrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble},), handle, n, x, incx, result)
end

function cublasDotEx(handle::cublasHandle_t, n::Cint, x::Ptr{Cvoid}, xType::cudaDataType, incx::Cint, y::Ptr{Cvoid}, yType::cudaDataType, incy::Cint, result::Ptr{Cvoid}, resultType::cudaDataType, executionType::cudaDataType)::cublasStatus_t
    return ccall((:cublasDotEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, cudaDataType,), handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
end

function cublasDotcEx(handle::cublasHandle_t, n::Cint, x::Ptr{Cvoid}, xType::cudaDataType, incx::Cint, y::Ptr{Cvoid}, yType::cudaDataType, incy::Cint, result::Ptr{Cvoid}, resultType::cudaDataType, executionType::cudaDataType)::cublasStatus_t
    return ccall((:cublasDotcEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, cudaDataType,), handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
end

function cublasSdot_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, result::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, result))
end

function cublasSdot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, result::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, n, x, incx, y, incy, result)
end

function cublasDdot_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, result::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, result))
end

function cublasDdot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, result::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, n, x, incx, y, incy, result)
end

function cublasCdotu_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, result::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasCdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, result))
end

function cublasCdotu_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, result::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasCdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex},), handle, n, x, incx, y, incy, result)
end

function cublasCdotc_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, result::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasCdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, result))
end

function cublasCdotc_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, result::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasCdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex},), handle, n, x, incx, y, incy, result)
end

function cublasZdotu_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, result::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, result))
end

function cublasZdotu_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, result::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), handle, n, x, incx, y, incy, result)
end

function cublasZdotc_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, result::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, result))
end

function cublasZdotc_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, result::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), handle, n, x, incx, y, incy, result)
end

function cublasScalEx(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cvoid}, alphaType::cudaDataType, x::Ptr{Cvoid}, xType::cudaDataType, incx::Cint, executionType::cudaDataType)::cublasStatus_t
    return ccall((:cublasScalEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cvoid}, cudaDataType, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType,), handle, n, alpha, alphaType, x, xType, incx, executionType)
end

function cublasSscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasSscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasSscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasSscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, n, alpha, x, incx)
end

function cublasDscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, n, alpha, x, incx)
end

function cublasCscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, n, alpha, x, incx)
end

function cublasCsscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCsscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint,), handle, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCsscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCsscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint,), handle, n, alpha, x, incx)
end

function cublasZscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, n, alpha, x, incx)
end

function cublasZdscal_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZdscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint,), handle, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZdscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZdscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint,), handle, n, alpha, x, incx)
end

function cublasAxpyEx(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cvoid}, alphaType::cudaDataType, x::Ptr{Cvoid}, xType::cudaDataType, incx::Cint, y::Ptr{Cvoid}, yType::cudaDataType, incy::Cint, executiontype::cudaDataType)::cublasStatus_t
    return ccall((:cublasAxpyEx, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cvoid}, cudaDataType, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType,), handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
end

function cublasSaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, n, alpha, x, incx, y, incy)
end

function cublasDaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, n, alpha, x, incx, y, incy)
end

function cublasCaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, n, alpha, x, incx, y, incy)
end

function cublasZaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, n, alpha, x, incx, y, incy)
end

function cublasScopy_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasScopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasScopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasScopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, n, x, incx, y, incy)
end

function cublasDcopy_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, n, x, incx, y, incy)
end

function cublasCcopy_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, n, x, incx, y, incy)
end

function cublasZcopy_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, n, x, incx, y, incy)
end

function cublasSswap_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, n, x, incx, y, incy)
end

function cublasDswap_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, n, x, incx, y, incy)
end

function cublasCswap_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, n, x, incx, y, incy)
end

function cublasZswap_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, n, x, incx, y, incy)
end

function cublasIsamax_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIsamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIsamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIsamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIdamax_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIdamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIdamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIdamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIcamax_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIcamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIcamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIcamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIzamax_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIzamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIzamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIzamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIsamin_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIsamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIsamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIsamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIdamin_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIdamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIdamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIdamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIcamin_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIcamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIcamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIcamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasIzamin_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, result::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasIzamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cint},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{Cint}, result))
end

function cublasIzamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasIzamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint},), handle, n, x, incx, result)
end

function cublasSasum_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, result::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, result))
end

function cublasSasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, n, x, incx, result)
end

function cublasDasum_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, result::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, result))
end

function cublasDasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, n, x, incx, result)
end

function cublasScasum_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, result::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasScasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{Cfloat}, result))
end

function cublasScasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasScasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat},), handle, n, x, incx, result)
end

function cublasDzasum_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, result::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDzasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{Cdouble}, result))
end

function cublasDzasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDzasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble},), handle, n, x, incx, result)
end

function cublasSrot_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, c::Array{Cfloat, 1}, s::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, c), Base.cconvert(Ref{Cfloat}, s))
end

function cublasSrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat},), handle, n, x, incx, y, incy, c, s)
end

function cublasDrot_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, c::Array{Cdouble, 1}, s::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, c), Base.cconvert(Ref{Cdouble}, s))
end

function cublasDrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble},), handle, n, x, incx, y, incy, c, s)
end

function cublasCrot_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, c::Array{Cfloat, 1}, s::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasCrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{Cfloat}, Ref{cuComplex},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{Cfloat}, c), Base.cconvert(Ref{cuComplex}, s))
end

function cublasCrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasCrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex},), handle, n, x, incx, y, incy, c, s)
end

function cublasCsrot_v2(handle::cublasHandle_t, n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, c::Array{Cfloat, 1}, s::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasCsrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{Cfloat}, Ref{Cfloat},), handle, n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{Cfloat}, c), Base.cconvert(Ref{Cfloat}, s))
end

function cublasCsrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasCsrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{Cfloat},), handle, n, x, incx, y, incy, c, s)
end

function cublasZrot_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, c::Array{Cdouble, 1}, s::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble}, Ref{cuDoubleComplex},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{Cdouble}, c), Base.cconvert(Ref{cuDoubleComplex}, s))
end

function cublasZrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex},), handle, n, x, incx, y, incy, c, s)
end

function cublasZdrot_v2(handle::cublasHandle_t, n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, c::Array{Cdouble, 1}, s::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasZdrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble}, Ref{Cdouble},), handle, n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{Cdouble}, c), Base.cconvert(Ref{Cdouble}, s))
end

function cublasZdrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasZdrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{Cdouble},), handle, n, x, incx, y, incy, c, s)
end

function cublasSrotg_v2(handle::cublasHandle_t, a::Array{Cfloat, 1}, b::Array{Cfloat, 1}, c::Array{Cfloat, 1}, s::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat},), handle, Base.cconvert(Ref{Cfloat}, a), Base.cconvert(Ref{Cfloat}, b), Base.cconvert(Ref{Cfloat}, c), Base.cconvert(Ref{Cfloat}, s))
end

function cublasSrotg_v2(handle::cublasHandle_t, a::Ptr{Cfloat}, b::Ptr{Cfloat}, c::Ptr{Cfloat}, s::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},), handle, a, b, c, s)
end

function cublasDrotg_v2(handle::cublasHandle_t, a::Array{Cdouble, 1}, b::Array{Cdouble, 1}, c::Array{Cdouble, 1}, s::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble},), handle, Base.cconvert(Ref{Cdouble}, a), Base.cconvert(Ref{Cdouble}, b), Base.cconvert(Ref{Cdouble}, c), Base.cconvert(Ref{Cdouble}, s))
end

function cublasDrotg_v2(handle::cublasHandle_t, a::Ptr{Cdouble}, b::Ptr{Cdouble}, c::Ptr{Cdouble}, s::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},), handle, a, b, c, s)
end

function cublasCrotg_v2(handle::cublasHandle_t, a::Array{cuComplex, 1}, b::Array{cuComplex, 1}, c::Array{Cfloat, 1}, s::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasCrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{cuComplex}, Ref{cuComplex}, Ref{Cfloat}, Ref{cuComplex},), handle, Base.cconvert(Ref{cuComplex}, a), Base.cconvert(Ref{cuComplex}, b), Base.cconvert(Ref{Cfloat}, c), Base.cconvert(Ref{cuComplex}, s))
end

function cublasCrotg_v2(handle::cublasHandle_t, a::Ptr{cuComplex}, b::Ptr{cuComplex}, c::Ptr{Cfloat}, s::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasCrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cfloat}, Ptr{cuComplex},), handle, a, b, c, s)
end

function cublasZrotg_v2(handle::cublasHandle_t, a::Array{cuDoubleComplex, 1}, b::Array{cuDoubleComplex, 1}, c::Array{Cdouble, 1}, s::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Ref{Cdouble}, Ref{cuDoubleComplex},), handle, Base.cconvert(Ref{cuDoubleComplex}, a), Base.cconvert(Ref{cuDoubleComplex}, b), Base.cconvert(Ref{Cdouble}, c), Base.cconvert(Ref{cuDoubleComplex}, s))
end

function cublasZrotg_v2(handle::cublasHandle_t, a::Ptr{cuDoubleComplex}, b::Ptr{cuDoubleComplex}, c::Ptr{Cdouble}, s::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cdouble}, Ptr{cuDoubleComplex},), handle, a, b, c, s)
end

function cublasSrotm_v2(handle::cublasHandle_t, n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, param::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, param))
end

function cublasSrotm_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, param::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, n, x, incx, y, incy, param)
end

function cublasDrotm_v2(handle::cublasHandle_t, n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, param::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, param))
end

function cublasDrotm_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, param::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, n, x, incx, y, incy, param)
end

function cublasSrotmg_v2(handle::cublasHandle_t, d1::Array{Cfloat, 1}, d2::Array{Cfloat, 1}, x1::Array{Cfloat, 1}, y1::Array{Cfloat, 1}, param::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat},), handle, Base.cconvert(Ref{Cfloat}, d1), Base.cconvert(Ref{Cfloat}, d2), Base.cconvert(Ref{Cfloat}, x1), Base.cconvert(Ref{Cfloat}, y1), Base.cconvert(Ref{Cfloat}, param))
end

function cublasSrotmg_v2(handle::cublasHandle_t, d1::Ptr{Cfloat}, d2::Ptr{Cfloat}, x1::Ptr{Cfloat}, y1::Ptr{Cfloat}, param::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},), handle, d1, d2, x1, y1, param)
end

function cublasDrotmg_v2(handle::cublasHandle_t, d1::Array{Cdouble, 1}, d2::Array{Cdouble, 1}, x1::Array{Cdouble, 1}, y1::Array{Cdouble, 1}, param::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble},), handle, Base.cconvert(Ref{Cdouble}, d1), Base.cconvert(Ref{Cdouble}, d2), Base.cconvert(Ref{Cdouble}, x1), Base.cconvert(Ref{Cdouble}, y1), Base.cconvert(Ref{Cdouble}, param))
end

function cublasDrotmg_v2(handle::cublasHandle_t, d1::Ptr{Cdouble}, d2::Ptr{Cdouble}, x1::Ptr{Cdouble}, y1::Ptr{Cdouble}, param::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},), handle, d1, d2, x1, y1, param)
end

# CUBLAS BLAS level 2 functions
function cublasSgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ref{Cfloat}, Ptr{Cfloat}, Cint,), handle, trans, m, n, Base.cconvert(Ref{Cfloat}, alpha), A, lda, x, incx, Base.cconvert(Ref{Cfloat}, beta), y, incy)
end

function cublasSgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ref{Cdouble}, Ptr{Cdouble}, Cint,), handle, trans, m, n, Base.cconvert(Ref{Cdouble}, alpha), A, lda, x, incx, Base.cconvert(Ref{Cdouble}, beta), y, incy)
end

function cublasDgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ref{cuComplex}, Ptr{cuComplex}, Cint,), handle, trans, m, n, Base.cconvert(Ref{cuComplex}, alpha), A, lda, x, incx, Base.cconvert(Ref{cuComplex}, beta), y, incy)
end

function cublasCgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, trans, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), A, lda, x, incx, Base.cconvert(Ref{cuDoubleComplex}, beta), y, incy)
end

function cublasZgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Array{Cfloat, 1}, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, trans, m, n, kl, ku, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Array{Cdouble, 1}, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, trans, m, n, kl, ku, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::Array{cuComplex, 1}, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, trans, m, n, kl, ku, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::Array{cuDoubleComplex, 1}, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, trans, m, n, kl, ku, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasStrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasStpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, AP, x, incx)
end

function cublasStbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::cublasStatus_t
    return ccall((:cublasStbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::cublasStatus_t
    return ccall((:cublasDtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasCtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, k, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::cublasStatus_t
    return ccall((:cublasZtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasSsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Array{Cfloat, 1}, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Array{Cdouble, 1}, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::Array{cuComplex, 1}, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasCsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::Array{cuDoubleComplex, 1}, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::Array{cuComplex, 1}, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::Array{cuDoubleComplex, 1}, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Array{Cfloat, 1}, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Array{Cdouble, 1}, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::Array{cuComplex, 1}, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::Array{cuDoubleComplex, 1}, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, beta::Array{Cfloat, 1}, y::Array{Cfloat, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)::cublasStatus_t
    return ccall((:cublasSspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasDspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, beta::Array{Cdouble, 1}, y::Array{Cdouble, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)::cublasStatus_t
    return ccall((:cublasDspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasChpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, beta::Array{cuComplex, 1}, y::Array{cuComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasChpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasZhpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::Array{cuDoubleComplex, 1}, y::Array{cuDoubleComplex, 1}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)::cublasStatus_t
    return ccall((:cublasZhpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasSger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, A::Array{Cfloat, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, A::Array{Cdouble, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, A::Array{Cfloat, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, A::Ptr{Cfloat}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasDsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, A::Array{Cdouble, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, A::Ptr{Cdouble}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasCher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{cuComplex, 1}, incx::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasZher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, x, incx, A, lda)
end

function cublasSspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, AP::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, AP))
end

function cublasSspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, AP::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, uplo, n, alpha, x, incx, AP)
end

function cublasDspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, AP::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, AP))
end

function cublasDspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, AP::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, uplo, n, alpha, x, incx, AP)
end

function cublasChpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{cuComplex, 1}, incx::Cint, AP::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasChpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint, Ref{cuComplex},), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, AP))
end

function cublasChpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint, AP::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasChpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex},), handle, uplo, n, alpha, x, incx, AP)
end

function cublasZhpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, AP::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZhpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, AP))
end

function cublasZhpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint, AP::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZhpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), handle, uplo, n, alpha, x, incx, AP)
end

function cublasSsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, A::Array{Cfloat, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)::cublasStatus_t
    return ccall((:cublasSsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, A::Array{Cdouble, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, AP::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasSspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, uplo, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, AP))
end

function cublasSspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, AP::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasSspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasDspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, AP::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, uplo, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, AP))
end

function cublasDspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, AP::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasChpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, AP::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasChpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex},), handle, uplo, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, AP))
end

function cublasChpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, AP::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasChpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex},), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasZhpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, AP::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZhpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, AP))
end

function cublasZhpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, AP::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZhpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), handle, uplo, n, alpha, x, incx, y, incy, AP)
end

# CUBLAS BLAS level 3 functions
function cublasSgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ref{Cfloat}, Ptr{Cfloat}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), A, lda, B, ldb, Base.cconvert(Ref{Cfloat}, beta), C, ldc)
end

function cublasDgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ref{Cdouble}, Ptr{Cdouble}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cdouble}, alpha), A, lda, B, ldb, Base.cconvert(Ref{Cdouble}, beta), C, ldc)
end

function cublasCgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ref{cuComplex}, Ptr{cuComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), A, lda, B, ldb, Base.cconvert(Ref{cuComplex}, beta), C, ldc)
end

function cublasZgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), A, lda, B, ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), C, ldc)
end

function cublasHgemm(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::__half, A::Ptr{__half}, lda::Cint, B::Ptr{__half}, ldb::Cint, beta::__half, C::Ptr{__half}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasHgemm, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{__half}, Ptr{__half}, Cint, Ptr{__half}, Cint, Ref{__half}, Ptr{__half}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{__half}, alpha), A, lda, B, ldb, Base.cconvert(Ref{__half}, beta), C, ldc)
end

function cublasSgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3m(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCgemm3m(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm3mEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Array{cuComplex, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{cuComplex}, beta), C, Ctype, ldc)
end

function cublasCgemm3mEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasZgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZgemm3m(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZgemm3m(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgemm3m, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasHgemm(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{__half, 1}, A::Array{__half, 1}, lda::Cint, B::Array{__half, 1}, ldb::Cint, beta::Array{__half, 1}, C::Array{__half, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasHgemm, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{__half}, Ref{__half}, Cint, Ref{__half}, Cint, Ref{__half}, Ref{__half}, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{__half}, alpha), Base.cconvert(Ref{__half}, A), lda, Base.cconvert(Ref{__half}, B), ldb, Base.cconvert(Ref{__half}, beta), Base.cconvert(Ref{__half}, C), ldc)
end

function cublasHgemm(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{__half}, A::Ptr{__half}, lda::Cint, B::Ptr{__half}, ldb::Cint, beta::Ptr{__half}, C::Ptr{__half}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasHgemm, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{__half}, Ptr{__half}, Cint, Ptr{__half}, Cint, Ptr{__half}, Ptr{__half}, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSgemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Array{Cfloat, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{Cfloat}, beta), C, Ctype, ldc)
end

function cublasSgemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Float16, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Float16, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Float16}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{Float16}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Float16}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{Float16}, beta), C, Ctype, ldc, computeType, algo)
end

# alpha and beta scalars can only have values Int32(0) or Int32(1)
# read https://docs.nvidia.com/cuda/archive/9.0/cublas/index.html#cublas-GemmEx
function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Int32, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Int32, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Int32}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{Int32}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Int32}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{Int32}, beta), C, Ctype, ldc, computeType, algo)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Float32, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Float32, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Float32}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{Float32}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Float32}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{Float32}, beta), C, Ctype, ldc, computeType, algo)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Float64, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Float64, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Float64}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{Float64}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Float64}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{Float64}, beta), C, Ctype, ldc, computeType, algo)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::cuFloatComplex, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::cuFloatComplex, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuFloatComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuFloatComplex}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuFloatComplex}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{cuFloatComplex}, beta), C, Ctype, ldc, computeType, algo)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::cuDoubleComplex, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuDoubleComplex}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), C, Ctype, ldc, computeType, algo)
end

function cublasGemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cvoid}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Ptr{Cvoid}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint, computeType::cudaDataType, algo::cublasGemmAlgo_t)::cublasStatus_t
    return ccall((:cublasGemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cvoid}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, Ptr{Cvoid}, cudaDataType, Cint, cudaDataType, cublasGemmAlgo_t,), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
end

function cublasCgemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Array{cuComplex, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), A, Atype, lda, B, Btype, ldb, Base.cconvert(Ref{cuComplex}, beta), C, Ctype, ldc)
end

function cublasCgemmEx(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, B::Ptr{Cvoid}, Btype::cudaDataType, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgemmEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cvoid}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
end

function cublasUint8gemmBias(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, transc::cublasOperation_t, m::Cint, n::Cint, k::Cint, A::Array{UInt8, 1}, A_bias::Cint, lda::Cint, B::Array{UInt8, 1}, B_bias::Cint, ldb::Cint, C::Array{UInt8, 1}, C_bias::Cint, ldc::Cint, C_mult::Cint, C_shift::Cint)::cublasStatus_t
    return ccall((:cublasUint8gemmBias, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{UInt8}, Cint, Cint, Ref{UInt8}, Cint, Cint, Ref{UInt8}, Cint, Cint, Cint, Cint,), handle, transa, transb, transc, m, n, k, Base.cconvert(Ref{UInt8}, A), A_bias, lda, Base.cconvert(Ref{UInt8}, B), B_bias, ldb, Base.cconvert(Ref{UInt8}, C), C_bias, ldc, C_mult, C_shift)
end

function cublasUint8gemmBias(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, transc::cublasOperation_t, m::Cint, n::Cint, k::Cint, A::Ptr{UInt8}, A_bias::Cint, lda::Cint, B::Ptr{UInt8}, B_bias::Cint, ldb::Cint, C::Ptr{UInt8}, C_bias::Cint, ldc::Cint, C_mult::Cint, C_shift::Cint)::cublasStatus_t
    return ccall((:cublasUint8gemmBias, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{UInt8}, Cint, Cint, Ptr{UInt8}, Cint, Cint, Ptr{UInt8}, Cint, Cint, Cint, Cint,), handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
end

function cublasSsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasDsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrkEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Array{cuComplex, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), A, Atype, lda, Base.cconvert(Ref{cuComplex}, beta), C, Ctype, ldc)
end

function cublasCsyrkEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Ptr{cuComplex}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCsyrk3mEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Array{cuComplex, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ref{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), A, Atype, lda, Base.cconvert(Ref{cuComplex}, beta), C, Ctype, ldc)
end

function cublasCsyrk3mEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Ptr{cuComplex}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{cuComplex}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{cuComplex, 1}, lda::Cint, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCherkEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Array{Cfloat, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), A, Atype, lda, Base.cconvert(Ref{Cfloat}, beta), C, Ctype, ldc)
end

function cublasCherkEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherkEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasCherk3mEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Array{Cfloat, 1}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ref{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), A, Atype, lda, Base.cconvert(Ref{Cfloat}, beta), C, Ctype, ldc)
end

function cublasCherk3mEx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cvoid}, Atype::cudaDataType, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cvoid}, Ctype::cudaDataType, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherk3mEx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint, Ptr{Cfloat}, Ptr{Cvoid}, cudaDataType, Cint,), handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
end

function cublasSsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{Cfloat}, Ref{cuComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{Cdouble}, Ref{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasChemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasChemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasChemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasChemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZhemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZhemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, side, uplo, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZhemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZhemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasStrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasStrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb)
end

function cublasStrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasStrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasDtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasDtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb)
end

function cublasDtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasDtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasCtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasCtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb)
end

function cublasCtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasCtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasZtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasZtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb)
end

function cublasZtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint)::cublasStatus_t
    return ccall((:cublasZtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasStrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasStrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasStrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasStrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasDtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasCtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasZtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasSgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, Aarray::Array{Ptr{Cfloat}, 1}, lda::Cint, Barray::Array{Ptr{Cfloat}, 1}, ldb::Cint, beta::Array{Cfloat, 1}, Carray::Array{Ptr{Cfloat}, 1}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasSgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cfloat}, Ref{Ptr{Cfloat}}, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cfloat}, Ref{Ptr{Cfloat}}, Cint, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Ptr{Cfloat}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cfloat}}, Barray), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Ptr{Cfloat}}, Carray), ldc, batchCount)
end

function cublasSgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, Barray::Ptr{Ptr{Cfloat}}, ldb::Cint, beta::Ptr{Cfloat}, Carray::Ptr{Ptr{Cfloat}}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasSgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Cint,), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasDgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, Aarray::Array{Ptr{Cdouble}, 1}, lda::Cint, Barray::Array{Ptr{Cdouble}, 1}, ldb::Cint, beta::Array{Cdouble, 1}, Carray::Array{Ptr{Cdouble}, 1}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cdouble}, Ref{Ptr{Cdouble}}, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cdouble}, Ref{Ptr{Cdouble}}, Cint, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Ptr{Cdouble}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cdouble}}, Barray), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Ptr{Cdouble}}, Carray), ldc, batchCount)
end

function cublasDgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, Barray::Ptr{Ptr{Cdouble}}, ldb::Cint, beta::Ptr{Cdouble}, Carray::Ptr{Ptr{Cdouble}}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Cint,), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasCgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, Aarray::Array{Ptr{cuComplex}, 1}, lda::Cint, Barray::Array{Ptr{cuComplex}, 1}, ldb::Cint, beta::Array{cuComplex, 1}, Carray::Array{Ptr{cuComplex}, 1}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{cuComplex}, Ref{Ptr{cuComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{Ptr{cuComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuComplex}}, Barray), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{Ptr{cuComplex}}, Carray), ldc, batchCount)
end

function cublasCgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, Barray::Ptr{Ptr{cuComplex}}, ldb::Cint, beta::Ptr{cuComplex}, Carray::Ptr{Ptr{cuComplex}}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasCgemm3mBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, Aarray::Array{Ptr{cuComplex}, 1}, lda::Cint, Barray::Array{Ptr{cuComplex}, 1}, ldb::Cint, beta::Array{cuComplex, 1}, Carray::Array{Ptr{cuComplex}, 1}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{cuComplex}, Ref{Ptr{cuComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{Ptr{cuComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuComplex}}, Barray), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{Ptr{cuComplex}}, Carray), ldc, batchCount)
end

function cublasCgemm3mBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, Barray::Ptr{Ptr{cuComplex}}, ldb::Cint, beta::Ptr{cuComplex}, Carray::Ptr{Ptr{cuComplex}}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasZgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, Aarray::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, Barray::Array{Ptr{cuDoubleComplex}, 1}, ldb::Cint, beta::Array{cuDoubleComplex, 1}, Carray::Array{Ptr{cuDoubleComplex}, 1}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{cuDoubleComplex}, Ref{Ptr{cuDoubleComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Barray), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Carray), ldc, batchCount)
end

function cublasZgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Barray::Ptr{Ptr{cuDoubleComplex}}, ldb::Cint, beta::Ptr{cuDoubleComplex}, Carray::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint,), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end

function cublasSgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, strideA::Clonglong, B::Array{Cfloat, 1}, ldb::Cint, strideB::Clonglong, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasSgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Clonglong, Ref{Cfloat}, Cint, Clonglong, Ref{Cfloat}, Ref{Cfloat}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, strideA, Base.cconvert(Ref{Cfloat}, B), ldb, strideB, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc, strideC, batchCount)
end

function cublasSgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, strideA::Clonglong, B::Ptr{Cfloat}, ldb::Cint, strideB::Clonglong, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasSgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Clonglong, Ptr{Cfloat}, Cint, Clonglong, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasDgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, strideA::Clonglong, B::Array{Cdouble, 1}, ldb::Cint, strideB::Clonglong, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Clonglong, Ref{Cdouble}, Cint, Clonglong, Ref{Cdouble}, Ref{Cdouble}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, strideA, Base.cconvert(Ref{Cdouble}, B), ldb, strideB, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc, strideC, batchCount)
end

function cublasDgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, strideA::Clonglong, B::Ptr{Cdouble}, ldb::Cint, strideB::Clonglong, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Clonglong, Ptr{Cdouble}, Cint, Clonglong, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasCgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, strideA::Clonglong, B::Array{cuComplex, 1}, ldb::Cint, strideB::Clonglong, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Clonglong, Ref{cuComplex}, Cint, Clonglong, Ref{cuComplex}, Ref{cuComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, strideA, Base.cconvert(Ref{cuComplex}, B), ldb, strideB, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc, strideC, batchCount)
end

function cublasCgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, strideA::Clonglong, B::Ptr{cuComplex}, ldb::Cint, strideB::Clonglong, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasCgemm3mStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, strideA::Clonglong, B::Array{cuComplex, 1}, ldb::Cint, strideB::Clonglong, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Clonglong, Ref{cuComplex}, Cint, Clonglong, Ref{cuComplex}, Ref{cuComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, strideA, Base.cconvert(Ref{cuComplex}, B), ldb, strideB, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc, strideC, batchCount)
end

function cublasCgemm3mStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, strideA::Clonglong, B::Ptr{cuComplex}, ldb::Cint, strideB::Clonglong, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCgemm3mStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Cint, Clonglong, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasZgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, strideA::Clonglong, B::Array{cuDoubleComplex, 1}, ldb::Cint, strideB::Clonglong, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Clonglong, Ref{cuDoubleComplex}, Cint, Clonglong, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, strideA, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, strideB, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc, strideC, batchCount)
end

function cublasZgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, strideA::Clonglong, B::Ptr{cuDoubleComplex}, ldb::Cint, strideB::Clonglong, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Clonglong, Ptr{cuDoubleComplex}, Cint, Clonglong, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasHgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Array{__half, 1}, A::Array{__half, 1}, lda::Cint, strideA::Clonglong, B::Array{__half, 1}, ldb::Cint, strideB::Clonglong, beta::Array{__half, 1}, C::Array{__half, 1}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasHgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ref{__half}, Ref{__half}, Cint, Clonglong, Ref{__half}, Cint, Clonglong, Ref{__half}, Ref{__half}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, Base.cconvert(Ref{__half}, alpha), Base.cconvert(Ref{__half}, A), lda, strideA, Base.cconvert(Ref{__half}, B), ldb, strideB, Base.cconvert(Ref{__half}, beta), Base.cconvert(Ref{__half}, C), ldc, strideC, batchCount)
end

function cublasHgemmStridedBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{__half}, A::Ptr{__half}, lda::Cint, strideA::Clonglong, B::Ptr{__half}, ldb::Cint, strideB::Clonglong, beta::Ptr{__half}, C::Ptr{__half}, ldc::Cint, strideC::Clonglong, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasHgemmStridedBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{__half}, Ptr{__half}, Cint, Clonglong, Ptr{__half}, Cint, Clonglong, Ptr{__half}, Ptr{__half}, Cint, Clonglong, Cint,), handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
end

function cublasSgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint, beta::Array{Cfloat, 1}, B::Array{Cfloat, 1}, ldb::Cint, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, transa, transb, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, B::Ptr{Cfloat}, ldb::Cint, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasDgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint, beta::Array{Cdouble, 1}, B::Array{Cdouble, 1}, ldb::Cint, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, transa, transb, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, B::Ptr{Cdouble}, ldb::Cint, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasCgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint, beta::Array{cuComplex, 1}, B::Array{cuComplex, 1}, ldb::Cint, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, transa, transb, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{cuComplex}, B::Ptr{cuComplex}, ldb::Cint, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasZgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint, beta::Array{cuDoubleComplex, 1}, B::Array{cuDoubleComplex, 1}, ldb::Cint, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{cuDoubleComplex}, B::Ptr{cuDoubleComplex}, ldb::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end

function cublasSgetrfBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cfloat}, 1}, lda::Cint, P::Array{Cint, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cfloat}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasSgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, n, A, lda, P, info, batchSize)
end

function cublasDgetrfBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cdouble}, 1}, lda::Cint, P::Array{Cint, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cdouble}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasDgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, n, A, lda, P, info, batchSize)
end

function cublasCgetrfBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuComplex}, 1}, lda::Cint, P::Array{Cint, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuComplex}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasCgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, n, A, lda, P, info, batchSize)
end

function cublasZgetrfBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, P::Array{Cint, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasZgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, n, A, lda, P, info, batchSize)
end

function cublasSgetriBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cfloat}, 1}, lda::Cint, P::Array{Cint, 1}, C::Array{Ptr{Cfloat}, 1}, ldc::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cfloat}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Ptr{Cfloat}}, C), ldc, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasSgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{Cfloat}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasDgetriBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cdouble}, 1}, lda::Cint, P::Array{Cint, 1}, C::Array{Ptr{Cdouble}, 1}, ldc::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cdouble}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Ptr{Cdouble}}, C), ldc, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasDgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{Cdouble}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasCgetriBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuComplex}, 1}, lda::Cint, P::Array{Cint, 1}, C::Array{Ptr{cuComplex}, 1}, ldc::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuComplex}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Ptr{cuComplex}}, C), ldc, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasCgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{cuComplex}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasZgetriBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, P::Array{Cint, 1}, C::Array{Ptr{cuDoubleComplex}, 1}, ldc::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, A), lda, Base.cconvert(Ref{Cint}, P), Base.cconvert(Ref{Ptr{cuDoubleComplex}}, C), ldc, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasZgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, P, C, ldc, info, batchSize)
end

function cublasSgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Array{Ptr{Cfloat}, 1}, lda::Cint, devIpiv::Array{Cint, 1}, Barray::Array{Ptr{Cfloat}, 1}, ldb::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Cint,), handle, trans, n, nrhs, Base.cconvert(Ref{Ptr{Cfloat}}, Aarray), lda, Base.cconvert(Ref{Cint}, devIpiv), Base.cconvert(Ref{Ptr{Cfloat}}, Barray), ldb, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasSgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, devIpiv::Ptr{Cint}, Barray::Ptr{Ptr{Cfloat}}, ldb::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint,), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasDgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Array{Ptr{Cdouble}, 1}, lda::Cint, devIpiv::Array{Cint, 1}, Barray::Array{Ptr{Cdouble}, 1}, ldb::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Cint,), handle, trans, n, nrhs, Base.cconvert(Ref{Ptr{Cdouble}}, Aarray), lda, Base.cconvert(Ref{Cint}, devIpiv), Base.cconvert(Ref{Ptr{Cdouble}}, Barray), ldb, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasDgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, devIpiv::Ptr{Cint}, Barray::Ptr{Ptr{Cdouble}}, ldb::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint,), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasCgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Array{Ptr{cuComplex}, 1}, lda::Cint, devIpiv::Array{Cint, 1}, Barray::Array{Ptr{cuComplex}, 1}, ldb::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Cint,), handle, trans, n, nrhs, Base.cconvert(Ref{Ptr{cuComplex}}, Aarray), lda, Base.cconvert(Ref{Cint}, devIpiv), Base.cconvert(Ref{Ptr{cuComplex}}, Barray), ldb, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasCgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, devIpiv::Ptr{Cint}, Barray::Ptr{Ptr{cuComplex}}, ldb::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint,), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasZgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, devIpiv::Array{Cint, 1}, Barray::Array{Ptr{cuDoubleComplex}, 1}, ldb::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Cint,), handle, trans, n, nrhs, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Aarray), lda, Base.cconvert(Ref{Cint}, devIpiv), Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Barray), ldb, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasZgetrsBatched(handle::cublasHandle_t, trans::cublasOperation_t, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, devIpiv::Ptr{Cint}, Barray::Ptr{Ptr{cuDoubleComplex}}, ldb::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgetrsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint,), handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
end

function cublasStrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cfloat, 1}, A::Array{Ptr{Cfloat}, 1}, lda::Cint, B::Array{Ptr{Cfloat}, 1}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasStrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cfloat}, Ref{Ptr{Cfloat}}, Cint, Ref{Ptr{Cfloat}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Ptr{Cfloat}}, A), lda, Base.cconvert(Ref{Ptr{Cfloat}}, B), ldb, batchCount)
end

function cublasStrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Ptr{Cfloat}}, lda::Cint, B::Ptr{Ptr{Cfloat}}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasStrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasDtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{Cdouble, 1}, A::Array{Ptr{Cdouble}, 1}, lda::Cint, B::Array{Ptr{Cdouble}, 1}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{Cdouble}, Ref{Ptr{Cdouble}}, Cint, Ref{Ptr{Cdouble}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Ptr{Cdouble}}, A), lda, Base.cconvert(Ref{Ptr{Cdouble}}, B), ldb, batchCount)
end

function cublasDtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Ptr{Cdouble}}, lda::Cint, B::Ptr{Ptr{Cdouble}}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasDtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasCtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuComplex, 1}, A::Array{Ptr{cuComplex}, 1}, lda::Cint, B::Array{Ptr{cuComplex}, 1}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuComplex}, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{Ptr{cuComplex}}, A), lda, Base.cconvert(Ref{Ptr{cuComplex}}, B), ldb, batchCount)
end

function cublasCtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{Ptr{cuComplex}}, lda::Cint, B::Ptr{Ptr{cuComplex}}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasCtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasZtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Array{cuDoubleComplex, 1}, A::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, B::Array{Ptr{cuDoubleComplex}, 1}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ref{cuDoubleComplex}, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{Ptr{cuDoubleComplex}}, A), lda, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, B), ldb, batchCount)
end

function cublasZtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, B::Ptr{Ptr{cuDoubleComplex}}, ldb::Cint, batchCount::Cint)::cublasStatus_t
    return ccall((:cublasZtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end

function cublasSmatinvBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cfloat}, 1}, lda::Cint, Ainv::Array{Ptr{Cfloat}, 1}, lda_inv::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cfloat}}, A), lda, Base.cconvert(Ref{Ptr{Cfloat}}, Ainv), lda_inv, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasSmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, Ainv::Ptr{Ptr{Cfloat}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasDmatinvBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{Cdouble}, 1}, lda::Cint, Ainv::Array{Ptr{Cdouble}, 1}, lda_inv::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{Cdouble}}, A), lda, Base.cconvert(Ref{Ptr{Cdouble}}, Ainv), lda_inv, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasDmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, Ainv::Ptr{Ptr{Cdouble}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasCmatinvBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuComplex}, 1}, lda::Cint, Ainv::Array{Ptr{cuComplex}, 1}, lda_inv::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuComplex}}, A), lda, Base.cconvert(Ref{Ptr{cuComplex}}, Ainv), lda_inv, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasCmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, Ainv::Ptr{Ptr{cuComplex}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasZmatinvBatched(handle::cublasHandle_t, n::Cint, A::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, Ainv::Array{Ptr{cuDoubleComplex}, 1}, lda_inv::Cint, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Cint,), handle, n, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, A), lda, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Ainv), lda_inv, Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasZmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Ainv::Ptr{Ptr{cuDoubleComplex}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint,), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end

function cublasSgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Array{Ptr{Cfloat}, 1}, lda::Cint, TauArray::Array{Ptr{Cfloat}, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Ptr{Cfloat}}, Ref{Cint}, Cint,), handle, m, n, Base.cconvert(Ref{Ptr{Cfloat}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cfloat}}, TauArray), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasSgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, TauArray::Ptr{Ptr{Cfloat}}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Cint,), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasDgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Array{Ptr{Cdouble}, 1}, lda::Cint, TauArray::Array{Ptr{Cdouble}, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Ptr{Cdouble}}, Ref{Cint}, Cint,), handle, m, n, Base.cconvert(Ref{Ptr{Cdouble}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cdouble}}, TauArray), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasDgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, TauArray::Ptr{Ptr{Cdouble}}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Cint}, Cint,), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasCgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Array{Ptr{cuComplex}, 1}, lda::Cint, TauArray::Array{Ptr{cuComplex}, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Ref{Cint}, Cint,), handle, m, n, Base.cconvert(Ref{Ptr{cuComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuComplex}}, TauArray), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasCgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, TauArray::Ptr{Ptr{cuComplex}}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Ptr{Cint}, Cint,), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasZgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, TauArray::Array{Ptr{cuDoubleComplex}, 1}, info::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Ptr{cuDoubleComplex}}, Ref{Cint}, Cint,), handle, m, n, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, TauArray), Base.cconvert(Ref{Cint}, info), batchSize)
end

function cublasZgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, TauArray::Ptr{Ptr{cuDoubleComplex}}, info::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Ptr{Cint}, Cint,), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end

function cublasSgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Array{Ptr{Cfloat}, 1}, lda::Cint, Carray::Array{Ptr{Cfloat}, 1}, ldc::Cint, info::Array{Cint, 1}, devInfoArray::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Ptr{Cfloat}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, trans, m, n, nrhs, Base.cconvert(Ref{Ptr{Cfloat}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cfloat}}, Carray), ldc, Base.cconvert(Ref{Cint}, info), Base.cconvert(Ref{Cint}, devInfoArray), batchSize)
end

function cublasSgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, Carray::Ptr{Ptr{Cfloat}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasSgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasDgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Array{Ptr{Cdouble}, 1}, lda::Cint, Carray::Array{Ptr{Cdouble}, 1}, ldc::Cint, info::Array{Cint, 1}, devInfoArray::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Ptr{Cdouble}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, trans, m, n, nrhs, Base.cconvert(Ref{Ptr{Cdouble}}, Aarray), lda, Base.cconvert(Ref{Ptr{Cdouble}}, Carray), ldc, Base.cconvert(Ref{Cint}, info), Base.cconvert(Ref{Cint}, devInfoArray), batchSize)
end

function cublasDgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, Carray::Ptr{Ptr{Cdouble}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasDgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasCgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Array{Ptr{cuComplex}, 1}, lda::Cint, Carray::Array{Ptr{cuComplex}, 1}, ldc::Cint, info::Array{Cint, 1}, devInfoArray::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Ptr{cuComplex}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, trans, m, n, nrhs, Base.cconvert(Ref{Ptr{cuComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuComplex}}, Carray), ldc, Base.cconvert(Ref{Cint}, info), Base.cconvert(Ref{Cint}, devInfoArray), batchSize)
end

function cublasCgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, Carray::Ptr{Ptr{cuComplex}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasCgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasZgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Array{Ptr{cuDoubleComplex}, 1}, lda::Cint, Carray::Array{Ptr{cuDoubleComplex}, 1}, ldc::Cint, info::Array{Cint, 1}, devInfoArray::Array{Cint, 1}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Ptr{cuDoubleComplex}}, Cint, Ref{Cint}, Ref{Cint}, Cint,), handle, trans, m, n, nrhs, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Aarray), lda, Base.cconvert(Ref{Ptr{cuDoubleComplex}}, Carray), ldc, Base.cconvert(Ref{Cint}, info), Base.cconvert(Ref{Cint}, devInfoArray), batchSize)
end

function cublasZgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Carray::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)::cublasStatus_t
    return ccall((:cublasZgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint,), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end

function cublasSdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, C::Array{Cfloat, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), handle, mode, m, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, C::Ptr{Cfloat}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasSdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasDdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, C::Array{Cdouble, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), handle, mode, m, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, C::Ptr{Cdouble}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasDdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasCdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, C::Array{cuComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), handle, mode, m, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, C::Ptr{cuComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasCdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasZdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, C::Array{cuDoubleComplex, 1}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), handle, mode, m, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)::cublasStatus_t
    return ccall((:cublasZdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), handle, mode, m, n, A, lda, x, incx, C, ldc)
end

function cublasStpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasStpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasStpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint)::cublasStatus_t
    return ccall((:cublasStpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), handle, uplo, n, AP, A, lda)
end

function cublasDtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), handle, uplo, n, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint)::cublasStatus_t
    return ccall((:cublasDtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), handle, uplo, n, AP, A, lda)
end

function cublasCtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasCtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), handle, uplo, n, AP, A, lda)
end

function cublasZtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint)::cublasStatus_t
    return ccall((:cublasZtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), handle, uplo, n, AP, A, lda)
end

function cublasStrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Array{Cfloat, 1}, lda::Cint, AP::Array{Cfloat, 1})::cublasStatus_t
    return ccall((:cublasStrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), handle, uplo, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, AP))
end

function cublasStrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, AP::Ptr{Cfloat})::cublasStatus_t
    return ccall((:cublasStrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), handle, uplo, n, A, lda, AP)
end

function cublasDtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Array{Cdouble, 1}, lda::Cint, AP::Array{Cdouble, 1})::cublasStatus_t
    return ccall((:cublasDtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), handle, uplo, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, AP))
end

function cublasDtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, AP::Ptr{Cdouble})::cublasStatus_t
    return ccall((:cublasDtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), handle, uplo, n, A, lda, AP)
end

function cublasCtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Array{cuComplex, 1}, lda::Cint, AP::Array{cuComplex, 1})::cublasStatus_t
    return ccall((:cublasCtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuComplex}, Cint, Ref{cuComplex},), handle, uplo, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, AP))
end

function cublasCtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, AP::Ptr{cuComplex})::cublasStatus_t
    return ccall((:cublasCtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex},), handle, uplo, n, A, lda, AP)
end

function cublasZtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, AP::Array{cuDoubleComplex, 1})::cublasStatus_t
    return ccall((:cublasZtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), handle, uplo, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, AP))
end

function cublasZtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, AP::Ptr{cuDoubleComplex})::cublasStatus_t
    return ccall((:cublasZtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), handle, uplo, n, A, lda, AP)
end

# CUBLAS functions from 'cublas.h'
function cublasInit()::cublasStatus_t
    return ccall((:cublasInit, libcublas), cublasStatus_t, ())
end

function cublasShutdown()::cublasStatus_t
    return ccall((:cublasShutdown, libcublas), cublasStatus_t, ())
end

function cublasGetError()::cublasStatus_t
    return ccall((:cublasGetError, libcublas), cublasStatus_t, ())
end

function cublasGetVersion(version::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasGetVersion, libcublas), cublasStatus_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, version))
end

function cublasGetVersion(version::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasGetVersion, libcublas), cublasStatus_t, (Ptr{Cint},), version)
end

function cublasAlloc(n::Cint, elemSize::Cint, devicePtr::Array{Ptr{Cvoid}, 1})::cublasStatus_t
    return ccall((:cublasAlloc, libcublas), cublasStatus_t, (Cint, Cint, Ref{Ptr{Cvoid}},), n, elemSize, Base.cconvert(Ref{Ptr{Cvoid}}, devicePtr))
end

function cublasAlloc(n::Cint, elemSize::Cint, devicePtr::Ptr{Ptr{Cvoid}})::cublasStatus_t
    return ccall((:cublasAlloc, libcublas), cublasStatus_t, (Cint, Cint, Ptr{Ptr{Cvoid}},), n, elemSize, devicePtr)
end

function cublasFree(devicePtr::Ptr{Cvoid})::cublasStatus_t
    return ccall((:cublasFree, libcublas), cublasStatus_t, (Ptr{Cvoid},), devicePtr)
end

function cublasSetKernelStream(stream::cudaStream_t)::cublasStatus_t
    return ccall((:cublasSetKernelStream, libcublas), cublasStatus_t, (cudaStream_t,), stream)
end

function cublasSnrm2(n::Cint, x::Array{Cfloat, 1}, incx::Cint)::Cfloat
    return ccall((:cublasSnrm2, libcublas), Cfloat, (Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasSnrm2(n::Cint, x::Ptr{Cfloat}, incx::Cint)::Cfloat
    return ccall((:cublasSnrm2, libcublas), Cfloat, (Cint, Ptr{Cfloat}, Cint,), n, x, incx)
end

function cublasDnrm2(n::Cint, x::Array{Cdouble, 1}, incx::Cint)::Cdouble
    return ccall((:cublasDnrm2, libcublas), Cdouble, (Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDnrm2(n::Cint, x::Ptr{Cdouble}, incx::Cint)::Cdouble
    return ccall((:cublasDnrm2, libcublas), Cdouble, (Cint, Ptr{Cdouble}, Cint,), n, x, incx)
end

function cublasScnrm2(n::Cint, x::Array{cuComplex, 1}, incx::Cint)::Cfloat
    return ccall((:cublasScnrm2, libcublas), Cfloat, (Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasScnrm2(n::Cint, x::Ptr{cuComplex}, incx::Cint)::Cfloat
    return ccall((:cublasScnrm2, libcublas), Cfloat, (Cint, Ptr{cuComplex}, Cint,), n, x, incx)
end

function cublasDznrm2(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Cdouble
    return ccall((:cublasDznrm2, libcublas), Cdouble, (Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasDznrm2(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Cdouble
    return ccall((:cublasDznrm2, libcublas), Cdouble, (Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx)
end

function cublasSdot(n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::Cfloat
    return ccall((:cublasSdot, libcublas), Cfloat, (Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSdot(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::Cfloat
    return ccall((:cublasSdot, libcublas), Cfloat, (Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), n, x, incx, y, incy)
end

function cublasDdot(n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::Cdouble
    return ccall((:cublasDdot, libcublas), Cdouble, (Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDdot(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::Cdouble
    return ccall((:cublasDdot, libcublas), Cdouble, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), n, x, incx, y, incy)
end

function cublasCdotu(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::cuComplex
    return ccall((:cublasCdotu, libcublas), cuComplex, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCdotu(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::cuComplex
    return ccall((:cublasCdotu, libcublas), cuComplex, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), n, x, incx, y, incy)
end

function cublasCdotc(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::cuComplex
    return ccall((:cublasCdotc, libcublas), cuComplex, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCdotc(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::cuComplex
    return ccall((:cublasCdotc, libcublas), cuComplex, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), n, x, incx, y, incy)
end

function cublasZdotu(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::cuDoubleComplex
    return ccall((:cublasZdotu, libcublas), cuDoubleComplex, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZdotu(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::cuDoubleComplex
    return ccall((:cublasZdotu, libcublas), cuDoubleComplex, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx, y, incy)
end

function cublasZdotc(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::cuDoubleComplex
    return ccall((:cublasZdotc, libcublas), cuDoubleComplex, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZdotc(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::cuDoubleComplex
    return ccall((:cublasZdotc, libcublas), cuDoubleComplex, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx, y, incy)
end

function cublasSscal(n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasSscal, libcublas), Nothing, (Cint, Cfloat, Ref{Cfloat}, Cint,), n, alpha, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasSscal(n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasSscal, libcublas), Nothing, (Cint, Cfloat, Ptr{Cfloat}, Cint,), n, alpha, x, incx)
end

function cublasDscal(n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDscal, libcublas), Nothing, (Cint, Cdouble, Ref{Cdouble}, Cint,), n, alpha, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDscal(n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDscal, libcublas), Nothing, (Cint, Cdouble, Ptr{Cdouble}, Cint,), n, alpha, x, incx)
end

function cublasCscal(n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCscal, libcublas), Nothing, (Cint, cuComplex, Ref{cuComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCscal(n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCscal, libcublas), Nothing, (Cint, cuComplex, Ptr{cuComplex}, Cint,), n, alpha, x, incx)
end

function cublasZscal(n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZscal, libcublas), Nothing, (Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZscal(n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZscal, libcublas), Nothing, (Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), n, alpha, x, incx)
end

function cublasCsscal(n::Cint, alpha::Cfloat, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCsscal, libcublas), Nothing, (Cint, Cfloat, Ref{cuComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCsscal(n::Cint, alpha::Cfloat, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCsscal, libcublas), Nothing, (Cint, Cfloat, Ptr{cuComplex}, Cint,), n, alpha, x, incx)
end

function cublasZdscal(n::Cint, alpha::Cdouble, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZdscal, libcublas), Nothing, (Cint, Cdouble, Ref{cuDoubleComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZdscal(n::Cint, alpha::Cdouble, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZdscal, libcublas), Nothing, (Cint, Cdouble, Ptr{cuDoubleComplex}, Cint,), n, alpha, x, incx)
end

function cublasSaxpy(n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSaxpy, libcublas), Nothing, (Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSaxpy(n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSaxpy, libcublas), Nothing, (Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), n, alpha, x, incx, y, incy)
end

function cublasDaxpy(n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDaxpy, libcublas), Nothing, (Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDaxpy(n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDaxpy, libcublas), Nothing, (Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), n, alpha, x, incx, y, incy)
end

function cublasCaxpy(n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasCaxpy, libcublas), Nothing, (Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCaxpy(n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasCaxpy, libcublas), Nothing, (Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), n, alpha, x, incx, y, incy)
end

function cublasZaxpy(n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZaxpy, libcublas), Nothing, (Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZaxpy(n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZaxpy, libcublas), Nothing, (Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), n, alpha, x, incx, y, incy)
end

function cublasScopy(n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasScopy, libcublas), Nothing, (Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasScopy(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasScopy, libcublas), Nothing, (Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), n, x, incx, y, incy)
end

function cublasDcopy(n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDcopy, libcublas), Nothing, (Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDcopy(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDcopy, libcublas), Nothing, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), n, x, incx, y, incy)
end

function cublasCcopy(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasCcopy, libcublas), Nothing, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCcopy(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasCcopy, libcublas), Nothing, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), n, x, incx, y, incy)
end

function cublasZcopy(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZcopy, libcublas), Nothing, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZcopy(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZcopy, libcublas), Nothing, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx, y, incy)
end

function cublasSswap(n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSswap, libcublas), Nothing, (Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSswap(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSswap, libcublas), Nothing, (Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), n, x, incx, y, incy)
end

function cublasDswap(n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDswap, libcublas), Nothing, (Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDswap(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDswap, libcublas), Nothing, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), n, x, incx, y, incy)
end

function cublasCswap(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasCswap, libcublas), Nothing, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCswap(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasCswap, libcublas), Nothing, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), n, x, incx, y, incy)
end

function cublasZswap(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZswap, libcublas), Nothing, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZswap(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZswap, libcublas), Nothing, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx, y, incy)
end

function cublasIsamax(n::Cint, x::Array{Cfloat, 1}, incx::Cint)::Cint
    return ccall((:cublasIsamax, libcublas), Cint, (Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasIsamax(n::Cint, x::Ptr{Cfloat}, incx::Cint)::Cint
    return ccall((:cublasIsamax, libcublas), Cint, (Cint, Ptr{Cfloat}, Cint,), n, x, incx)
end

function cublasIdamax(n::Cint, x::Array{Cdouble, 1}, incx::Cint)::Cint
    return ccall((:cublasIdamax, libcublas), Cint, (Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasIdamax(n::Cint, x::Ptr{Cdouble}, incx::Cint)::Cint
    return ccall((:cublasIdamax, libcublas), Cint, (Cint, Ptr{Cdouble}, Cint,), n, x, incx)
end

function cublasIcamax(n::Cint, x::Array{cuComplex, 1}, incx::Cint)::Cint
    return ccall((:cublasIcamax, libcublas), Cint, (Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasIcamax(n::Cint, x::Ptr{cuComplex}, incx::Cint)::Cint
    return ccall((:cublasIcamax, libcublas), Cint, (Cint, Ptr{cuComplex}, Cint,), n, x, incx)
end

function cublasIzamax(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Cint
    return ccall((:cublasIzamax, libcublas), Cint, (Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasIzamax(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Cint
    return ccall((:cublasIzamax, libcublas), Cint, (Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx)
end

function cublasIsamin(n::Cint, x::Array{Cfloat, 1}, incx::Cint)::Cint
    return ccall((:cublasIsamin, libcublas), Cint, (Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasIsamin(n::Cint, x::Ptr{Cfloat}, incx::Cint)::Cint
    return ccall((:cublasIsamin, libcublas), Cint, (Cint, Ptr{Cfloat}, Cint,), n, x, incx)
end

function cublasIdamin(n::Cint, x::Array{Cdouble, 1}, incx::Cint)::Cint
    return ccall((:cublasIdamin, libcublas), Cint, (Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasIdamin(n::Cint, x::Ptr{Cdouble}, incx::Cint)::Cint
    return ccall((:cublasIdamin, libcublas), Cint, (Cint, Ptr{Cdouble}, Cint,), n, x, incx)
end

function cublasIcamin(n::Cint, x::Array{cuComplex, 1}, incx::Cint)::Cint
    return ccall((:cublasIcamin, libcublas), Cint, (Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasIcamin(n::Cint, x::Ptr{cuComplex}, incx::Cint)::Cint
    return ccall((:cublasIcamin, libcublas), Cint, (Cint, Ptr{cuComplex}, Cint,), n, x, incx)
end

function cublasIzamin(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Cint
    return ccall((:cublasIzamin, libcublas), Cint, (Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasIzamin(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Cint
    return ccall((:cublasIzamin, libcublas), Cint, (Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx)
end

function cublasSasum(n::Cint, x::Array{Cfloat, 1}, incx::Cint)::Cfloat
    return ccall((:cublasSasum, libcublas), Cfloat, (Cint, Ref{Cfloat}, Cint,), n, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasSasum(n::Cint, x::Ptr{Cfloat}, incx::Cint)::Cfloat
    return ccall((:cublasSasum, libcublas), Cfloat, (Cint, Ptr{Cfloat}, Cint,), n, x, incx)
end

function cublasDasum(n::Cint, x::Array{Cdouble, 1}, incx::Cint)::Cdouble
    return ccall((:cublasDasum, libcublas), Cdouble, (Cint, Ref{Cdouble}, Cint,), n, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDasum(n::Cint, x::Ptr{Cdouble}, incx::Cint)::Cdouble
    return ccall((:cublasDasum, libcublas), Cdouble, (Cint, Ptr{Cdouble}, Cint,), n, x, incx)
end

function cublasScasum(n::Cint, x::Array{cuComplex, 1}, incx::Cint)::Cfloat
    return ccall((:cublasScasum, libcublas), Cfloat, (Cint, Ref{cuComplex}, Cint,), n, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasScasum(n::Cint, x::Ptr{cuComplex}, incx::Cint)::Cfloat
    return ccall((:cublasScasum, libcublas), Cfloat, (Cint, Ptr{cuComplex}, Cint,), n, x, incx)
end

function cublasDzasum(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Cdouble
    return ccall((:cublasDzasum, libcublas), Cdouble, (Cint, Ref{cuDoubleComplex}, Cint,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasDzasum(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Cdouble
    return ccall((:cublasDzasum, libcublas), Cdouble, (Cint, Ptr{cuDoubleComplex}, Cint,), n, x, incx)
end

function cublasSrot(n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, sc::Cfloat, ss::Cfloat)::Nothing
    return ccall((:cublasSrot, libcublas), Nothing, (Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Cfloat,), n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, sc, ss)
end

function cublasSrot(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, sc::Cfloat, ss::Cfloat)::Nothing
    return ccall((:cublasSrot, libcublas), Nothing, (Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Cfloat,), n, x, incx, y, incy, sc, ss)
end

function cublasDrot(n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, sc::Cdouble, ss::Cdouble)::Nothing
    return ccall((:cublasDrot, libcublas), Nothing, (Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Cdouble,), n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, sc, ss)
end

function cublasDrot(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, sc::Cdouble, ss::Cdouble)::Nothing
    return ccall((:cublasDrot, libcublas), Nothing, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Cdouble,), n, x, incx, y, incy, sc, ss)
end

function cublasCrot(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, c::Cfloat, s::cuComplex)::Nothing
    return ccall((:cublasCrot, libcublas), Nothing, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Cfloat, cuComplex,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, c, s)
end

function cublasCrot(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Cfloat, s::cuComplex)::Nothing
    return ccall((:cublasCrot, libcublas), Nothing, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Cfloat, cuComplex,), n, x, incx, y, incy, c, s)
end

function cublasZrot(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, sc::Cdouble, cs::cuDoubleComplex)::Nothing
    return ccall((:cublasZrot, libcublas), Nothing, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Cdouble, cuDoubleComplex,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, sc, cs)
end

function cublasZrot(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, sc::Cdouble, cs::cuDoubleComplex)::Nothing
    return ccall((:cublasZrot, libcublas), Nothing, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Cdouble, cuDoubleComplex,), n, x, incx, y, incy, sc, cs)
end

function cublasCsrot(n::Cint, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, c::Cfloat, s::Cfloat)::Nothing
    return ccall((:cublasCsrot, libcublas), Nothing, (Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Cfloat, Cfloat,), n, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, c, s)
end

function cublasCsrot(n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Cfloat, s::Cfloat)::Nothing
    return ccall((:cublasCsrot, libcublas), Nothing, (Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Cfloat, Cfloat,), n, x, incx, y, incy, c, s)
end

function cublasZdrot(n::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, c::Cdouble, s::Cdouble)::Nothing
    return ccall((:cublasZdrot, libcublas), Nothing, (Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Cdouble, Cdouble,), n, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, c, s)
end

function cublasZdrot(n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, c::Cdouble, s::Cdouble)::Nothing
    return ccall((:cublasZdrot, libcublas), Nothing, (Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Cdouble, Cdouble,), n, x, incx, y, incy, c, s)
end

function cublasSrotg(sa::Array{Cfloat, 1}, sb::Array{Cfloat, 1}, sc::Array{Cfloat, 1}, ss::Array{Cfloat, 1})::Nothing
    return ccall((:cublasSrotg, libcublas), Nothing, (Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat},), Base.cconvert(Ref{Cfloat}, sa), Base.cconvert(Ref{Cfloat}, sb), Base.cconvert(Ref{Cfloat}, sc), Base.cconvert(Ref{Cfloat}, ss))
end

function cublasSrotg(sa::Ptr{Cfloat}, sb::Ptr{Cfloat}, sc::Ptr{Cfloat}, ss::Ptr{Cfloat})::Nothing
    return ccall((:cublasSrotg, libcublas), Nothing, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},), sa, sb, sc, ss)
end

function cublasDrotg(sa::Array{Cdouble, 1}, sb::Array{Cdouble, 1}, sc::Array{Cdouble, 1}, ss::Array{Cdouble, 1})::Nothing
    return ccall((:cublasDrotg, libcublas), Nothing, (Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble},), Base.cconvert(Ref{Cdouble}, sa), Base.cconvert(Ref{Cdouble}, sb), Base.cconvert(Ref{Cdouble}, sc), Base.cconvert(Ref{Cdouble}, ss))
end

function cublasDrotg(sa::Ptr{Cdouble}, sb::Ptr{Cdouble}, sc::Ptr{Cdouble}, ss::Ptr{Cdouble})::Nothing
    return ccall((:cublasDrotg, libcublas), Nothing, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},), sa, sb, sc, ss)
end

function cublasCrotg(ca::Array{cuComplex, 1}, cb::cuComplex, sc::Array{Cfloat, 1}, cs::Array{cuComplex, 1})::Nothing
    return ccall((:cublasCrotg, libcublas), Nothing, (Ref{cuComplex}, cuComplex, Ref{Cfloat}, Ref{cuComplex},), Base.cconvert(Ref{cuComplex}, ca), cb, Base.cconvert(Ref{Cfloat}, sc), Base.cconvert(Ref{cuComplex}, cs))
end

function cublasCrotg(ca::Ptr{cuComplex}, cb::cuComplex, sc::Ptr{Cfloat}, cs::Ptr{cuComplex})::Nothing
    return ccall((:cublasCrotg, libcublas), Nothing, (Ptr{cuComplex}, cuComplex, Ptr{Cfloat}, Ptr{cuComplex},), ca, cb, sc, cs)
end

function cublasZrotg(ca::Array{cuDoubleComplex, 1}, cb::cuDoubleComplex, sc::Array{Cdouble, 1}, cs::Array{cuDoubleComplex, 1})::Nothing
    return ccall((:cublasZrotg, libcublas), Nothing, (Ref{cuDoubleComplex}, cuDoubleComplex, Ref{Cdouble}, Ref{cuDoubleComplex},), Base.cconvert(Ref{cuDoubleComplex}, ca), cb, Base.cconvert(Ref{Cdouble}, sc), Base.cconvert(Ref{cuDoubleComplex}, cs))
end

function cublasZrotg(ca::Ptr{cuDoubleComplex}, cb::cuDoubleComplex, sc::Ptr{Cdouble}, cs::Ptr{cuDoubleComplex})::Nothing
    return ccall((:cublasZrotg, libcublas), Nothing, (Ptr{cuDoubleComplex}, cuDoubleComplex, Ptr{Cdouble}, Ptr{cuDoubleComplex},), ca, cb, sc, cs)
end

function cublasSrotm(n::Cint, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, sparam::Array{Cfloat, 1})::Nothing
    return ccall((:cublasSrotm, libcublas), Nothing, (Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), n, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, sparam))
end

function cublasSrotm(n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, sparam::Ptr{Cfloat})::Nothing
    return ccall((:cublasSrotm, libcublas), Nothing, (Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), n, x, incx, y, incy, sparam)
end

function cublasDrotm(n::Cint, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, sparam::Array{Cdouble, 1})::Nothing
    return ccall((:cublasDrotm, libcublas), Nothing, (Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), n, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, sparam))
end

function cublasDrotm(n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, sparam::Ptr{Cdouble})::Nothing
    return ccall((:cublasDrotm, libcublas), Nothing, (Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), n, x, incx, y, incy, sparam)
end

function cublasSrotmg(sd1::Array{Cfloat, 1}, sd2::Array{Cfloat, 1}, sx1::Array{Cfloat, 1}, sy1::Array{Cfloat, 1}, sparam::Array{Cfloat, 1})::Nothing
    return ccall((:cublasSrotmg, libcublas), Nothing, (Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat},), Base.cconvert(Ref{Cfloat}, sd1), Base.cconvert(Ref{Cfloat}, sd2), Base.cconvert(Ref{Cfloat}, sx1), Base.cconvert(Ref{Cfloat}, sy1), Base.cconvert(Ref{Cfloat}, sparam))
end

function cublasSrotmg(sd1::Ptr{Cfloat}, sd2::Ptr{Cfloat}, sx1::Ptr{Cfloat}, sy1::Ptr{Cfloat}, sparam::Ptr{Cfloat})::Nothing
    return ccall((:cublasSrotmg, libcublas), Nothing, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},), sd1, sd2, sx1, sy1, sparam)
end

function cublasDrotmg(sd1::Array{Cdouble, 1}, sd2::Array{Cdouble, 1}, sx1::Array{Cdouble, 1}, sy1::Array{Cdouble, 1}, sparam::Array{Cdouble, 1})::Nothing
    return ccall((:cublasDrotmg, libcublas), Nothing, (Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble},), Base.cconvert(Ref{Cdouble}, sd1), Base.cconvert(Ref{Cdouble}, sd2), Base.cconvert(Ref{Cdouble}, sx1), Base.cconvert(Ref{Cdouble}, sy1), Base.cconvert(Ref{Cdouble}, sparam))
end

function cublasDrotmg(sd1::Ptr{Cdouble}, sd2::Ptr{Cdouble}, sx1::Ptr{Cdouble}, sy1::Ptr{Cdouble}, sparam::Ptr{Cdouble})::Nothing
    return ccall((:cublasDrotmg, libcublas), Nothing, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},), sd1, sd2, sx1, sy1, sparam)
end

function cublasSgemv(trans::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Cfloat, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSgemv, libcublas), Nothing, (UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), trans, m, n, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, beta, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSgemv(trans::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSgemv, libcublas), Nothing, (UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgemv(trans::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Cdouble, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDgemv, libcublas), Nothing, (UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), trans, m, n, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, beta, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDgemv(trans::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDgemv, libcublas), Nothing, (UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgemv(trans::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::cuComplex, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasCgemv, libcublas), Nothing, (UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), trans, m, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, beta, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCgemv(trans::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasCgemv, libcublas), Nothing, (UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgemv(trans::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::cuDoubleComplex, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZgemv, libcublas), Nothing, (UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), trans, m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, beta, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZgemv(trans::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZgemv, libcublas), Nothing, (UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Cfloat, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), trans, m, n, kl, ku, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, beta, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Cdouble, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), trans, m, n, kl, ku, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, beta, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasCgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::cuComplex, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasCgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), trans, m, n, kl, ku, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, beta, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasCgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasCgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::cuDoubleComplex, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), trans, m, n, kl, ku, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, beta, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZgbmv(trans::UInt8, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZgbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasStrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtrmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtrmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtbmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtbmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasStpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtpmv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtpmv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasStrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasDtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasCtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasZtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtrsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtrsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, A, lda, x, incx)
end

function cublasStpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cfloat}, Ref{Cfloat}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasDtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{Cdouble}, Ref{Cdouble}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasCtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuComplex}, Ref{cuComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasZtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtpsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtpsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, AP, x, incx)
end

function cublasStbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint)::Nothing
    return ccall((:cublasStbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx)
end

function cublasStbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)::Nothing
    return ccall((:cublasStbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasDtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint)::Nothing
    return ccall((:cublasDtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx)
end

function cublasDtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)::Nothing
    return ccall((:cublasDtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasCtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasCtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx)
end

function cublasCtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)::Nothing
    return ccall((:cublasCtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasZtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint)::Nothing
    return ccall((:cublasZtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, trans, diag, n, k, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx)
end

function cublasZtbsv(uplo::UInt8, trans::UInt8, diag::UInt8, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)::Nothing
    return ccall((:cublasZtbsv, libcublas), Nothing, (UInt8, UInt8, UInt8, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, trans, diag, n, k, A, lda, x, incx)
end

function cublasSsymv(uplo::UInt8, n::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Cfloat, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSsymv, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, beta, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSsymv(uplo::UInt8, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSsymv, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsymv(uplo::UInt8, n::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Cdouble, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDsymv, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, beta, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDsymv(uplo::UInt8, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDsymv, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChemv(uplo::UInt8, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::cuComplex, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasChemv, libcublas), Nothing, (UInt8, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, beta, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChemv(uplo::UInt8, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasChemv, libcublas), Nothing, (UInt8, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhemv(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::cuDoubleComplex, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZhemv, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, beta, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhemv(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZhemv, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSsbmv(uplo::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, x::Array{Cfloat, 1}, incx::Cint, beta::Cfloat, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSsbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), uplo, n, k, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, x), incx, beta, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSsbmv(uplo::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSsbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasDsbmv(uplo::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, x::Array{Cdouble, 1}, incx::Cint, beta::Cdouble, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDsbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), uplo, n, k, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, x), incx, beta, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDsbmv(uplo::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDsbmv, libcublas), Nothing, (UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasChbmv(uplo::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, x::Array{cuComplex, 1}, incx::Cint, beta::cuComplex, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasChbmv, libcublas), Nothing, (UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), uplo, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, x), incx, beta, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChbmv(uplo::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasChbmv, libcublas), Nothing, (UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasZhbmv(uplo::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::cuDoubleComplex, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZhbmv, libcublas), Nothing, (UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), uplo, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, x), incx, beta, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhbmv(uplo::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZhbmv, libcublas), Nothing, (UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end

function cublasSspmv(uplo::UInt8, n::Cint, alpha::Cfloat, AP::Array{Cfloat, 1}, x::Array{Cfloat, 1}, incx::Cint, beta::Cfloat, y::Array{Cfloat, 1}, incy::Cint)::Nothing
    return ccall((:cublasSspmv, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, x), incx, beta, Base.cconvert(Ref{Cfloat}, y), incy)
end

function cublasSspmv(uplo::UInt8, n::Cint, alpha::Cfloat, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, beta::Cfloat, y::Ptr{Cfloat}, incy::Cint)::Nothing
    return ccall((:cublasSspmv, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasDspmv(uplo::UInt8, n::Cint, alpha::Cdouble, AP::Array{Cdouble, 1}, x::Array{Cdouble, 1}, incx::Cint, beta::Cdouble, y::Array{Cdouble, 1}, incy::Cint)::Nothing
    return ccall((:cublasDspmv, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, x), incx, beta, Base.cconvert(Ref{Cdouble}, y), incy)
end

function cublasDspmv(uplo::UInt8, n::Cint, alpha::Cdouble, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, beta::Cdouble, y::Ptr{Cdouble}, incy::Cint)::Nothing
    return ccall((:cublasDspmv, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasChpmv(uplo::UInt8, n::Cint, alpha::cuComplex, AP::Array{cuComplex, 1}, x::Array{cuComplex, 1}, incx::Cint, beta::cuComplex, y::Array{cuComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasChpmv, libcublas), Nothing, (UInt8, Cint, cuComplex, Ref{cuComplex}, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, x), incx, beta, Base.cconvert(Ref{cuComplex}, y), incy)
end

function cublasChpmv(uplo::UInt8, n::Cint, alpha::cuComplex, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, beta::cuComplex, y::Ptr{cuComplex}, incy::Cint)::Nothing
    return ccall((:cublasChpmv, libcublas), Nothing, (UInt8, Cint, cuComplex, Ptr{cuComplex}, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasZhpmv(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, AP::Array{cuDoubleComplex, 1}, x::Array{cuDoubleComplex, 1}, incx::Cint, beta::cuDoubleComplex, y::Array{cuDoubleComplex, 1}, incy::Cint)::Nothing
    return ccall((:cublasZhpmv, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, x), incx, beta, Base.cconvert(Ref{cuDoubleComplex}, y), incy)
end

function cublasZhpmv(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, beta::cuDoubleComplex, y::Ptr{cuDoubleComplex}, incy::Cint)::Nothing
    return ccall((:cublasZhpmv, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), uplo, n, alpha, AP, x, incx, beta, y, incy)
end

function cublasSger(m::Cint, n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, A::Array{Cfloat, 1}, lda::Cint)::Nothing
    return ccall((:cublasSger, libcublas), Nothing, (Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), m, n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSger(m::Cint, n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)::Nothing
    return ccall((:cublasSger, libcublas), Nothing, (Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDger(m::Cint, n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, A::Array{Cdouble, 1}, lda::Cint)::Nothing
    return ccall((:cublasDger, libcublas), Nothing, (Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), m, n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDger(m::Cint, n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)::Nothing
    return ccall((:cublasDger, libcublas), Nothing, (Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgeru(m::Cint, n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasCgeru, libcublas), Nothing, (Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), m, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCgeru(m::Cint, n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::Nothing
    return ccall((:cublasCgeru, libcublas), Nothing, (Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCgerc(m::Cint, n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasCgerc, libcublas), Nothing, (Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), m, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCgerc(m::Cint, n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::Nothing
    return ccall((:cublasCgerc, libcublas), Nothing, (Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgeru(m::Cint, n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasZgeru, libcublas), Nothing, (Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZgeru(m::Cint, n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::Nothing
    return ccall((:cublasZgeru, libcublas), Nothing, (Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZgerc(m::Cint, n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasZgerc, libcublas), Nothing, (Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZgerc(m::Cint, n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::Nothing
    return ccall((:cublasZgerc, libcublas), Nothing, (Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), m, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSsyr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, A::Array{Cfloat, 1}, lda::Cint)::Nothing
    return ccall((:cublasSsyr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSsyr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, A::Ptr{Cfloat}, lda::Cint)::Nothing
    return ccall((:cublasSsyr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, n, alpha, x, incx, A, lda)
end

function cublasDsyr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, A::Array{Cdouble, 1}, lda::Cint)::Nothing
    return ccall((:cublasDsyr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDsyr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, A::Ptr{Cdouble}, lda::Cint)::Nothing
    return ccall((:cublasDsyr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, n, alpha, x, incx, A, lda)
end

function cublasCher(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{cuComplex, 1}, incx::Cint, A::Array{cuComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasCher, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCher(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{cuComplex}, incx::Cint, A::Ptr{cuComplex}, lda::Cint)::Nothing
    return ccall((:cublasCher, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, n, alpha, x, incx, A, lda)
end

function cublasZher(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{cuDoubleComplex, 1}, incx::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasZher, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZher(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{cuDoubleComplex}, incx::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::Nothing
    return ccall((:cublasZher, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, n, alpha, x, incx, A, lda)
end

function cublasSspr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, AP::Array{Cfloat, 1})::Nothing
    return ccall((:cublasSspr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat},), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, AP))
end

function cublasSspr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, AP::Ptr{Cfloat})::Nothing
    return ccall((:cublasSspr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat},), uplo, n, alpha, x, incx, AP)
end

function cublasDspr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, AP::Array{Cdouble, 1})::Nothing
    return ccall((:cublasDspr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble},), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, AP))
end

function cublasDspr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, AP::Ptr{Cdouble})::Nothing
    return ccall((:cublasDspr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble},), uplo, n, alpha, x, incx, AP)
end

function cublasChpr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{cuComplex, 1}, incx::Cint, AP::Array{cuComplex, 1})::Nothing
    return ccall((:cublasChpr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{cuComplex}, Cint, Ref{cuComplex},), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, AP))
end

function cublasChpr(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{cuComplex}, incx::Cint, AP::Ptr{cuComplex})::Nothing
    return ccall((:cublasChpr, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{cuComplex}, Cint, Ptr{cuComplex},), uplo, n, alpha, x, incx, AP)
end

function cublasZhpr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{cuDoubleComplex, 1}, incx::Cint, AP::Array{cuDoubleComplex, 1})::Nothing
    return ccall((:cublasZhpr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, AP))
end

function cublasZhpr(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{cuDoubleComplex}, incx::Cint, AP::Ptr{cuDoubleComplex})::Nothing
    return ccall((:cublasZhpr, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), uplo, n, alpha, x, incx, AP)
end

function cublasSsyr2(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, A::Array{Cfloat, 1}, lda::Cint)::Nothing
    return ccall((:cublasSsyr2, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, A), lda)
end

function cublasSsyr2(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)::Nothing
    return ccall((:cublasSsyr2, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasDsyr2(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, A::Array{Cdouble, 1}, lda::Cint)::Nothing
    return ccall((:cublasDsyr2, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, A), lda)
end

function cublasDsyr2(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)::Nothing
    return ccall((:cublasDsyr2, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasCher2(uplo::UInt8, n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, A::Array{cuComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasCher2, libcublas), Nothing, (UInt8, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, A), lda)
end

function cublasCher2(uplo::UInt8, n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)::Nothing
    return ccall((:cublasCher2, libcublas), Nothing, (UInt8, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasZher2(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, A::Array{cuDoubleComplex, 1}, lda::Cint)::Nothing
    return ccall((:cublasZher2, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, A), lda)
end

function cublasZher2(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)::Nothing
    return ccall((:cublasZher2, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), uplo, n, alpha, x, incx, y, incy, A, lda)
end

function cublasSspr2(uplo::UInt8, n::Cint, alpha::Cfloat, x::Array{Cfloat, 1}, incx::Cint, y::Array{Cfloat, 1}, incy::Cint, AP::Array{Cfloat, 1})::Nothing
    return ccall((:cublasSspr2, libcublas), Nothing, (UInt8, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Ref{Cfloat},), uplo, n, alpha, Base.cconvert(Ref{Cfloat}, x), incx, Base.cconvert(Ref{Cfloat}, y), incy, Base.cconvert(Ref{Cfloat}, AP))
end

function cublasSspr2(uplo::UInt8, n::Cint, alpha::Cfloat, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, AP::Ptr{Cfloat})::Nothing
    return ccall((:cublasSspr2, libcublas), Nothing, (UInt8, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat},), uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasDspr2(uplo::UInt8, n::Cint, alpha::Cdouble, x::Array{Cdouble, 1}, incx::Cint, y::Array{Cdouble, 1}, incy::Cint, AP::Array{Cdouble, 1})::Nothing
    return ccall((:cublasDspr2, libcublas), Nothing, (UInt8, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Ref{Cdouble},), uplo, n, alpha, Base.cconvert(Ref{Cdouble}, x), incx, Base.cconvert(Ref{Cdouble}, y), incy, Base.cconvert(Ref{Cdouble}, AP))
end

function cublasDspr2(uplo::UInt8, n::Cint, alpha::Cdouble, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, AP::Ptr{Cdouble})::Nothing
    return ccall((:cublasDspr2, libcublas), Nothing, (UInt8, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble},), uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasChpr2(uplo::UInt8, n::Cint, alpha::cuComplex, x::Array{cuComplex, 1}, incx::Cint, y::Array{cuComplex, 1}, incy::Cint, AP::Array{cuComplex, 1})::Nothing
    return ccall((:cublasChpr2, libcublas), Nothing, (UInt8, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Ref{cuComplex},), uplo, n, alpha, Base.cconvert(Ref{cuComplex}, x), incx, Base.cconvert(Ref{cuComplex}, y), incy, Base.cconvert(Ref{cuComplex}, AP))
end

function cublasChpr2(uplo::UInt8, n::Cint, alpha::cuComplex, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, AP::Ptr{cuComplex})::Nothing
    return ccall((:cublasChpr2, libcublas), Nothing, (UInt8, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex},), uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasZhpr2(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, x::Array{cuDoubleComplex, 1}, incx::Cint, y::Array{cuDoubleComplex, 1}, incy::Cint, AP::Array{cuDoubleComplex, 1})::Nothing
    return ccall((:cublasZhpr2, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex},), uplo, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, x), incx, Base.cconvert(Ref{cuDoubleComplex}, y), incy, Base.cconvert(Ref{cuDoubleComplex}, AP))
end

function cublasZhpr2(uplo::UInt8, n::Cint, alpha::cuDoubleComplex, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, AP::Ptr{cuDoubleComplex})::Nothing
    return ccall((:cublasZhpr2, libcublas), Nothing, (UInt8, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},), uplo, n, alpha, x, incx, y, incy, AP)
end

function cublasSgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Cfloat, C::Array{Cfloat, 1}, ldc::Cint)::Nothing
    return ccall((:cublasSgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), transa, transb, m, n, k, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, beta, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Nothing
    return ccall((:cublasSgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Cdouble, C::Array{Cdouble, 1}, ldc::Cint)::Nothing
    return ccall((:cublasDgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), transa, transb, m, n, k, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, beta, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Nothing
    return ccall((:cublasDgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::cuComplex, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), transa, transb, m, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::cuDoubleComplex, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), transa, transb, m, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZgemm(transa::UInt8, transb::UInt8, m::Cint, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZgemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, beta::Cfloat, C::Array{Cfloat, 1}, ldc::Cint)::Nothing
    return ccall((:cublasSsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{Cfloat}, A), lda, beta, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Nothing
    return ccall((:cublasSsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasDsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, beta::Cdouble, C::Array{Cdouble, 1}, ldc::Cint)::Nothing
    return ccall((:cublasDsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{Cdouble}, A), lda, beta, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Nothing
    return ccall((:cublasDsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, beta::cuComplex, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, beta::cuDoubleComplex, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsyrk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZsyrk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasCherk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Array{cuComplex, 1}, lda::Cint, beta::Cfloat, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCherk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ref{cuComplex}, Cint, Cfloat, Ref{cuComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCherk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{cuComplex}, lda::Cint, beta::Cfloat, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCherk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ptr{cuComplex}, Cint, Cfloat, Ptr{cuComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasZherk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Array{cuDoubleComplex, 1}, lda::Cint, beta::Cdouble, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZherk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ref{cuDoubleComplex}, Cint, Cdouble, Ref{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZherk(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Cdouble, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZherk, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasSsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Cfloat, C::Array{Cfloat, 1}, ldc::Cint)::Nothing
    return ccall((:cublasSsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, beta, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Nothing
    return ccall((:cublasSsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Cdouble, C::Array{Cdouble, 1}, ldc::Cint)::Nothing
    return ccall((:cublasDsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, beta, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Nothing
    return ccall((:cublasDsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::cuComplex, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::cuDoubleComplex, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsyr2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZsyr2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCher2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::Cfloat, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCher2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, Cfloat, Ref{cuComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCher2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Cfloat, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCher2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Cfloat, Ptr{cuComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZher2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::Cdouble, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZher2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, Cdouble, Ref{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZher2k(uplo::UInt8, trans::UInt8, n::Cint, k::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Cdouble, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZher2k, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Cdouble, Ptr{cuDoubleComplex}, Cint,), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasSsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint, beta::Cfloat, C::Array{Cfloat, 1}, ldc::Cint)::Nothing
    return ccall((:cublasSsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint, Cfloat, Ref{Cfloat}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, beta, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasSsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Cfloat, C::Ptr{Cfloat}, ldc::Cint)::Nothing
    return ccall((:cublasSsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Cfloat, Ptr{Cfloat}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasDsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint, beta::Cdouble, C::Array{Cdouble, 1}, ldc::Cint)::Nothing
    return ccall((:cublasDsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint, Cdouble, Ref{Cdouble}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, beta, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasDsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Cdouble, C::Ptr{Cdouble}, ldc::Cint)::Nothing
    return ccall((:cublasDsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Cdouble, Ptr{Cdouble}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasCsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::cuComplex, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasCsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasCsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasCsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::cuDoubleComplex, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZsymm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZsymm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasChemm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint, beta::cuComplex, C::Array{cuComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasChemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint, cuComplex, Ref{cuComplex}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, beta, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasChemm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::cuComplex, C::Ptr{cuComplex}, ldc::Cint)::Nothing
    return ccall((:cublasChemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, cuComplex, Ptr{cuComplex}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasZhemm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint, beta::cuDoubleComplex, C::Array{cuDoubleComplex, 1}, ldc::Cint)::Nothing
    return ccall((:cublasZhemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint,), side, uplo, m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, beta, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasZhemm(side::UInt8, uplo::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::cuDoubleComplex, C::Ptr{cuDoubleComplex}, ldc::Cint)::Nothing
    return ccall((:cublasZhemm, libcublas), Nothing, (UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint,), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasStrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint)::Nothing
    return ccall((:cublasStrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb)
end

function cublasStrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Nothing
    return ccall((:cublasStrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasDtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint)::Nothing
    return ccall((:cublasDtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb)
end

function cublasDtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Nothing
    return ccall((:cublasDtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasCtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint)::Nothing
    return ccall((:cublasCtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb)
end

function cublasCtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint)::Nothing
    return ccall((:cublasCtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasZtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint)::Nothing
    return ccall((:cublasZtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb)
end

function cublasZtrsm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint)::Nothing
    return ccall((:cublasZtrsm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasStrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Array{Cfloat, 1}, lda::Cint, B::Array{Cfloat, 1}, ldb::Cint)::Nothing
    return ccall((:cublasStrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cfloat, Ref{Cfloat}, Cint, Ref{Cfloat}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb)
end

function cublasStrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cfloat, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)::Nothing
    return ccall((:cublasStrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cfloat, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasDtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Array{Cdouble, 1}, lda::Cint, B::Array{Cdouble, 1}, ldb::Cint)::Nothing
    return ccall((:cublasDtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cdouble, Ref{Cdouble}, Cint, Ref{Cdouble}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb)
end

function cublasDtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::Cdouble, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)::Nothing
    return ccall((:cublasDtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, Cdouble, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasCtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Array{cuComplex, 1}, lda::Cint, B::Array{cuComplex, 1}, ldb::Cint)::Nothing
    return ccall((:cublasCtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuComplex, Ref{cuComplex}, Cint, Ref{cuComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb)
end

function cublasCtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuComplex, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint)::Nothing
    return ccall((:cublasCtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuComplex, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasZtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Array{cuDoubleComplex, 1}, lda::Cint, B::Array{cuDoubleComplex, 1}, ldb::Cint)::Nothing
    return ccall((:cublasZtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ref{cuDoubleComplex}, Cint, Ref{cuDoubleComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb)
end

function cublasZtrmm(side::UInt8, uplo::UInt8, transa::UInt8, diag::UInt8, m::Cint, n::Cint, alpha::cuDoubleComplex, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint)::Nothing
    return ccall((:cublasZtrmm, libcublas), Nothing, (UInt8, UInt8, UInt8, UInt8, Cint, Cint, cuDoubleComplex, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint,), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

# CUBLAS XT functions from 'cublasXt.h'
function cublasXtCreate(handle::Array{cublasXtHandle_t, 1})::cublasStatus_t
    return ccall((:cublasXtCreate, libcublas), cublasStatus_t, (Ref{cublasXtHandle_t},), Base.cconvert(Ref{cublasXtHandle_t}, handle))
end

function cublasXtCreate(handle::Ptr{cublasXtHandle_t})::cublasStatus_t
    return ccall((:cublasXtCreate, libcublas), cublasStatus_t, (Ptr{cublasXtHandle_t},), handle)
end

function cublasXtDestroy(handle::cublasXtHandle_t)::cublasStatus_t
    return ccall((:cublasXtDestroy, libcublas), cublasStatus_t, (cublasXtHandle_t,), handle)
end

function cublasXtGetNumBoards(nbDevices::Cint, deviceId::Array{Cint, 1}, nbBoards::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasXtGetNumBoards, libcublas), cublasStatus_t, (Cint, Ref{Cint}, Ref{Cint},), nbDevices, Base.cconvert(Ref{Cint}, deviceId), Base.cconvert(Ref{Cint}, nbBoards))
end

function cublasXtGetNumBoards(nbDevices::Cint, deviceId::Ptr{Cint}, nbBoards::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasXtGetNumBoards, libcublas), cublasStatus_t, (Cint, Ptr{Cint}, Ptr{Cint},), nbDevices, deviceId, nbBoards)
end

function cublasXtMaxBoards(nbGpuBoards::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasXtMaxBoards, libcublas), cublasStatus_t, (Ref{Cint},), Base.cconvert(Ref{Cint}, nbGpuBoards))
end

function cublasXtMaxBoards(nbGpuBoards::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasXtMaxBoards, libcublas), cublasStatus_t, (Ptr{Cint},), nbGpuBoards)
end

function cublasXtDeviceSelect(handle::cublasXtHandle_t, nbDevices::Cint, deviceId::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasXtDeviceSelect, libcublas), cublasStatus_t, (cublasXtHandle_t, Cint, Ref{Cint},), handle, nbDevices, Base.cconvert(Ref{Cint}, deviceId))
end

function cublasXtDeviceSelect(handle::cublasXtHandle_t, nbDevices::Cint, deviceId::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasXtDeviceSelect, libcublas), cublasStatus_t, (cublasXtHandle_t, Cint, Ptr{Cint},), handle, nbDevices, deviceId)
end

function cublasXtSetBlockDim(handle::cublasXtHandle_t, blockDim::Cint)::cublasStatus_t
    return ccall((:cublasXtSetBlockDim, libcublas), cublasStatus_t, (cublasXtHandle_t, Cint,), handle, blockDim)
end

function cublasXtGetBlockDim(handle::cublasXtHandle_t, blockDim::Array{Cint, 1})::cublasStatus_t
    return ccall((:cublasXtGetBlockDim, libcublas), cublasStatus_t, (cublasXtHandle_t, Ref{Cint},), handle, Base.cconvert(Ref{Cint}, blockDim))
end

function cublasXtGetBlockDim(handle::cublasXtHandle_t, blockDim::Ptr{Cint})::cublasStatus_t
    return ccall((:cublasXtGetBlockDim, libcublas), cublasStatus_t, (cublasXtHandle_t, Ptr{Cint},), handle, blockDim)
end

function cublasXtGetPinningMemMode(handle::cublasXtHandle_t, mode::Array{cublasXtPinnedMemMode_t, 1})::cublasStatus_t
    return ccall((:cublasXtGetPinningMemMode, libcublas), cublasStatus_t, (cublasXtHandle_t, Ref{cublasXtPinnedMemMode_t},), handle, Base.cconvert(Ref{cublasXtPinnedMemMode_t}, mode))
end

function cublasXtGetPinningMemMode(handle::cublasXtHandle_t, mode::Ptr{cublasXtPinnedMemMode_t})::cublasStatus_t
    return ccall((:cublasXtGetPinningMemMode, libcublas), cublasStatus_t, (cublasXtHandle_t, Ptr{cublasXtPinnedMemMode_t},), handle, mode)
end

function cublasXtSetPinningMemMode(handle::cublasXtHandle_t, mode::cublasXtPinnedMemMode_t)::cublasStatus_t
    return ccall((:cublasXtSetPinningMemMode, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasXtPinnedMemMode_t,), handle, mode)
end

function cublasXtSetCpuRoutine(handle::cublasXtHandle_t, blasOp::cublasXtBlasOp_t, type::cublasXtOpType_t, blasFunctor::Ptr{Cvoid})::cublasStatus_t
    return ccall((:cublasXtSetCpuRoutine, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, Ptr{Cvoid},), handle, blasOp, type, blasFunctor)
end

function cublasXtSetCpuRatio(handle::cublasXtHandle_t, blasOp::cublasXtBlasOp_t, type::cublasXtOpType_t, ratio::Cfloat)::cublasStatus_t
    return ccall((:cublasXtSetCpuRatio, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, Cfloat,), handle, blasOp, type, ratio)
end

function cublasXtSgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, transa, transb, m, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZgemm(handle::cublasXtHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Csize_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZgemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasOperation_t, cublasOperation_t, Csize_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtDsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtCsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtZsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZsyrk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyrk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtCherk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cfloat, 1}, A::Array{cuComplex, 1}, lda::Csize_t, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCherk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{cuComplex}, Csize_t, Ref{Cfloat}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCherk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{cuComplex}, lda::Csize_t, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCherk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{cuComplex}, Csize_t, Ptr{Cfloat}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtZherk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cdouble, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZherk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{cuDoubleComplex}, Csize_t, Ref{Cdouble}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZherk(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{cuDoubleComplex}, lda::Csize_t, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZherk, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Csize_t, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end

function cublasXtSsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZsyr2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyr2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCherkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCherkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{Cfloat}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCherkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCherkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{Cfloat}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZherkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZherkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{Cdouble}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZherkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZherkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtStrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtStrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb)
end

function cublasXtStrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtStrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtDtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb)
end

function cublasXtDtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtCtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb)
end

function cublasXtCtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtZtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb)
end

function cublasXtZtrsm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZtrsm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end

function cublasXtSsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZsymm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsymm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtChemm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtChemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtChemm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtChemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZhemm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZhemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZhemm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZhemm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtDsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZsyrkx(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZsyrkx, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtCher2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCher2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{Cfloat}, Ref{cuComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCher2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCher2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{Cfloat}, Ptr{cuComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtZher2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZher2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{Cdouble}, Ref{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZher2k(handle::cublasXtHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Csize_t, k::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZher2k, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Csize_t,), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end

function cublasXtSspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{Cfloat, 1}, AP::Array{Cfloat, 1}, B::Array{Cfloat, 1}, ldb::Csize_t, beta::Array{Cfloat, 1}, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, AP), Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, beta), Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtSspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cfloat}, AP::Ptr{Cfloat}, B::Ptr{Cfloat}, ldb::Csize_t, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtSspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t,), handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtDspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{Cdouble, 1}, AP::Array{Cdouble, 1}, B::Array{Cdouble, 1}, ldb::Csize_t, beta::Array{Cdouble, 1}, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, AP), Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, beta), Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cdouble}, AP::Ptr{Cdouble}, B::Ptr{Cdouble}, ldb::Csize_t, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t,), handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtCspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuComplex, 1}, AP::Array{cuComplex, 1}, B::Array{cuComplex, 1}, ldb::Csize_t, beta::Array{cuComplex, 1}, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, AP), Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, beta), Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuComplex}, AP::Ptr{cuComplex}, B::Ptr{cuComplex}, ldb::Csize_t, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t,), handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtZspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Array{cuDoubleComplex, 1}, AP::Array{cuDoubleComplex, 1}, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, beta::Array{cuDoubleComplex, 1}, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, AP), Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, beta), Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZspmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuDoubleComplex}, AP::Ptr{cuDoubleComplex}, B::Ptr{cuDoubleComplex}, ldb::Csize_t, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZspmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t,), handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc)
end

function cublasXtStrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{Cfloat, 1}, A::Array{Cfloat, 1}, lda::Csize_t, B::Array{Cfloat, 1}, ldb::Csize_t, C::Array{Cfloat, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtStrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{Cfloat}, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t, Ref{Cfloat}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cfloat}, alpha), Base.cconvert(Ref{Cfloat}, A), lda, Base.cconvert(Ref{Cfloat}, B), ldb, Base.cconvert(Ref{Cfloat}, C), ldc)
end

function cublasXtStrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Csize_t, B::Ptr{Cfloat}, ldb::Csize_t, C::Ptr{Cfloat}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtStrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t, Ptr{Cfloat}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtDtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{Cdouble, 1}, A::Array{Cdouble, 1}, lda::Csize_t, B::Array{Cdouble, 1}, ldb::Csize_t, C::Array{Cdouble, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{Cdouble}, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t, Ref{Cdouble}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{Cdouble}, alpha), Base.cconvert(Ref{Cdouble}, A), lda, Base.cconvert(Ref{Cdouble}, B), ldb, Base.cconvert(Ref{Cdouble}, C), ldc)
end

function cublasXtDtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Csize_t, B::Ptr{Cdouble}, ldb::Csize_t, C::Ptr{Cdouble}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtDtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtCtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{cuComplex, 1}, A::Array{cuComplex, 1}, lda::Csize_t, B::Array{cuComplex, 1}, ldb::Csize_t, C::Array{cuComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{cuComplex}, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t, Ref{cuComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuComplex}, alpha), Base.cconvert(Ref{cuComplex}, A), lda, Base.cconvert(Ref{cuComplex}, B), ldb, Base.cconvert(Ref{cuComplex}, C), ldc)
end

function cublasXtCtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Csize_t, B::Ptr{cuComplex}, ldb::Csize_t, C::Ptr{cuComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtCtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{cuComplex}, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t, Ptr{cuComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end

function cublasXtZtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Array{cuDoubleComplex, 1}, A::Array{cuDoubleComplex, 1}, lda::Csize_t, B::Array{cuDoubleComplex, 1}, ldb::Csize_t, C::Array{cuDoubleComplex, 1}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ref{cuDoubleComplex}, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t, Ref{cuDoubleComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, Base.cconvert(Ref{cuDoubleComplex}, alpha), Base.cconvert(Ref{cuDoubleComplex}, A), lda, Base.cconvert(Ref{cuDoubleComplex}, B), ldb, Base.cconvert(Ref{cuDoubleComplex}, C), ldc)
end

function cublasXtZtrmm(handle::cublasXtHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Csize_t, n::Csize_t, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Csize_t, B::Ptr{cuDoubleComplex}, ldb::Csize_t, C::Ptr{cuDoubleComplex}, ldc::Csize_t)::cublasStatus_t
    return ccall((:cublasXtZtrmm, libcublas), cublasStatus_t, (cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Csize_t, Csize_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t, Ptr{cuDoubleComplex}, Csize_t,), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end



