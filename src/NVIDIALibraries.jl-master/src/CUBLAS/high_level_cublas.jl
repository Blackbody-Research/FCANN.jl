#=*
* High level CUBLAS functions
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

export cublasGetErrorName

function cublasGetErrorName(error_status::cublasStatus_t)::String
    if (error_status == cublasStatus_t(0))
        return "CUBLAS_STATUS_SUCCESS"
    elseif (error_status == cublasStatus_t(1))
        return "CUBLAS_STATUS_NOT_INITIALIZED"
    elseif (error_status == cublasStatus_t(3))
        return "CUBLAS_STATUS_ALLOC_FAILED"
    elseif (error_status == cublasStatus_t(7))
        return "CUBLAS_STATUS_INVALID_VALUE"
    elseif (error_status == cublasStatus_t(8))
        return "CUBLAS_STATUS_ARCH_MISMATCH"
    elseif (error_status == cublasStatus_t(11))
        return "CUBLAS_STATUS_MAPPING_ERROR"
    elseif (error_status == cublasStatus_t(13))
        return "CUBLAS_STATUS_EXECUTION_FAILED"
    elseif (error_status == cublasStatus_t(14))
        return "CUBLAS_STATUS_INTERNAL_ERROR"
    elseif (error_status == cublasStatus_t(15))
        return "CUBLAS_STATUS_NOT_SUPPORTED"
    elseif (error_status == cublasStatus_t(16))
        return "CUBLAS_STATUS_LICENSE_ERROR"
    else
        error("cublasGetErrorName(): invalid cublasStatus_t value, ", error_status, "!")
    end
end

function cublasCreate_v2()::cublasHandle_t
    local handle_array::Array{cublasHandle_t, 1} = [C_NULL]
    local result::cublasStatus_t = cublasCreate_v2(handle_array)
    @assert (result == CUBLAS_STATUS_SUCCESS) ("cublasCreate_v2() error: " * cublasGetErrorName(result))
    return pop!(handle_array)
end

# CUDAArray functions
using ..DeviceArray

# GEMM functions

function cublasHgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float16, A::CUDAArray, B::CUDAArray, beta::Float16, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float16) &&
            (B.element_type == Float16) &&
            (C.element_type == Float16))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint
    local n::Cint
    local k::Cint

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldb::Cint = max(1, B.size[1])
    local ldc::Cint = max(1, C.size[1])

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
        m = A.size[1]
        k = A.size[2]
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
        m = A.size[2]
        k = A.size[1]
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
        m = A.size[2]
        k = A.size[1]
    end
    if (tb === 'N')
        transB = CUBLAS_OP_N
        n = B.size[2]
    elseif (tb === 'T')
        transB = CUBLAS_OP_T
        n = B.size[1]
    elseif (tb === 'C')
        transB = CUBLAS_OP_C
        n = B.size[1]
    end

    # check if dimensions are wrong
    if (m != C.size[1])
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of rows in C (m), $(C.size[1])"))
    elseif (n != C.size[2])
        throw(DimensionMismatch("number of columns in B (n), $n, does not equal the number of columns in C (n), $(C.size[2])"))
    elseif (!(((tb === 'N') && (k == B.size[1])) || (((tb === 'T') || (tb === 'C')) && (k == B.size[2]))))
        throw(DimensionMismatch("number of columns in A (k) does not equal the number of rows in B (k)"))
    end

    local result::cublasStatus_t = cublasHgemm(handle, transA, transB, m, n, k, alpha, Ptr{Float16}(A.ptr), lda, Ptr{Float16}(B.ptr), ldb, beta, Ptr{Float16}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasHgemm() error: " * cublasGetErrorName(result))
end

# use cublasSgemm_v2() over legacy cublasSgemm()
function cublasSgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float32, A::CUDAArray, B::CUDAArray, beta::Float32, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float32) &&
            (B.element_type == Float32) &&
            (C.element_type == Float32))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint
    local n::Cint
    local k::Cint

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldb::Cint = max(1, B.size[1])
    local ldc::Cint = max(1, C.size[1])

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
        m = A.size[1]
        k = A.size[2]
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
        m = A.size[2]
        k = A.size[1]
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
        m = A.size[2]
        k = A.size[1]
    end
    if (tb === 'N')
        transB = CUBLAS_OP_N
        n = B.size[2]
    elseif (tb === 'T')
        transB = CUBLAS_OP_T
        n = B.size[1]
    elseif (tb === 'C')
        transB = CUBLAS_OP_C
        n = B.size[1]
    end

    # check if dimensions are wrong
    if (m != C.size[1])
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of rows in C (m), $(C.size[1])"))
    elseif (n != C.size[2])
        throw(DimensionMismatch("number of columns in B (n), $n, does not equal the number of columns in C (n), $(C.size[2])"))
    elseif (!(((tb === 'N') && (k == B.size[1])) || (((tb === 'T') || (tb === 'C')) && (k == B.size[2]))))
        throw(DimensionMismatch("number of columns in A (k) does not equal the number of rows in B (k)"))
    end

    local result::cublasStatus_t = cublasSgemm_v2(handle, transA, transB, m, n, k, alpha, Ptr{Float32}(A.ptr), lda, Ptr{Float32}(B.ptr), ldb, beta, Ptr{Float32}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasSgemm() error: " * cublasGetErrorName(result))
end

# use cublasDgemm_v2() over legacy cublasDgemm()
function cublasDgemm(handle::cublasHandle_t, ta::Char, tb::Char, alpha::Float64, A::CUDAArray, B::CUDAArray, beta::Float64, C::CUDAArray)::Nothing
    @assert ((A.element_type == Float64) &&
            (B.element_type == Float64) &&
            (C.element_type == Float64))
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint
    local n::Cint
    local k::Cint

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldb::Cint = max(1, B.size[1])
    local ldc::Cint = max(1, C.size[1])

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
        m = A.size[1]
        k = A.size[2]
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
        m = A.size[2]
        k = A.size[1]
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
        m = A.size[2]
        k = A.size[1]
    end
    if (tb === 'N')
        transB = CUBLAS_OP_N
        n = B.size[2]
    elseif (tb === 'T')
        transB = CUBLAS_OP_T
        n = B.size[1]
    elseif (tb === 'C')
        transB = CUBLAS_OP_C
        n = B.size[1]
    end

    # check if dimensions are wrong
    if (m != C.size[1])
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of rows in C (m), $(C.size[1])"))
    elseif (n != C.size[2])
        throw(DimensionMismatch("number of columns in B (n), $n, does not equal the number of columns in C (n), $(C.size[2])"))
    elseif (!(((tb === 'N') && (k == B.size[1])) || (((tb === 'T') || (tb === 'C')) && (k == B.size[2]))))
        throw(DimensionMismatch("number of columns in A (k) does not equal the number of rows in B (k)"))
    end

    local result::cublasStatus_t = cublasDgemm_v2(handle, transA, transB, m, n, k, alpha, Ptr{Float64}(A.ptr), lda, Ptr{Float64}(B.ptr), ldb, beta, Ptr{Float64}(C.ptr), ldc)
    @assert (result == cudaSuccess) ("cublasDgemm() error: " * cublasGetErrorName(result))
end

function get_cuda_datatype(dt::DataType)::cudaDataType
    if (dt === __half)
        return CUDA_R_16F
    elseif (dt === __half2)
        return CUDA_C_16F
    elseif (dt === Float32)
        return CUDA_R_32F
    elseif (dt === cuFloatComplex)
        return CUDA_C_32F
    elseif (dt === Float64)
        return CUDA_R_64F
    elseif (dt === cuDoubleComplex)
        return CUDA_C_64F
    elseif (dt === Int8)
        return CUDA_R_8I
    elseif (dt === char2)
        return CUDA_C_8I
    elseif (dt === UInt8)
        return CUDA_R_8U
    elseif (dt === uchar2)
        return CUDA_C_8U
    elseif (dt === Int32)
        return CUDA_R_32I
    elseif (dt === int2)
        return CUDA_C_32I
    elseif (dt === UInt32)
        return CUDA_R_32U
    elseif (dt === uint2)
        return CUDA_C_32U
    else
        error("get_cuda_datatype() error: could not find corresponding cudaDataType for ", dt, "!")
    end
end

function cublasGemmEx(handle::cublasHandle_t, algo::cublasGemmAlgo_t, ta::Char, tb::Char, alpha::T, A::CUDAArray, B::CUDAArray, beta::T, C::CUDAArray)::Nothing where T
    local transA::cublasOperation_t
    local transB::cublasOperation_t
    local m::Cint
    local n::Cint
    local k::Cint

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldb::Cint = max(1, B.size[1])
    local ldc::Cint = max(1, C.size[1])

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
        m = A.size[1]
        k = A.size[2]
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
        m = A.size[2]
        k = A.size[1]
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
        m = A.size[2]
        k = A.size[1]
    end
    if (tb === 'N')
        transB = CUBLAS_OP_N
        n = B.size[2]
    elseif (tb === 'T')
        transB = CUBLAS_OP_T
        n = B.size[1]
    elseif (tb === 'C')
        transB = CUBLAS_OP_C
        n = B.size[1]
    end

    # check if dimensions are wrong
    if (m != C.size[1])
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of rows in C (m), $(C.size[1])"))
    elseif (n != C.size[2])
        throw(DimensionMismatch("number of columns in B (n), $n, does not equal the number of columns in C (n), $(C.size[2])"))
    elseif (!(((tb === 'N') && (k == B.size[1])) || (((tb === 'T') || (tb === 'C')) && (k == B.size[2]))))
        throw(DimensionMismatch("number of columns in A (k) does not equal the number of rows in B (k)"))
    end

    @assert (A.element_type == B.element_type)

    local AB_cdt::cudaDataType = get_cuda_datatype(A.element_type)
    local C_cdt::cudaDataType = get_cuda_datatype(C.element_type)
    local compute_cdt::cudaDataType = get_cuda_datatype(T)

    @assert ((AB_cdt, C_cdt, compute_cdt) in ((CUDA_R_16F, CUDA_R_16F, CUDA_R_16F),
                                            (CUDA_R_8I, CUDA_R_32I, CUDA_R_32I),
                                            (CUDA_R_16F, CUDA_R_16F, CUDA_R_32F),
                                            (CUDA_R_8I, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_16F, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_32F, CUDA_R_32F, CUDA_R_32F),
                                            (CUDA_R_64F, CUDA_R_64F, CUDA_R_64F),
                                            (CUDA_C_8I, CUDA_C_32F, CUDA_C_32F),
                                            (CUDA_C_32F, CUDA_C_32F, CUDA_C_32F),
                                            (CUDA_C_64F, CUDA_C_64F, CUDA_C_64F))) ("cublasGemmEx() error: Atype/Btype ("
                                                                                    * string(AB_cdt) * "), Ctype ("
                                                                                    * string(C_cdt) * "), and computeType ("
                                                                                    * string(compute_cdt)
                                                                                    * ") is not a supported combination of cudaDataTypes!")


    local result::cublasStatus_t = cublasGemmEx(handle, transA, transB, m, n, k, alpha, A.ptr, AB_cdt, lda, B.ptr, AB_cdt, ldb, beta, C.ptr, C_cdt, ldc, compute_cdt, algo)
    @assert (result == cudaSuccess) ("cublasGemmEx() error: " * cublasGetErrorName(result))
end

# GEMV functions

# use cublasSgemv_v2() over legacy cublasSgemv()
function cublasSgemv(handle::cublasHandle_t, ta::Char, alpha::Float32, A::CUDAArray, x::CUDAArray, beta::Float32, y::CUDAArray)::Nothing
    @assert ((A.element_type == Float32) &&
            (x.element_type == Float32) &&
            (y.element_type == Float32))
    local transA::cublasOperation_t
    local m::Cint = A.size[1]
    local n::Cint = A.size[2]

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldx::Cint = 1
    local ldy::Cint = 1

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
    end

    # check if dimensions are wrong
    if ((ta === 'N') && (n != x.size[1]))
        throw(DimensionMismatch("number of columns in A (n), $n, does not equal the number of components in x (n), $(x.size[1])"))
    elseif (((ta === 'T') || (ta === 'C')) && (m != x.size[1]))
        throw(DimensionMismatch("number of columns in A (m), $m, does not equal the number of components in x (m), $(x.size[1])"))
    elseif ((ta === 'N') && (m != y.size[1]))
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of components in y (m), $(y.size[1])"))
    elseif (((ta === 'T') || (ta === 'C')) && (n != y.size[1]))
        throw(DimensionMismatch("number of rows in A (n), $n, does not equal the number of components in y (n), $(y.size[1])"))
    elseif (length(x.size) != 1)
        throw(DimensionMismatch("x ($(length(x.size)) dimension(s)) is not a vector"))
    elseif (length(y.size) != 1)
        throw(DimensionMismatch("y ($(length(y.size)) dimension(s)) is not a vector"))
    end

    local result::cublasStatus_t = cublasSgemv_v2(handle, transA, m, n, alpha, Ptr{Float32}(A.ptr), lda, Ptr{Float32}(x.ptr), ldx, beta, Ptr{Float32}(y.ptr), ldy)
    @assert (result == cudaSuccess) ("cublasSgemv() error: " * cublasGetErrorName(result))
end

# use cublasDgemv_v2() over legacy cublasDgemv()
function cublasDgemv(handle::cublasHandle_t, ta::Char, alpha::Float64, A::CUDAArray, x::CUDAArray, beta::Float64, y::CUDAArray)::Nothing
    @assert ((A.element_type == Float64) &&
            (x.element_type == Float64) &&
            (y.element_type == Float64))
    local transA::cublasOperation_t
    local m::Cint = A.size[1]
    local n::Cint = A.size[2]

    # assign leading dimensions of A, B, and C
    local lda::Cint = max(1, A.size[1])
    local ldx::Cint = 1
    local ldy::Cint = 1

    # assign values of M, N, K
    if (ta === 'N')
        transA = CUBLAS_OP_N
    elseif (ta === 'T')
        transA = CUBLAS_OP_T
    elseif (ta === 'C')
        transA = CUBLAS_OP_C
    end

    # check if dimensions are wrong
    if ((ta === 'N') && (n != x.size[1]))
        throw(DimensionMismatch("number of columns in A (n), $n, does not equal the number of components in x (n), $(x.size[1])"))
    elseif (((ta === 'T') || (ta === 'C')) && (m != x.size[1]))
        throw(DimensionMismatch("number of columns in A (m), $m, does not equal the number of components in x (m), $(x.size[1])"))
    elseif ((ta === 'N') && (m != y.size[1]))
        throw(DimensionMismatch("number of rows in A (m), $m, does not equal the number of components in y (m), $(y.size[1])"))
    elseif (((ta === 'T') || (ta === 'C')) && (n != y.size[1]))
        throw(DimensionMismatch("number of rows in A (n), $n, does not equal the number of components in y (n), $(y.size[1])"))
    elseif (length(x.size) != 1)
        throw(DimensionMismatch("x ($(length(x.size)) dimension(s)) is not a vector"))
    elseif (length(y.size) != 1)
        throw(DimensionMismatch("y ($(length(y.size)) dimension(s)) is not a vector"))
    end

    local result::cublasStatus_t = cublasDgemv_v2(handle, transA, m, n, alpha, Ptr{Float64}(A.ptr), lda, Ptr{Float64}(x.ptr), ldx, beta, Ptr{Float64}(y.ptr), ldy)
    @assert (result == cudaSuccess) ("cublasDgemv() error: " * cublasGetErrorName(result))
end
