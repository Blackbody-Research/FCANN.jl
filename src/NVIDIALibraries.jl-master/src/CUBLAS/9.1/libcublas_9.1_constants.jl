#=*
* CUBLAS API v9.1 definitions
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

# CUBLAS constants from 'cublas_api.h'

const cublasStatus_t = Cuint

# possible cublasStatus_t values
const CUBLAS_STATUS_SUCCESS             = cublasStatus_t(0)
const CUBLAS_STATUS_NOT_INITIALIZED     = cublasStatus_t(1)
const CUBLAS_STATUS_ALLOC_FAILED        = cublasStatus_t(3)
const CUBLAS_STATUS_INVALID_VALUE       = cublasStatus_t(7)
const CUBLAS_STATUS_ARCH_MISMATCH       = cublasStatus_t(8)
const CUBLAS_STATUS_MAPPING_ERROR       = cublasStatus_t(11)
const CUBLAS_STATUS_EXECUTION_FAILED    = cublasStatus_t(13)
const CUBLAS_STATUS_INTERNAL_ERROR      = cublasStatus_t(14)
const CUBLAS_STATUS_NOT_SUPPORTED       = cublasStatus_t(15)
const CUBLAS_STATUS_LICENSE_ERROR       = cublasStatus_t(16)

const cublasFillMode_t = Cuint

# possible cublasFillMode_t values
const CUBLAS_FILL_MODE_LOWER    = cublasFillMode_t(0)
const CUBLAS_FILL_MODE_UPPER    = cublasFillMode_t(1)

const cublasDiagType_t = Cuint

# possible cublasDiagType_t values
const CUBLAS_DIAG_NON_UNIT  = cublasDiagType_t(0)
const CUBLAS_DIAG_UNIT      = cublasDiagType_t(1)

const cublasSideMode_t = Cuint

# possible cublasSideMode_t values
const CUBLAS_SIDE_LEFT  = cublasSideMode_t(0)
const CUBLAS_SIDE_RIGHT = cublasSideMode_t(1)

const cublasOperation_t = Cuint

# possible cublasOperation_t values
const CUBLAS_OP_N   = cublasOperation_t(0)
const CUBLAS_OP_T   = cublasOperation_t(1)
const CUBLAS_OP_C   = cublasOperation_t(2)

const cublasPointerMode_t = Cuint

# possible cublasPointerMode_t values
const CUBLAS_POINTER_MODE_HOST      = cublasPointerMode_t(0)
const CUBLAS_POINTER_MODE_DEVICE    = cublasPointerMode_t(1)

const cublasAtomicsMode_t = Cuint

# possible cublasAtomicsMode_t values
const CUBLAS_ATOMICS_NOT_ALLOWED    = cublasAtomicsMode_t(0)
const CUBLAS_ATOMICS_ALLOWED        = cublasAtomicsMode_t(1)

const cublasGemmAlgo_t = Cint

# possible cublasGemmAlgo_t values
const CUBLAS_GEMM_DFALT             = cublasGemmAlgo_t(-1)
const CUBLAS_GEMM_DEFAULT           = cublasGemmAlgo_t(-1)
const CUBLAS_GEMM_ALGO0             = cublasGemmAlgo_t(0)
const CUBLAS_GEMM_ALGO1             = cublasGemmAlgo_t(1)
const CUBLAS_GEMM_ALGO2             = cublasGemmAlgo_t(2)
const CUBLAS_GEMM_ALGO3             = cublasGemmAlgo_t(3)
const CUBLAS_GEMM_ALGO4             = cublasGemmAlgo_t(4)
const CUBLAS_GEMM_ALGO5             = cublasGemmAlgo_t(5)
const CUBLAS_GEMM_ALGO6             = cublasGemmAlgo_t(6)
const CUBLAS_GEMM_ALGO7             = cublasGemmAlgo_t(7)
const CUBLAS_GEMM_ALGO8             = cublasGemmAlgo_t(8)
const CUBLAS_GEMM_ALGO9             = cublasGemmAlgo_t(9)
const CUBLAS_GEMM_ALGO10            = cublasGemmAlgo_t(10)
const CUBLAS_GEMM_ALGO11            = cublasGemmAlgo_t(11)
const CUBLAS_GEMM_ALGO12            = cublasGemmAlgo_t(12)
const CUBLAS_GEMM_ALGO13            = cublasGemmAlgo_t(13)
const CUBLAS_GEMM_ALGO14            = cublasGemmAlgo_t(14)
const CUBLAS_GEMM_ALGO15            = cublasGemmAlgo_t(15)
const CUBLAS_GEMM_ALGO16            = cublasGemmAlgo_t(16)
const CUBLAS_GEMM_ALGO17            = cublasGemmAlgo_t(17)
const CUBLAS_GEMM_DEFAULT_TENSOR_OP = cublasGemmAlgo_t(99)
const CUBLAS_GEMM_DFALT_TENSOR_OP   = cublasGemmAlgo_t(99)
const CUBLAS_GEMM_ALGO0_TENSOR_OP   = cublasGemmAlgo_t(100)
const CUBLAS_GEMM_ALGO1_TENSOR_OP   = cublasGemmAlgo_t(101)
const CUBLAS_GEMM_ALGO2_TENSOR_OP   = cublasGemmAlgo_t(102)
const CUBLAS_GEMM_ALGO3_TENSOR_OP   = cublasGemmAlgo_t(103)
const CUBLAS_GEMM_ALGO4_TENSOR_OP   = cublasGemmAlgo_t(104)

const cublasMath_t = Cuint

# possible cublasMath_t values
const CUBLAS_DEFAULT_MATH   = cublasMath_t(0)
const CUBLAS_TENSOR_OP_MATH = cublasMath_t(1)

const cublasDataType_t = cudaDataType

const cublasHandle_t = Ptr{Nothing}

# CUBLAS XT constants from 'cublasXt.h'
const cublasXtHandle_t = Ptr{Nothing}

const cublasXtPinnedMemMode_t = Cuint

# possible cublasXtPinnedMemMode_t values
const CUBLASXT_PINNING_DISABLED = cublasXtPinnedMemMode_t(0)
const CUBLASXT_PINNING_ENABLED  = cublasXtPinnedMemMode_t(1)

const cublasXtOpType_t = Cuint

# possible cublasXtOpType_t values
const CUBLASXT_FLOAT            = cublasXtOpType_t(0)
const CUBLASXT_DOUBLE           = cublasXtOpType_t(1)
const CUBLASXT_COMPLEX          = cublasXtOpType_t(2)
const CUBLASXT_DOUBLECOMPLEX    = cublasXtOpType_t(3)

const cublasXtBlasOp_t = Cuint

# possible cublasXtBlasOp_t values
const CUBLASXT_GEMM         = cublasXtBlasOp_t(0)
const CUBLASXT_SYRK         = cublasXtBlasOp_t(1)
const CUBLASXT_HERK         = cublasXtBlasOp_t(2)
const CUBLASXT_SYMM         = cublasXtBlasOp_t(3)
const CUBLASXT_HEMM         = cublasXtBlasOp_t(4)
const CUBLASXT_TRSM         = cublasXtBlasOp_t(5)
const CUBLASXT_SYR2K        = cublasXtBlasOp_t(6)
const CUBLASXT_HER2K        = cublasXtBlasOp_t(7)
const CUBLASXT_SPMM         = cublasXtBlasOp_t(8)
const CUBLASXT_SYRKX        = cublasXtBlasOp_t(9)
const CUBLASXT_HERKX        = cublasXtBlasOp_t(10)
const CUBLASXT_TRMM         = cublasXtBlasOp_t(11)
const CUBLASXT_ROUTINE_MAX  = cublasXtBlasOp_t(12)


