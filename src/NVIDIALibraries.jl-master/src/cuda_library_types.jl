#=*
* CUDA library type definitions
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

export
    cudaDataType,
    CUDA_R_16F,
    CUDA_C_16F,
    CUDA_R_32F,
    CUDA_C_32F,
    CUDA_R_64F,
    CUDA_C_64F,
    CUDA_R_8I,
    CUDA_C_8I,
    CUDA_R_8U,
    CUDA_C_8U,
    CUDA_R_32I,
    CUDA_C_32I,
    CUDA_R_32U,
    CUDA_C_32U,
    libraryPropertyType,
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_LEVEL

const cudaDataType = Cuint

# possible cudaDataType values
const CUDA_R_16F    = cudaDataType(2)
const CUDA_C_16F    = cudaDataType(6)
const CUDA_R_32F    = cudaDataType(0)
const CUDA_C_32F    = cudaDataType(4)
const CUDA_R_64F    = cudaDataType(1)
const CUDA_C_64F    = cudaDataType(5)
const CUDA_R_8I     = cudaDataType(3)
const CUDA_C_8I     = cudaDataType(7)
const CUDA_R_8U     = cudaDataType(8)
const CUDA_C_8U     = cudaDataType(9)
const CUDA_R_32I    = cudaDataType(10)
const CUDA_C_32I    = cudaDataType(11)
const CUDA_R_32U    = cudaDataType(12)
const CUDA_C_32U    = cudaDataType(13)

const libraryPropertyType = Cuint

# possible libraryPropertyType values
const MAJOR_VERSION = libraryPropertyType(0)
const MINOR_VERSION = libraryPropertyType(1)
const PATCH_LEVEL   = libraryPropertyType(2)

