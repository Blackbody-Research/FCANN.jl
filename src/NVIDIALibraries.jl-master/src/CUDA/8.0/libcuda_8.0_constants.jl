#=*
* CUDA API v8.0 definitions
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

# These definitions are defined according to how they were declared
# in CUDA 8.0's 'cuda.h'.

const cuuint32_t = Cuint

const cuuint64_t = Culonglong

const CUDA_VERSION = 8000

# CUdeviceptr available since CUDA 3.2
const CUdeviceptr = Ptr{Nothing}

const CUdevice = Cint

const CUcontext = Ptr{Nothing}

const CUmodule = Ptr{Nothing}

const CUfunction = Ptr{Nothing}

const CUarray = Ptr{Nothing}

const CUmipmappedArray = Ptr{Nothing}

const CUtexref = Ptr{Nothing}

const CUsurfref = Ptr{Nothing}

const CUevent = Ptr{Nothing}

const CUstream = Ptr{Nothing}

const CUgraphicsResource = Ptr{Nothing}

const CUtexObject = Culonglong

const CUsurfObject = Culonglong

struct CUuuid
    bytes::NTuple{16, UInt8}
end

Base.zero(::Type{CUuuid}) = CUuuid(
    (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))
Base.zero(x::CUuuid) = zero(typeof(x))

# CU_IPC_HANDLE_SIZE as 64 since CUDA 4.1
const CU_IPC_HANDLE_SIZE = Csize_t(64)

# CUDA IPC event handle available since CUDA 4.1
struct CUipcEventHandle
    reserved::NTuple{Int(CU_IPC_HANDLE_SIZE), UInt8}
end

Base.zero(::Type{CUipcEventHandle}) = CUipcEventHandle(
    (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))
Base.zero(x::CUipcEventHandle) = zero(typeof(x))

# CUDA IPC mem handle available since CUDA 4.1
struct CUipcMemHandle
    reserved::NTuple{Int(CU_IPC_HANDLE_SIZE), UInt8}
end

Base.zero(::Type{CUipcMemHandle}) = CUipcMemHandle(
    (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))
Base.zero(x::CUipcMemHandle) = zero(typeof(x))

# CUDA IPC mem flags available since CUDA 4.1
const CUipcMem_flags = Cuint

# possible CUipcMem_flags values
const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = CUipcMem_flags(0x1)

const CUmemAttach_flags = Cuint

# possible CUmemAttach_flags values
const CU_MEM_ATTACH_GLOBAL = CUmemAttach_flags(0x1)
const CU_MEM_ATTACH_HOST   = CUmemAttach_flags(0x2)
const CU_MEM_ATTACH_SINGLE = CUmemAttach_flags(0x4)

const CUctx_flags = Cuint

# possible CUctx_flags values
const CU_CTX_SCHED_AUTO          = CUctx_flags(0x00)
const CU_CTX_SCHED_SPIN          = CUctx_flags(0x01)
const CU_CTX_SCHED_YIELD         = CUctx_flags(0x02)
const CU_CTX_SCHED_BLOCKING_SYNC = CUctx_flags(0x04)
const CU_CTX_BLOCKING_SYNC       = CUctx_flags(0x04)
const CU_CTX_SCHED_MASK          = CUctx_flags(0x07)
const CU_CTX_MAP_HOST            = CUctx_flags(0x08)
const CU_CTX_LMEM_RESIZE_TO_MAX  = CUctx_flags(0x10)
const CU_CTX_FLAGS_MASK          = CUctx_flags(0x1f)

const CUstream_flags = Cuint

# possible CUstream_flags values
const CU_STREAM_DEFAULT      = CUstream_flags(0x0)
const CU_STREAM_NON_BLOCKING = CUstream_flags(0x1)

const CU_STREAM_LEGACY     = CUstream(UInt(0x1))
const CU_STREAM_PER_THREAD = CUstream(UInt(0x2))

const CUevent_flags = Cuint

# possible CUevent_flags values
const CU_EVENT_DEFAULT        = CUevent_flags(0x0)
const CU_EVENT_BLOCKING_SYNC  = CUevent_flags(0x1)
const CU_EVENT_DISABLE_TIMING = CUevent_flags(0x2)
const CU_EVENT_INTERPROCESS   = CUevent_flags(0x4)

# CUstreamWaitValue_flags available since CUDA 8.0
const CUstreamWaitValue_flags = Cuint

# possible CUstreamWaitValue_flags values
const CU_STREAM_WAIT_VALUE_GEQ   = CUstreamWaitValue_flags(0x0)
const CU_STREAM_WAIT_VALUE_EQ    = CUstreamWaitValue_flags(0x1)
const CU_STREAM_WAIT_VALUE_AND   = CUstreamWaitValue_flags(0x2)
const CU_STREAM_WAIT_VALUE_NOR   = CUstreamWaitValue_flags(0x3)
const CU_STREAM_WAIT_VALUE_FLUSH = CUstreamWaitValue_flags(0x1) << 30

# CUstreamWriteValue_flags available since CUDA 8.0
const CUstreamWriteValue_flags = Cuint

# possible CUstreamWriteValue_flags values
const CU_STREAM_WRITE_VALUE_DEFAULT           = CUstreamWriteValue_flags(0x0)
const CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = CUstreamWriteValue_flags(0x1)

# CUstreamBatchMemOpType available since CUDA 8.0
const CUstreamBatchMemOpType = Cuint

# possible CUstreamBatchMemOpType values
const CU_STREAM_MEM_OP_WAIT_VALUE_32       = CUstreamBatchMemOpType(1)
const CU_STREAM_MEM_OP_WRITE_VALUE_32      = CUstreamBatchMemOpType(2)
const CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = CUstreamBatchMemOpType(3)

struct CUstreamBatchMemOpParams_operation
    operation::CUstreamBatchMemOpType
    pad::NTuple{11, Cuint}
end

Base.zero(::Type{CUstreamBatchMemOpParams_operation}) = CUstreamBatchMemOpParams_operation(
    0x00000000,
    (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000))
Base.zero(x::CUstreamBatchMemOpParams_operation) = zero(typeof(x))

# should be 48 bytes, WORD_SIZE affects CUdeviceptr size
if (Sys.WORD_SIZE == 64)
    struct CUstreamBatchMemOpParams_waitValue_32
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value::cuuint32_t
        value_pad::cuuint32_t   # pad the rest of the union
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{1, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_waitValue_32}) = CUstreamBatchMemOpParams_waitValue_32(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint32_t(0),
        cuuint32_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0),))
elseif (Sys.WORD_SIZE == 32)
    struct CUstreamBatchMemOpParams_waitValue_32
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value::cuuint32_t
        value_pad::cuuint32_t   # pad the rest of the union
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{6, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_waitValue_32}) = CUstreamBatchMemOpParams_waitValue_32(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint32_t(0),
        cuuint32_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::CUstreamBatchMemOpParams_waitValue_32) = zero(typeof(x))

# should be 48 bytes, WORD_SIZE affects CUdeviceptr size
if (Sys.WORD_SIZE == 64)
    struct CUstreamBatchMemOpParams_waitValue_64
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value64::cuuint64_t
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{2, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_waitValue_64}) = CUstreamBatchMemOpParams_waitValue_64(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint64_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct CUstreamBatchMemOpParams_waitValue_64
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value64::cuuint64_t
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{6, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_waitValue_64}) = CUstreamBatchMemOpParams_waitValue_64(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint64_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::CUstreamBatchMemOpParams_waitValue_64) = zero(typeof(x))

# should be 48 bytes, WORD_SIZE affects CUdeviceptr size
if (Sys.WORD_SIZE == 64)
    struct CUstreamBatchMemOpParams_writeValue_32
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value::cuuint32_t
        value_pad::cuuint32_t   # pad the rest of the union
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{1, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_writeValue_32}) = CUstreamBatchMemOpParams_writeValue_32(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint32_t(0),
        cuuint32_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0),))
elseif (Sys.WORD_SIZE == 32)
    struct CUstreamBatchMemOpParams_writeValue_32
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value::cuuint32_t
        value_pad::cuuint32_t   # pad the rest of the union
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{6, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_writeValue_32}) = CUstreamBatchMemOpParams_writeValue_32(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint32_t(0),
        cuuint32_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::CUstreamBatchMemOpParams_writeValue_32) = zero(typeof(x))

# should be 48 bytes, WORD_SIZE affects CUdeviceptr size
if (Sys.WORD_SIZE == 64)
    struct CUstreamBatchMemOpParams_writeValue_64
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value64::cuuint64_t
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{2, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_writeValue_64}) = CUstreamBatchMemOpParams_writeValue_64(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint64_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct CUstreamBatchMemOpParams_writeValue_64
        operation::CUstreamBatchMemOpType
        address::CUdeviceptr
        value64::cuuint64_t
        flags::Cuint
        alias::CUdeviceptr
        pad::NTuple{6, Cuint}
    end
    Base.zero(::Type{CUstreamBatchMemOpParams_writeValue_64}) = CUstreamBatchMemOpParams_writeValue_64(
        CUstreamBatchMemOpType(0),
        CUdeviceptr(0),
        cuuint64_t(0),
        Cuint(0),
        CUdeviceptr(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::CUstreamBatchMemOpParams_writeValue_64) = zero(typeof(x))

# should be 48 bytes
struct CUstreamBatchMemOpParams_flushRemoteWrites
    operation::CUstreamBatchMemOpType
    flags::Cuint
    pad::NTuple{10, Cuint}
end

Base.zero(::Type{CUstreamBatchMemOpParams_flushRemoteWrites}) = CUstreamBatchMemOpParams_flushRemoteWrites(
    CUstreamBatchMemOpType(0),
    Cuint(0),
    (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
Base.zero(x::CUstreamBatchMemOpParams_flushRemoteWrites) = zero(typeof(x))

# should be 48 bytes
struct CUstreamBatchMemOpParams_pad
    pad::NTuple{6, cuuint64_t}
end

Base.zero(::Type{CUstreamBatchMemOpParams_pad}) = CUstreamBatchMemOpParams_pad(
    (cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0)))
Base.zero(x::CUstreamBatchMemOpParams_pad) = zero(typeof(x))

# CUstreamBatchMemOpParams available since CUDA 8.0
const CUstreamBatchMemOpParams = Union{CUstreamBatchMemOpParams_operation,
                                        CUstreamBatchMemOpParams_waitValue_32,
                                        CUstreamBatchMemOpParams_waitValue_64,
                                        CUstreamBatchMemOpParams_writeValue_32,
                                        CUstreamBatchMemOpParams_writeValue_64,
                                        CUstreamBatchMemOpParams_flushRemoteWrites,
                                        CUstreamBatchMemOpParams_pad}

# use 'CUstreamBatchMemOpParams_pad' by default for zero()
Base.zero(::Type{CUstreamBatchMemOpParams}) = CUstreamBatchMemOpParams_pad(
    (cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0), cuuint64_t(0)))
Base.zero(x::CUstreamBatchMemOpParams) = zero(typeof(x))

@assert((sizeof(CUstreamBatchMemOpParams_operation) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_waitValue_32) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_waitValue_64) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_writeValue_32) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_writeValue_64) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_flushRemoteWrites) == 48) &&
        (sizeof(CUstreamBatchMemOpParams_pad) == 48))


# type casting between possible CUstreamBatchMemOpParams data types
CUstreamBatchMemOpParams_operation(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_operation(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_operation(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_operation(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_operation(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))
CUstreamBatchMemOpParams_operation(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_operation}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_waitValue_32(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_waitValue_32(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_waitValue_32(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_waitValue_32(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_waitValue_32(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))
CUstreamBatchMemOpParams_waitValue_32(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_waitValue_64(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_waitValue_64(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_waitValue_64(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_waitValue_64(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_waitValue_64(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))
CUstreamBatchMemOpParams_waitValue_64(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_waitValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_writeValue_32(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_writeValue_32(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_writeValue_32(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_writeValue_32(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_writeValue_32(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))
CUstreamBatchMemOpParams_writeValue_32(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_32}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_writeValue_64(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_writeValue_64(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_writeValue_64(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_writeValue_64(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_writeValue_64(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))
CUstreamBatchMemOpParams_writeValue_64(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_writeValue_64}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_flushRemoteWrites(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_flushRemoteWrites(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_flushRemoteWrites(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_flushRemoteWrites(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_flushRemoteWrites(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_flushRemoteWrites(csbmopp::CUstreamBatchMemOpParams_pad) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_pad}, Base.cconvert(Ref{CUstreamBatchMemOpParams_pad}, csbmopp))))

CUstreamBatchMemOpParams_pad(csbmopo::CUstreamBatchMemOpParams_operation) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_operation}, Base.cconvert(Ref{CUstreamBatchMemOpParams_operation}, csbmopo))))
CUstreamBatchMemOpParams_pad(csbmopwaitv32::CUstreamBatchMemOpParams_waitValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_32}, csbmopwaitv32))))
CUstreamBatchMemOpParams_pad(csbmopwaitv64::CUstreamBatchMemOpParams_waitValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_waitValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_waitValue_64}, csbmopwaitv64))))
CUstreamBatchMemOpParams_pad(csbmopwritev32::CUstreamBatchMemOpParams_writeValue_32) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_32}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_32}, csbmopwritev32))))
CUstreamBatchMemOpParams_pad(csbmopwritev64::CUstreamBatchMemOpParams_writeValue_64) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_writeValue_64}, Base.cconvert(Ref{CUstreamBatchMemOpParams_writeValue_64}, csbmopwritev64))))
CUstreamBatchMemOpParams_pad(csbmopfrw::CUstreamBatchMemOpParams_flushRemoteWrites) = Base.unsafe_load(Ptr{CUstreamBatchMemOpParams_pad}(Base.unsafe_convert(Ptr{CUstreamBatchMemOpParams_flushRemoteWrites}, Base.cconvert(Ref{CUstreamBatchMemOpParams_flushRemoteWrites}, csbmopfrw))))

const CUoccupancy_flags = Cuint

# possible CUoccpancy_flags values
const CU_OCCUPANCY_DEFAULT                  = CUoccupancy_flags(0x0)
const CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = CUoccupancy_flags(0x1)

const CUarray_format = Cuint

# possible CUarray_format values
const CU_AD_FORMAT_UNSIGNED_INT8  = CUarray_format(0x01)
const CU_AD_FORMAT_UNSIGNED_INT16 = CUarray_format(0x02)
const CU_AD_FORMAT_UNSIGNED_INT32 = CUarray_format(0x03)
const CU_AD_FORMAT_SIGNED_INT8    = CUarray_format(0x08)
const CU_AD_FORMAT_SIGNED_INT16   = CUarray_format(0x09)
const CU_AD_FORMAT_SIGNED_INT32   = CUarray_format(0x0a)
const CU_AD_FORMAT_HALF           = CUarray_format(0x10)
const CU_AD_FORMAT_FLOAT          = CUarray_format(0x20)

const CUaddress_mode = Cuint

# possible CUaddress_mode values
const CU_TR_ADDRESS_MODE_WRAP   = CUaddress_mode(0)
const CU_TR_ADDRESS_MODE_CLAMP  = CUaddress_mode(1)
const CU_TR_ADDRESS_MODE_MIRROR = CUaddress_mode(2)
const CU_TR_ADDRESS_MODE_BORDER = CUaddress_mode(3)

const CUfilter_mode = Cuint

# possible CUfilter_mode values
const CU_TR_FILTER_MODE_POINT  = CUfilter_mode(0)
const CU_TR_FILTER_MODE_LINEAR = CUfilter_mode(1)

const CUdevice_attribute = Cuint

# possible CUdevice_attribute values
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                        = CUdevice_attribute(1)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                              = CUdevice_attribute(2)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                              = CUdevice_attribute(3)
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                              = CUdevice_attribute(4)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                               = CUdevice_attribute(5)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                               = CUdevice_attribute(6)
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                               = CUdevice_attribute(7)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK                  = CUdevice_attribute(8)
const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK                      = CUdevice_attribute(8)
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                        = CUdevice_attribute(9)
const CU_DEVICE_ATTRIBUTE_WARP_SIZE                                    = CUdevice_attribute(10)
const CU_DEVICE_ATTRIBUTE_MAX_PITCH                                    = CUdevice_attribute(11)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK                      = CUdevice_attribute(12)
const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK                          = CUdevice_attribute(12)
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE                                   = CUdevice_attribute(13)
const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                            = CUdevice_attribute(14)
const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP                                  = CUdevice_attribute(15)
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                         = CUdevice_attribute(16)
const CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                          = CUdevice_attribute(17)
const CU_DEVICE_ATTRIBUTE_INTEGRATED                                   = CUdevice_attribute(18)
const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                          = CUdevice_attribute(19)
const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                                 = CUdevice_attribute(20)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH                      = CUdevice_attribute(21)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH                      = CUdevice_attribute(22)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT                     = CUdevice_attribute(23)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH                      = CUdevice_attribute(24)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT                     = CUdevice_attribute(25)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH                      = CUdevice_attribute(26)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH              = CUdevice_attribute(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT             = CUdevice_attribute(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS             = CUdevice_attribute(29)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH                = CUdevice_attribute(27)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT               = CUdevice_attribute(28)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES            = CUdevice_attribute(29)
const CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                            = CUdevice_attribute(30)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                           = CUdevice_attribute(31)
const CU_DEVICE_ATTRIBUTE_ECC_ENABLED                                  = CUdevice_attribute(32)
const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                                   = CUdevice_attribute(33)
const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                                = CUdevice_attribute(34)
const CU_DEVICE_ATTRIBUTE_TCC_DRIVER                                   = CUdevice_attribute(35)
const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                            = CUdevice_attribute(36)
const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH                      = CUdevice_attribute(37)
const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                                = CUdevice_attribute(38)
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR               = CUdevice_attribute(39)
const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                           = CUdevice_attribute(40)
const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                           = CUdevice_attribute(41)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH              = CUdevice_attribute(42)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS             = CUdevice_attribute(43)
const CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER                             = CUdevice_attribute(44)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH               = CUdevice_attribute(45)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT              = CUdevice_attribute(46)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE            = CUdevice_attribute(47)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE           = CUdevice_attribute(48)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE            = CUdevice_attribute(49)
const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                                = CUdevice_attribute(50)
const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT                      = CUdevice_attribute(51)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH                 = CUdevice_attribute(52)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH         = CUdevice_attribute(53)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS        = CUdevice_attribute(54)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH                      = CUdevice_attribute(55)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH                      = CUdevice_attribute(56)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT                     = CUdevice_attribute(57)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH                      = CUdevice_attribute(58)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT                     = CUdevice_attribute(59)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH                      = CUdevice_attribute(60)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH              = CUdevice_attribute(61)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS             = CUdevice_attribute(62)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH              = CUdevice_attribute(63)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT             = CUdevice_attribute(64)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS             = CUdevice_attribute(65)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH                 = CUdevice_attribute(66)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH         = CUdevice_attribute(67)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS        = CUdevice_attribute(68)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH               = CUdevice_attribute(69)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH               = CUdevice_attribute(70)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT              = CUdevice_attribute(71)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH               = CUdevice_attribute(72)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH            = CUdevice_attribute(73)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT           = CUdevice_attribute(74)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR                     = CUdevice_attribute(75)
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR                     = CUdevice_attribute(76)
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH            = CUdevice_attribute(77)
const CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED                  = CUdevice_attribute(78)
const CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED                    = CUdevice_attribute(79)
const CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED                     = CUdevice_attribute(80)
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR         = CUdevice_attribute(81)
const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR             = CUdevice_attribute(82)
const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                               = CUdevice_attribute(83)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                              = CUdevice_attribute(84)
const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID                     = CUdevice_attribute(85)
const CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED                 = CUdevice_attribute(86)
const CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO        = CUdevice_attribute(87)
const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS                       = CUdevice_attribute(88)
const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS                    = CUdevice_attribute(89)
const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED                 = CUdevice_attribute(90)
const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM      = CUdevice_attribute(91)
#=
In CUDA 8.0, 'CU_DEVICE_ATTRIBUTE_MAX' is 92 after enumerating from
'CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM'
=#
const CU_DEVICE_ATTRIBUTE_MAX                                          = CUdevice_attribute(92)

struct CUdevprop
    maxThreadsPerBlock::Cint
    maxThreadsDim::NTuple{3, Cint}
    maxGridSize::NTuple{3, Cint}
    sharedMemPerBlock::Cint
    totalConstantMemory::Cint
    SIMDWidth::Cint
    memPitch::Cint
    regsPerBlock::Cint
    clockRate::Cint
    textureAlign::Cint
end

Base.zero(::Type{CUdevprop})=CUdevprop(Cint(0),
                                    (Cint(0), Cint(0), Cint(0)),
                                    (Cint(0), Cint(0), Cint(0)),
                                    Cint(0),
                                    Cint(0),
                                    Cint(0),
                                    Cint(0),
                                    Cint(0),
                                    Cint(0),
                                    Cint(0))
Base.zero(x::CUdevprop) = zero(typeof(x))

const CUpointer_attribute = Cuint

# possible CUpointer_attribute values
const CU_POINTER_ATTRIBUTE_CONTEXT        = CUpointer_attribute(1)
const CU_POINTER_ATTRIBUTE_MEMORY_TYPE    = CUpointer_attribute(2)
const CU_POINTER_ATTRIBUTE_DEVICE_POINTER = CUpointer_attribute(3)
const CU_POINTER_ATTRIBUTE_HOST_POINTER   = CUpointer_attribute(4)
const CU_POINTER_ATTRIBUTE_P2P_TOKENS     = CUpointer_attribute(5)
const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS    = CUpointer_attribute(6)
const CU_POINTER_ATTRIBUTE_BUFFER_ID      = CUpointer_attribute(7)
const CU_POINTER_ATTRIBUTE_IS_MANAGED     = CUpointer_attribute(8)

const CUfunction_attribute = Cuint

# possible CUfunction_attribute values
const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK             = CUfunction_attribute(0)
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES                 = CUfunction_attribute(1)
const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES                  = CUfunction_attribute(2)
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                  = CUfunction_attribute(3)
const CU_FUNC_ATTRIBUTE_NUM_REGS                          = CUfunction_attribute(4)
const CU_FUNC_ATTRIBUTE_PTX_VERSION                       = CUfunction_attribute(5)
const CU_FUNC_ATTRIBUTE_BINARY_VERSION                    = CUfunction_attribute(6)
const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA                     = CUfunction_attribute(7)
#=
In CUDA 8.0, 'CU_FUNC_ATTRIBUTE_MAX' is 8 by enumerating from
'CU_FUNC_ATTRIBUTE_CACHE_MODE_CA'.
=#
const CU_FUNC_ATTRIBUTE_MAX                               = CUfunction_attribute(8)

const CUfunc_cache = Cuint

# possible CUfunc_cache values
const CU_FUNC_CACHE_PREFER_NONE    = CUfunc_cache(0x00)
const CU_FUNC_CACHE_PREFER_SHARED  = CUfunc_cache(0x01)
const CU_FUNC_CACHE_PREFER_L1      = CUfunc_cache(0x02)
const CU_FUNC_CACHE_PREFER_EQUAL   = CUfunc_cache(0x03)

const CUsharedconfig = Cuint

# possible CUsharedconfig values
const CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = CUsharedconfig(0x00)
const CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = CUsharedconfig(0x01)
const CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = CUsharedconfig(0x02)

const CUmemorytype = Cuint

# possible CUmemorytype values
const CU_MEMORYTYPE_HOST    = CUmemorytype(0x01)
const CU_MEMORYTYPE_DEVICE  = CUmemorytype(0x02)
const CU_MEMORYTYPE_ARRAY   = CUmemorytype(0x03)
const CU_MEMORYTYPE_UNIFIED = CUmemorytype(0x04)

const CUcomputemode = Cuint

# possible CUcomputemode values
const CU_COMPUTEMODE_DEFAULT           = CUcomputemode(0)
const CU_COMPUTEMODE_PROHIBITED        = CUcomputemode(2)
const CU_COMPUTEMODE_EXCLUSIVE_PROCESS = CUcomputemode(3)

const CUmem_advise = Cuint

# possible CUmem_advise values
const CU_MEM_ADVISE_SET_READ_MOSTLY          = CUmem_advise(1)
const CU_MEM_ADVISE_UNSET_READ_MOSTLY        = CUmem_advise(2)
const CU_MEM_ADVISE_SET_PREFERRED_LOCATION   = CUmem_advise(3)
const CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = CUmem_advise(4)
const CU_MEM_ADVISE_SET_ACCESSED_BY          = CUmem_advise(5)
const CU_MEM_ADVISE_UNSET_ACCESSED_BY        = CUmem_advise(6)

const CUmem_range_attribute = Cuint

# possible CUmem_range_attribute values
const CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY            = CUmem_range_attribute(1)
const CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION     = CUmem_range_attribute(2)
const CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY            = CUmem_range_attribute(3)
const CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = CUmem_range_attribute(4)

const CUjit_option = Cuint

# possible CUjit_option values
const CU_JIT_MAX_REGISTERS                = CUjit_option(0)
const CU_JIT_THREADS_PER_BLOCK            = CUjit_option(1)
const CU_JIT_WALL_TIME                    = CUjit_option(2)
const CU_JIT_INFO_LOG_BUFFER              = CUjit_option(3)
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES   = CUjit_option(4)
const CU_JIT_ERROR_LOG_BUFFER             = CUjit_option(5)
const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  = CUjit_option(6)
const CU_JIT_OPTIMIZATION_LEVEL           = CUjit_option(7)
const CU_JIT_TARGET_FROM_CUCONTEXT        = CUjit_option(8)
const CU_JIT_TARGET                       = CUjit_option(9)
const CU_JIT_FALLBACK_STRATEGY            = CUjit_option(10)
const CU_JIT_GENERATE_DEBUG_INFO          = CUjit_option(11)
const CU_JIT_LOG_VERBOSE                  = CUjit_option(12)
const CU_JIT_GENERATE_LINE_INFO           = CUjit_option(13)
const CU_JIT_CACHE_MODE                   = CUjit_option(14)
#=
'CU_JIT_NEW_SM3X_OPT' and 'CU_JIT_FAST_COMPILE' are used for internal purposes only.
=#
const CU_JIT_NEW_SM3X_OPT                 = CUjit_option(15)
const CU_JIT_FAST_COMPILE                 = CUjit_option(16)
#=
In CUDA 8.0, 'CU_JIT_NUM_OPTIONS' is 17 by enumerating from
'CU_JIT_FAST_COMPILE'.
=#
const CU_JIT_NUM_OPTIONS                  = CUjit_option(17)

const CUjit_target = Cuint

# possible CUjit_target values
const CU_TARGET_COMPUTE_10 = CUjit_target(10)       # < Compute device class 1.0
const CU_TARGET_COMPUTE_11 = CUjit_target(11)       # < Compute device class 1.1
const CU_TARGET_COMPUTE_12 = CUjit_target(12)       # < Compute device class 1.2
const CU_TARGET_COMPUTE_13 = CUjit_target(13)       # < Compute device class 1.3
const CU_TARGET_COMPUTE_20 = CUjit_target(20)       # < Compute device class 2.0
const CU_TARGET_COMPUTE_21 = CUjit_target(21)       # < Compute device class 2.1
const CU_TARGET_COMPUTE_30 = CUjit_target(30)       # < Compute device class 3.0
const CU_TARGET_COMPUTE_32 = CUjit_target(32)       # < Compute device class 3.2
const CU_TARGET_COMPUTE_35 = CUjit_target(35)       # < Compute device class 3.5
const CU_TARGET_COMPUTE_37 = CUjit_target(37)       # < Compute device class 3.7
const CU_TARGET_COMPUTE_50 = CUjit_target(50)       # < Compute device class 5.0
const CU_TARGET_COMPUTE_52 = CUjit_target(52)       # < Compute device class 5.2
const CU_TARGET_COMPUTE_53 = CUjit_target(53)       # < Compute device class 5.3
const CU_TARGET_COMPUTE_60 = CUjit_target(60)       # < Compute device class 6.0
const CU_TARGET_COMPUTE_61 = CUjit_target(61)       # < Compute device class 6.1
const CU_TARGET_COMPUTE_62 = CUjit_target(62)       # < Compute device class 6.2

const CUjit_fallback = Cuint

# possible CUjit_fallback values
const CU_PREFER_PTX       = CUjit_fallback(0)
const CU_PREFER_BINARY    = CUjit_fallback(1)

const CUjit_cacheMode = Cuint

# possible CUjit_cacheMode values
const CU_JIT_CACHE_OPTION_NONE = CUjit_cacheMode(0)
const CU_JIT_CACHE_OPTION_CG   = CUjit_cacheMode(1)
const CU_JIT_CACHE_OPTION_CA   = CUjit_cacheMode(2)

const CUjitInputType = Cuint

# possible CUjitInputType values
const CU_JIT_INPUT_CUBIN      = CUjitInputType(0)
const CU_JIT_INPUT_PTX        = CUjitInputType(1)
const CU_JIT_INPUT_FATBINARY  = CUjitInputType(2)
const CU_JIT_INPUT_OBJECT     = CUjitInputType(3)
const CU_JIT_INPUT_LIBRARY    = CUjitInputType(4)
#=
In CUDA 8.0, 'CU_JIT_NUM_INPUT_TYPES' is 5 by enumerating from
'CU_JIT_INPUT_LIBRARY'.
=#
const CU_JIT_NUM_INPUT_TYPES  = CUjitInputType(5)

# CUlinkState available since CUDA 5.5
const CUlinkState = Ptr{Nothing}

const CUgraphicsRegisterFlags = Cuint

# possible CUgraphicsRegisterFlags values
const CU_GRAPHICS_REGISTER_FLAGS_NONE           = CUgraphicsRegisterFlags(0x00)
const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY      = CUgraphicsRegisterFlags(0x01)
const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD  = CUgraphicsRegisterFlags(0x02)
const CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST   = CUgraphicsRegisterFlags(0x04)
const CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = CUgraphicsRegisterFlags(0x08)

const CUgraphicsMapResourceFlags = Cuint

# possible CUgraphicsMapResourceFlags values
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = CUgraphicsMapResourceFlags(0x00)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = CUgraphicsMapResourceFlags(0x01)
const CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = CUgraphicsMapResourceFlags(0x02)

const CUarray_cubemap_face = Cuint

# possible CUarray_cubemap_face values
const CU_CUBEMAP_FACE_POSITIVE_X  = CUarray_cubemap_face(0x00)
const CU_CUBEMAP_FACE_NEGATIVE_X  = CUarray_cubemap_face(0x01)
const CU_CUBEMAP_FACE_POSITIVE_Y  = CUarray_cubemap_face(0x02)
const CU_CUBEMAP_FACE_NEGATIVE_Y  = CUarray_cubemap_face(0x03)
const CU_CUBEMAP_FACE_POSITIVE_Z  = CUarray_cubemap_face(0x04)
const CU_CUBEMAP_FACE_NEGATIVE_Z  = CUarray_cubemap_face(0x05)

const CUlimit = Cuint

# possible CUlimit values
const CU_LIMIT_STACK_SIZE                       = CUlimit(0x00)
const CU_LIMIT_PRINTF_FIFO_SIZE                 = CUlimit(0x01)
const CU_LIMIT_MALLOC_HEAP_SIZE                 = CUlimit(0x02)
const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = CUlimit(0x03)
const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = CUlimit(0x04)
#=
In CUDA 8.0, 'CU_LIMIT_MAX' is 5 by enumerating from
'CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT'.
=#
const CU_LIMIT_MAX                              = CUlimit(0x05)

const CUresourcetype = Cuint

# possible CUresourcetype values
const CU_RESOURCE_TYPE_ARRAY           = CUresourcetype(0x00)
const CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = CUresourcetype(0x01)
const CU_RESOURCE_TYPE_LINEAR          = CUresourcetype(0x02)
const CU_RESOURCE_TYPE_PITCH2D         = CUresourcetype(0x03)

const CUresult = Cuint

# possible CUresult values
const CUDA_SUCCESS                              = CUresult(0)
const CUDA_ERROR_INVALID_VALUE                  = CUresult(1)
const CUDA_ERROR_OUT_OF_MEMORY                  = CUresult(2)
const CUDA_ERROR_NOT_INITIALIZED                = CUresult(3)
const CUDA_ERROR_DEINITIALIZED                  = CUresult(4)
const CUDA_ERROR_PROFILER_DISABLED              = CUresult(5)
const CUDA_ERROR_PROFILER_NOT_INITIALIZED       = CUresult(6)
const CUDA_ERROR_PROFILER_ALREADY_STARTED       = CUresult(7)
const CUDA_ERROR_PROFILER_ALREADY_STOPPED       = CUresult(8)
const CUDA_ERROR_NO_DEVICE                      = CUresult(100)
const CUDA_ERROR_INVALID_DEVICE                 = CUresult(101)
const CUDA_ERROR_INVALID_IMAGE                  = CUresult(200)
const CUDA_ERROR_INVALID_CONTEXT                = CUresult(201)
const CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = CUresult(202)
const CUDA_ERROR_MAP_FAILED                     = CUresult(205)
const CUDA_ERROR_UNMAP_FAILED                   = CUresult(206)
const CUDA_ERROR_ARRAY_IS_MAPPED                = CUresult(207)
const CUDA_ERROR_ALREADY_MAPPED                 = CUresult(208)
const CUDA_ERROR_NO_BINARY_FOR_GPU              = CUresult(209)
const CUDA_ERROR_ALREADY_ACQUIRED               = CUresult(210)
const CUDA_ERROR_NOT_MAPPED                     = CUresult(211)
const CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = CUresult(212)
const CUDA_ERROR_NOT_MAPPED_AS_POINTER          = CUresult(213)
const CUDA_ERROR_ECC_UNCORRECTABLE              = CUresult(214)
const CUDA_ERROR_UNSUPPORTED_LIMIT              = CUresult(215)
const CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = CUresult(216)
const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = CUresult(217)
const CUDA_ERROR_INVALID_PTX                    = CUresult(218)
const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = CUresult(219)
const CUDA_ERROR_NVLINK_UNCORRECTABLE           = CUresult(220)
const CUDA_ERROR_INVALID_SOURCE                 = CUresult(300)
const CUDA_ERROR_FILE_NOT_FOUND                 = CUresult(301)
const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = CUresult(302)
const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = CUresult(303)
const CUDA_ERROR_OPERATING_SYSTEM               = CUresult(304)
const CUDA_ERROR_INVALID_HANDLE                 = CUresult(400)
const CUDA_ERROR_NOT_FOUND                      = CUresult(500)
const CUDA_ERROR_NOT_READY                      = CUresult(600)
const CUDA_ERROR_ILLEGAL_ADDRESS                = CUresult(700)
const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = CUresult(701)
const CUDA_ERROR_LAUNCH_TIMEOUT                 = CUresult(702)
const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = CUresult(703)
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    = CUresult(704)
const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        = CUresult(705)
const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = CUresult(708)
const CUDA_ERROR_CONTEXT_IS_DESTROYED           = CUresult(709)
const CUDA_ERROR_ASSERT                         = CUresult(710)
const CUDA_ERROR_TOO_MANY_PEERS                 = CUresult(711)
const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = CUresult(712)
const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = CUresult(713)
const CUDA_ERROR_HARDWARE_STACK_ERROR           = CUresult(714)
const CUDA_ERROR_ILLEGAL_INSTRUCTION            = CUresult(715)
const CUDA_ERROR_MISALIGNED_ADDRESS             = CUresult(716)
const CUDA_ERROR_INVALID_ADDRESS_SPACE          = CUresult(717)
const CUDA_ERROR_INVALID_PC                     = CUresult(718)
const CUDA_ERROR_LAUNCH_FAILED                  = CUresult(719)
const CUDA_ERROR_NOT_PERMITTED                  = CUresult(800)
const CUDA_ERROR_NOT_SUPPORTED                  = CUresult(801)
const CUDA_ERROR_UNKNOWN                        = CUresult(999)

const CUdevice_P2PAttribute = Cuint

# possible CUdevice_P2PAttribute values
const CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK              = CUdevice_P2PAttribute(0x01)
const CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED              = CUdevice_P2PAttribute(0x02)
const CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED       = CUdevice_P2PAttribute(0x03)

# type association for CUDA stream callback function
const CUstreamCallback = Ptr{Nothing}

# type association for a C-callable function that returns the dynamic shared memory
# needed by a block as Csize_t
const CUoccupancyB2DSize = Ptr{Nothing}

# possible cuMemHostAlloc() flags
const CU_MEMHOSTALLOC_PORTABLE      = Cuint(0x01)
const CU_MEMHOSTALLOC_DEVICEMAP     = Cuint(0x02)
const CU_MEMHOSTALLOC_WRITECOMBINED = Cuint(0x04)

# possible cuMemHostRegister() flags
const CU_MEMHOSTREGISTER_PORTABLE   = Cuint(0x01)
const CU_MEMHOSTREGISTER_DEVICEMAP  = Cuint(0x02)
const CU_MEMHOSTREGISTER_IOMEMORY   = Cuint(0x04)

# CUDA_MEMCPY2D available since CUDA 3.2
struct CUDA_MEMCPY2D
    srcXInBytes::Csize_t
    srcY::Csize_t

    srcMemoryType::CUmemorytype
    srcHost::Ptr{Nothing}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcPitch::Csize_t

    dstXInBytes::Csize_t
    dstY::Csize_t

    dstMemoryType::CUmemorytype
    dstHost::Ptr{Nothing}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstPitch::Csize_t

    WidthInBytes::Csize_t
    Height::Csize_t
end

Base.zero(::Type{CUDA_MEMCPY2D}) = CUDA_MEMCPY2D(Csize_t(0),
                                                Csize_t(0),
                                                CUmemorytype(0),
                                                Ptr{Nothing}(0),
                                                CUdeviceptr(0),
                                                CUarray(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                CUmemorytype(0),
                                                Ptr{Nothing}(0),
                                                CUdeviceptr(0),
                                                CUarray(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0))
Base.zero(x::CUDA_MEMCPY2D) = zero(typeof(x))

# CUDA_MEMCPY3D available since CUDA 3.2
struct CUDA_MEMCPY3D
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Nothing}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    reserved0::Ptr{Nothing}
    srcPitch::Csize_t
    srcHeight::Csize_t

    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Nothing}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    reserved1::Ptr{Nothing}
    dstPitch::Csize_t
    dstHeight::Csize_t

    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

Base.zero(::Type{CUDA_MEMCPY3D}) = CUDA_MEMCPY3D(Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                CUmemorytype(0),
                                                Ptr{Nothing}(0),
                                                CUdeviceptr(0),
                                                CUarray(0),
                                                Ptr{Nothing}(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                CUmemorytype(0),
                                                Ptr{Nothing}(0),
                                                CUdeviceptr(0),
                                                CUarray(0),
                                                Ptr{Nothing}(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0),
                                                Csize_t(0))
Base.zero(x::CUDA_MEMCPY3D) = zero(typeof(x))

# CUDA_MEMCPY3D_PEER available since CUDA 3.2
struct CUDA_MEMCPY3D_PEER
    srcXInBytes::Csize_t
    srcY::Csize_t
    srcZ::Csize_t
    srcLOD::Csize_t
    srcMemoryType::CUmemorytype
    srcHost::Ptr{Nothing}
    srcDevice::CUdeviceptr
    srcArray::CUarray
    srcContext::CUcontext
    srcPitch::Csize_t
    srcHeight::Csize_t

    dstXInBytes::Csize_t
    dstY::Csize_t
    dstZ::Csize_t
    dstLOD::Csize_t
    dstMemoryType::CUmemorytype
    dstHost::Ptr{Nothing}
    dstDevice::CUdeviceptr
    dstArray::CUarray
    dstContext::CUcontext
    dstPitch::Csize_t
    dstHeight::Csize_t

    WidthInBytes::Csize_t
    Height::Csize_t
    Depth::Csize_t
end

Base.zero(::Type{CUDA_MEMCPY3D_PEER}) = CUDA_MEMCPY3D_PEER(Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        CUmemorytype(0),
                                                        Ptr{Nothing}(0),
                                                        CUdeviceptr(0),
                                                        CUarray(0),
                                                        CUcontext(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        CUmemorytype(0),
                                                        Ptr{Nothing}(0),
                                                        CUdeviceptr(0),
                                                        CUarray(0),
                                                        CUcontext(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0),
                                                        Csize_t(0))
Base.zero(x::CUDA_MEMCPY3D_PEER) = zero(typeof(x))

# CUDA_ARRAY_DESCRIPTOR available since CUDA 3.2
struct CUDA_ARRAY_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t

    Format::CUarray_format
    NumChannels::Cuint
end

Base.zero(::Type{CUDA_ARRAY_DESCRIPTOR}) = CUDA_ARRAY_DESCRIPTOR(
    Csize_t(0),
    Csize_t(0),
    CUarray_format(0),
    Cuint(0))
Base.zero(x::CUDA_ARRAY_DESCRIPTOR) = zero(typeof(x))

# CUDA_ARRAY3D_DESCRIPTOR available since CUDA 3.2
struct CUDA_ARRAY3D_DESCRIPTOR
    Width::Csize_t
    Height::Csize_t
    Depth::Csize_t

    Format::CUarray_format
    NumChannels::Cuint
    Flags::Cuint
end

Base.zero(::Type{CUDA_ARRAY3D_DESCRIPTOR}) = CUDA_ARRAY3D_DESCRIPTOR(
    Csize_t(0),
    Csize_t(0),
    Csize_t(0),
    CUarray_format(0),
    Cuint(0),
    Cuint(0))
Base.zero(x::CUDA_ARRAY3D_DESCRIPTOR) = zero(typeof(x))

# should be 128 bytes, Sys.WORD_SIZE affects CUarray and CUdeviceptr
if (Sys.WORD_SIZE == 64)
    struct CUDA_RESOURCE_DESC_res_array
        hArray::CUarray
        pad::NTuple{30, Cuint}
    end
    Base.zero(::Type{CUDA_RESOURCE_DESC_res_array}) = CUDA_RESOURCE_DESC_res_array(
        CUarray(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct CUDA_RESOURCE_DESC_res_array
        hArray::CUarray
        pad::NTuple{31, Cuint}
    end
    Base.zero(::Type{CUDA_RESOURCE_DESC_res_array}) = CUDA_RESOURCE_DESC_res_array(
        CUarray(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::CUDA_RESOURCE_DESC_res_array) = zero(typeof(x))

# should be 128 bytes, Sys.WORD_SIZE affects CUmipmappedArray
if (Sys.WORD_SIZE == 64)
    struct CUDA_RESOURCE_DESC_res_mipmap
        hMipmappedArray::CUmipmappedArray
        pad::NTuple{30, Cuint}
    end
    Base.zero(::Type{CUDA_RESOURCE_DESC_res_mipmap}) = CUDA_RESOURCE_DESC_res_mipmap(
        CUmipmappedArray(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct CUDA_RESOURCE_DESC_res_mipmap
        hMipmappedArray::CUmipmappedArray
        pad::NTuple{31, Cuint}
    end
    Base.zero(::Type{CUDA_RESOURCE_DESC_res_mipmap}) = CUDA_RESOURCE_DESC_res_mipmap(
        CUmipmappedArray(0),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end

Base.zero(x::CUDA_RESOURCE_DESC_res_mipmap) = zero(typeof(x))

# should be 128 bytes, Sys.WORD_SIZE affects CUdeviceptr
struct CUDA_RESOURCE_DESC_res_linear
    devPtr::CUdeviceptr
    format::CUarray_format
    numChannels::Cuint
    sizeInBytes::Csize_t
    pad::NTuple{26, Cuint}
end

Base.zero(::Type{CUDA_RESOURCE_DESC_res_linear}) = CUDA_RESOURCE_DESC_res_linear(
    CUdeviceptr(0),
    CUarray_format(0),
    Cuint(0),
    Csize_t(0),
    (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
Base.zero(x::CUDA_RESOURCE_DESC_res_linear) = zero(typeof(x))

# should be 128 bytes, Sys.WORD_SIZE affects CUdeviceptr
struct CUDA_RESOURCE_DESC_res_pitch2D
    devPtr::CUdeviceptr
    format::CUarray_format
    numChannels::Cuint
    width::Csize_t
    height::Csize_t
    pitchInBytes::Csize_t
    pad::NTuple{22, Cuint}
end

Base.zero(::Type{CUDA_RESOURCE_DESC_res_pitch2D}) = CUDA_RESOURCE_DESC_res_pitch2D(
    CUdeviceptr(0), CUarray_format(0), Cuint(0), Csize_t(0), Csize_t(0), Csize_t(0), (0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000))
Base.zero(x::CUDA_RESOURCE_DESC_res_pitch2D) = zero(typeof(x))

# should be 128 bytes
struct CUDA_RESOURCE_DESC_res_reserved
    reserved::NTuple{32, Cint}
end

Base.zero(::Type{CUDA_RESOURCE_DESC_res_reserved}) = CUDA_RESOURCE_DESC_res_reserved(
    (Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0)))
Base.zero(x::CUDA_RESOURCE_DESC_res_reserved) = zero(typeof(x))

const CUDA_RESOURCE_DESC_res = Union{CUDA_RESOURCE_DESC_res_array,
                                    CUDA_RESOURCE_DESC_res_mipmap,
                                    CUDA_RESOURCE_DESC_res_linear,
                                    CUDA_RESOURCE_DESC_res_pitch2D,
                                    CUDA_RESOURCE_DESC_res_reserved}

@assert((sizeof(CUDA_RESOURCE_DESC_res_array) == 128) &&
        (sizeof(CUDA_RESOURCE_DESC_res_mipmap) == 128) &&
        (sizeof(CUDA_RESOURCE_DESC_res_linear) == 128) &&
        (sizeof(CUDA_RESOURCE_DESC_res_pitch2D) == 128) &&
        (sizeof(CUDA_RESOURCE_DESC_res_reserved) == 128))

Base.zero(::Type{CUDA_RESOURCE_DESC_res}) = CUDA_RESOURCE_DESC_res_reserved(
    (Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0)))
Base.zero(x::CUDA_RESOURCE_DESC_res) = zero(typeof(x))

# type casting between possible CUDA_RESOURCE_DESC_res data types
CUDA_RESOURCE_DESC_res_array(crdrm::CUDA_RESOURCE_DESC_res_mipmap) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_array}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_mipmap}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_mipmap}, crdrm))))
CUDA_RESOURCE_DESC_res_array(crdrl::CUDA_RESOURCE_DESC_res_linear) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_array}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_linear}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_linear}, crdrl))))
CUDA_RESOURCE_DESC_res_array(crdrp::CUDA_RESOURCE_DESC_res_pitch2D) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_array}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_pitch2D}, crdrp))))
CUDA_RESOURCE_DESC_res_array(crdrr::CUDA_RESOURCE_DESC_res_reserved) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_array}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_reserved}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_reserved}, crdrr))))

CUDA_RESOURCE_DESC_res_mipmap(crdra::CUDA_RESOURCE_DESC_res_array) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_mipmap}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_array}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_array}, crdra))))
CUDA_RESOURCE_DESC_res_mipmap(crdrl::CUDA_RESOURCE_DESC_res_linear) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_mipmap}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_linear}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_linear}, crdrl))))
CUDA_RESOURCE_DESC_res_mipmap(crdrp::CUDA_RESOURCE_DESC_res_pitch2D) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_mipmap}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_pitch2D}, crdrp))))
CUDA_RESOURCE_DESC_res_mipmap(crdrr::CUDA_RESOURCE_DESC_res_reserved) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_mipmap}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_reserved}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_reserved}, crdrr))))

CUDA_RESOURCE_DESC_res_linear(crdra::CUDA_RESOURCE_DESC_res_array) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_linear}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_array}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_array}, crdra))))
CUDA_RESOURCE_DESC_res_linear(crdrm::CUDA_RESOURCE_DESC_res_mipmap) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_linear}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_mipmap}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_mipmap}, crdrm))))
CUDA_RESOURCE_DESC_res_linear(crdrp::CUDA_RESOURCE_DESC_res_pitch2D) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_linear}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_pitch2D}, crdrp))))
CUDA_RESOURCE_DESC_res_linear(crdrr::CUDA_RESOURCE_DESC_res_reserved) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_linear}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_reserved}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_reserved}, crdrr))))

CUDA_RESOURCE_DESC_res_pitch2D(crdra::CUDA_RESOURCE_DESC_res_array) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_array}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_array}, crdra))))
CUDA_RESOURCE_DESC_res_pitch2D(crdrm::CUDA_RESOURCE_DESC_res_mipmap) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_mipmap}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_mipmap}, crdrm))))
CUDA_RESOURCE_DESC_res_pitch2D(crdrl::CUDA_RESOURCE_DESC_res_linear) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_linear}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_linear}, crdrl))))
CUDA_RESOURCE_DESC_res_pitch2D(crdrr::CUDA_RESOURCE_DESC_res_reserved) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_reserved}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_reserved}, crdrr))))

CUDA_RESOURCE_DESC_res_reserved(crdra::CUDA_RESOURCE_DESC_res_array) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_reserved}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_array}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_array}, crdra))))
CUDA_RESOURCE_DESC_res_reserved(crdrm::CUDA_RESOURCE_DESC_res_mipmap) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_reserved}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_mipmap}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_mipmap}, crdrm))))
CUDA_RESOURCE_DESC_res_reserved(crdrl::CUDA_RESOURCE_DESC_res_linear) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_reserved}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_linear}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_linear}, crdrl))))
CUDA_RESOURCE_DESC_res_reserved(crdrp::CUDA_RESOURCE_DESC_res_pitch2D) = Base.unsafe_load(Ptr{CUDA_RESOURCE_DESC_res_reserved}(Base.unsafe_convert(Ptr{CUDA_RESOURCE_DESC_res_pitch2D}, Base.cconvert(Ref{CUDA_RESOURCE_DESC_res_pitch2D}, crdrp))))

# CUDA_RESOURCE_DESC available since CUDA 5.0
struct CUDA_RESOURCE_DESC
    resType::CUresourcetype
    res::CUDA_RESOURCE_DESC_res
    flags::Cuint
end

Base.zero(::Type{CUDA_RESOURCE_DESC}) = CUDA_RESOURCE_DESC(CUresourcetype(0),
                                        zero(CUDA_RESOURCE_DESC_res),
                                        Cuint(0))
Base.zero(x::CUDA_RESOURCE_DESC) = zero(typeof(x))

# CUDA_TEXTURE_DESC available since CUDA 5.0
struct CUDA_TEXTURE_DESC
    addressMode::NTuple{3, CUaddress_mode}
    filterMode::CUfilter_mode
    flags::Cuint
    maxAnisotropy::Cuint
    mipmapFilterMode::CUfilter_mode
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    borderColor::NTuple{4, Cfloat}
    reserved::NTuple{12, Cint}
end

Base.zero(::Type{CUDA_TEXTURE_DESC}) = CUDA_TEXTURE_DESC(
                        (CUaddress_mode(0), CUaddress_mode(0), CUaddress_mode(0)),
                        CUfilter_mode(0),
                        Cuint(0),
                        Cuint(0),
                        CUfilter_mode(0),
                        Cfloat(0),
                        Cfloat(0),
                        Cfloat(0),
                        (Cfloat(0), Cfloat(0), Cfloat(0), Cfloat(0)),
                        (Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0)))
Base.zero(x::CUDA_TEXTURE_DESC) = zero(typeof(x))

# CUresourceViewFormat available since CUDA 5.0
const CUresourceViewFormat = Cuint

# possible CUresourceViewFormat values
const CU_RES_VIEW_FORMAT_NONE          = CUresourceViewFormat(0x00)
const CU_RES_VIEW_FORMAT_UINT_1X8      = CUresourceViewFormat(0x01)
const CU_RES_VIEW_FORMAT_UINT_2X8      = CUresourceViewFormat(0x02)
const CU_RES_VIEW_FORMAT_UINT_4X8      = CUresourceViewFormat(0x03)
const CU_RES_VIEW_FORMAT_SINT_1X8      = CUresourceViewFormat(0x04)
const CU_RES_VIEW_FORMAT_SINT_2X8      = CUresourceViewFormat(0x05)
const CU_RES_VIEW_FORMAT_SINT_4X8      = CUresourceViewFormat(0x06)
const CU_RES_VIEW_FORMAT_UINT_1X16     = CUresourceViewFormat(0x07)
const CU_RES_VIEW_FORMAT_UINT_2X16     = CUresourceViewFormat(0x08)
const CU_RES_VIEW_FORMAT_UINT_4X16     = CUresourceViewFormat(0x09)
const CU_RES_VIEW_FORMAT_SINT_1X16     = CUresourceViewFormat(0x0a)
const CU_RES_VIEW_FORMAT_SINT_2X16     = CUresourceViewFormat(0x0b)
const CU_RES_VIEW_FORMAT_SINT_4X16     = CUresourceViewFormat(0x0c)
const CU_RES_VIEW_FORMAT_UINT_1X32     = CUresourceViewFormat(0x0d)
const CU_RES_VIEW_FORMAT_UINT_2X32     = CUresourceViewFormat(0x0e)
const CU_RES_VIEW_FORMAT_UINT_4X32     = CUresourceViewFormat(0x0f)
const CU_RES_VIEW_FORMAT_SINT_1X32     = CUresourceViewFormat(0x10)
const CU_RES_VIEW_FORMAT_SINT_2X32     = CUresourceViewFormat(0x11)
const CU_RES_VIEW_FORMAT_SINT_4X32     = CUresourceViewFormat(0x12)
const CU_RES_VIEW_FORMAT_FLOAT_1X16    = CUresourceViewFormat(0x13)
const CU_RES_VIEW_FORMAT_FLOAT_2X16    = CUresourceViewFormat(0x14)
const CU_RES_VIEW_FORMAT_FLOAT_4X16    = CUresourceViewFormat(0x15)
const CU_RES_VIEW_FORMAT_FLOAT_1X32    = CUresourceViewFormat(0x16)
const CU_RES_VIEW_FORMAT_FLOAT_2X32    = CUresourceViewFormat(0x17)
const CU_RES_VIEW_FORMAT_FLOAT_4X32    = CUresourceViewFormat(0x18)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC1  = CUresourceViewFormat(0x19)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC2  = CUresourceViewFormat(0x1a)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC3  = CUresourceViewFormat(0x1b)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC4  = CUresourceViewFormat(0x1c)
const CU_RES_VIEW_FORMAT_SIGNED_BC4    = CUresourceViewFormat(0x1d)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC5  = CUresourceViewFormat(0x1e)
const CU_RES_VIEW_FORMAT_SIGNED_BC5    = CUresourceViewFormat(0x1f)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = CUresourceViewFormat(0x20)
const CU_RES_VIEW_FORMAT_SIGNED_BC6H   = CUresourceViewFormat(0x21)
const CU_RES_VIEW_FORMAT_UNSIGNED_BC7  = CUresourceViewFormat(0x22)

# CUDA_RESOURCE_VIEW_DESC available since CUDA 5.0
struct CUDA_RESOURCE_VIEW_DESC
    format::CUresourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::Cuint
    lastMipmapLevel::Cuint
    firstLayer::Cuint
    lastLayer::Cuint
    reserved::NTuple{16, Cuint}
end

Base.zero(::Type{CUDA_RESOURCE_VIEW_DESC}) = CUDA_RESOURCE_VIEW_DESC(
    CUresourceViewFormat(0),
    Csize_t(0),
    Csize_t(0),
    Csize_t(0),
    Cuint(0),
    Cuint(0),
    Cuint(0),
    Cuint(0),
    (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
Base.zero(x::CUDA_RESOURCE_VIEW_DESC) = zero(typeof(x))

# CUDA_POINTER_ATTRIBUTE_P2P_TOKENS available since CUDA 5.0
struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
    p2pToken::Culonglong
    vaSpaceToken::Cuint
end

Base.zero(::Type{CUDA_POINTER_ATTRIBUTE_P2P_TOKENS}) = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS(
    Culonglong(0),
    Cuint(0))
Base.zero(x::CUDA_POINTER_ATTRIBUTE_P2P_TOKENS) = zero(typeof(x))

const CUDA_ARRAY3D_LAYERED        = Cuint(0x01)
const CUDA_ARRAY3D_2DARRAY        = Cuint(0x01)
const CUDA_ARRAY3D_SURFACE_LDST   = Cuint(0x02)
const CUDA_ARRAY3D_CUBEMAP        = Cuint(0x04)
const CUDA_ARRAY3D_TEXTURE_GATHER = Cuint(0x08)
const CUDA_ARRAY3D_DEPTH_TEXTURE  = Cuint(0x10)

# possible cuTexRefSetArray() flags
const CU_TRSA_OVERRIDE_FORMAT = Cuint(0x01)

# possible cuTexRefSetFlags() flags
const CU_TRSF_READ_AS_INTEGER        = Cuint(0x01)
const CU_TRSF_NORMALIZED_COORDINATES = Cuint(0x02)
const CU_TRSF_SRGB                   = Cuint(0x10)

# indicators used by cuLaunchKernel() in the 'extra' parameter
const CU_LAUNCH_PARAM_END            = Ptr{Nothing}(UInt(0x00))
const CU_LAUNCH_PARAM_BUFFER_POINTER = Ptr{Nothing}(UInt(0x01))
const CU_LAUNCH_PARAM_BUFFER_SIZE    = Ptr{Nothing}(UInt(0x02))

# possible values for 'texunit' parameter in cuParamSetTexRef()
const CU_PARAM_TR_DEFAULT = Cint(-1)

const CU_DEVICE_CPU     = CUdevice(-1)
const CU_DEVICE_INVALID = CUdevice(-2)
