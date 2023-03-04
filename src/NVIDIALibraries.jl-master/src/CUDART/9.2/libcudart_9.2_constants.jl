#=*
* CUDA runtime API v9.2 definitions
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

const cudaStream_t = CUstream

const cudaEvent_t = CUevent

const cudaGraphicsResource_t = CUgraphicsResource

const cudaUUID_t = CUuuid

const cudaHostAllocDefault          = Cuint(0x00)
const cudaHostAllocPortable         = Cuint(0x01)
const cudaHostAllocMapped           = Cuint(0x02)
const cudaHostAllocWriteCombined    = Cuint(0x04)

const cudaHostRegisterDefault   = Cuint(0x00)
const cudaHostRegisterPortable  = Cuint(0x01)
const cudaHostRegisterMapped    = Cuint(0x02)
const cudaHostRegisterIoMemory  = Cuint(0x04)

const cudaPeerAccessDefault = Cuint(0x00)

const cudaStreamDefault     = Cuint(0x00)
const cudaStreamNonBlocking = Cuint(0x01)

const cudaStreamLegacy      = cudaStream_t(UInt(0x1))
const cudaStreamPerThread   = cudaStream_t(UInt(0x2))

const cudaEventDefault          = Cuint(0x00)
const cudaEventBlockingSync     = Cuint(0x01)
const cudaEventDisableTiming    = Cuint(0x02)
const cudaEventInterprocess     = Cuint(0x04)

const cudaDeviceScheduleAuto            = Cuint(0x00)
const cudaDeviceScheduleSpin            = Cuint(0x01)
const cudaDeviceScheduleYield           = Cuint(0x02)
const cudaDeviceScheduleBlockingSync    = Cuint(0x04)
const cudaDeviceBlockingSync            = Cuint(0x04)
const cudaDeviceScheduleMask            = Cuint(0x07)
const cudaDeviceMapHost                 = Cuint(0x08)
const cudaDeviceLmemResizeToMax         = Cuint(0x10)
const cudaDeviceMask                    = Cuint(0x1f)

const cudaArrayDefault          = Cuint(0x00)
const cudaArrayLayered          = Cuint(0x01)
const cudaArraySurfaceLoadStore = Cuint(0x02)
const cudaArrayCubemap          = Cuint(0x04)
const cudaArrayTextureGather    = Cuint(0x08)

const cudaIpcMemLazyEnablePeerAccess    = Cuint(0x01)

const cudaMemAttachGlobal   = Cuint(0x01)
const cudaMemAttachHost     = Cuint(0x02)
const cudaMemAttachSingle   = Cuint(0x04)

const cudaOccupancyDefault                  = Cuint(0x00)
const cudaOccupancyDisableCachingOverride   = Cuint(0x01)

const cudaCpuDeviceId       = Cint(-1)
const cudaInvalidDeviceId   = Cint(-2)

const cudaCooperativeLaunchMultiDeviceNoPreSync     = Cuint(0x01)
const cudaCooperativeLaunchMultiDeviceNoPostSync    = Cuint(0x02)

const cudaError_t = Cuint

# possible cudaError_t values
const cudaSuccess                           = cudaError_t(0)
const cudaErrorMissingConfiguration         = cudaError_t(1)
const cudaErrorMemoryAllocation             = cudaError_t(2)
const cudaErrorInitializationError          = cudaError_t(3)
const cudaErrorLaunchFailure                = cudaError_t(4)
const cudaErrorPriorLaunchFailure           = cudaError_t(5)
const cudaErrorLaunchTimeout                = cudaError_t(6)
const cudaErrorLaunchOutOfResources         = cudaError_t(7)
const cudaErrorInvalidDeviceFunction        = cudaError_t(8)
const cudaErrorInvalidConfiguration         = cudaError_t(9)
const cudaErrorInvalidDevice                = cudaError_t(10)
const cudaErrorInvalidValue                 = cudaError_t(11)
const cudaErrorInvalidPitchValue            = cudaError_t(12)
const cudaErrorInvalidSymbol                = cudaError_t(13)
const cudaErrorMapBufferObjectFailed        = cudaError_t(14)
const cudaErrorUnmapBufferObjectFailed      = cudaError_t(15)
const cudaErrorInvalidHostPointer           = cudaError_t(16)
const cudaErrorInvalidDevicePointer         = cudaError_t(17)
const cudaErrorInvalidTexture               = cudaError_t(18)
const cudaErrorInvalidTextureBinding        = cudaError_t(19)
const cudaErrorInvalidChannelDescriptor     = cudaError_t(20)
const cudaErrorInvalidMemcpyDirection       = cudaError_t(21)
const cudaErrorAddressOfConstant            = cudaError_t(22)
const cudaErrorTextureFetchFailed           = cudaError_t(23)
const cudaErrorTextureNotBound              = cudaError_t(24)
const cudaErrorSynchronizationError         = cudaError_t(25)
const cudaErrorInvalidFilterSetting         = cudaError_t(26)
const cudaErrorInvalidNormSetting           = cudaError_t(27)
const cudaErrorMixedDeviceExecution         = cudaError_t(28)
const cudaErrorCudartUnloading              = cudaError_t(29)
const cudaErrorUnknown                      = cudaError_t(30)
const cudaErrorNotYetImplemented            = cudaError_t(31)
const cudaErrorMemoryValueTooLarge          = cudaError_t(32)
const cudaErrorInvalidResourceHandle        = cudaError_t(33)
const cudaErrorNotReady                     = cudaError_t(34)
const cudaErrorInsufficientDriver           = cudaError_t(35)
const cudaErrorSetOnActiveProcess           = cudaError_t(36)
const cudaErrorInvalidSurface               = cudaError_t(37)
const cudaErrorNoDevice                     = cudaError_t(38)
const cudaErrorECCUncorrectable             = cudaError_t(39)
const cudaErrorSharedObjectSymbolNotFound   = cudaError_t(40)
const cudaErrorSharedObjectInitFailed       = cudaError_t(41)
const cudaErrorUnsupportedLimit             = cudaError_t(42)
const cudaErrorDuplicateVariableName        = cudaError_t(43)
const cudaErrorDuplicateTextureName         = cudaError_t(44)
const cudaErrorDuplicateSurfaceName         = cudaError_t(45)
const cudaErrorDevicesUnavailable           = cudaError_t(46)
const cudaErrorInvalidKernelImage           = cudaError_t(47)
const cudaErrorNoKernelImageForDevice       = cudaError_t(48)
const cudaErrorIncompatibleDriverContext    = cudaError_t(49)
const cudaErrorPeerAccessAlreadyEnabled     = cudaError_t(50)
const cudaErrorPeerAccessNotEnabled         = cudaError_t(51)
const cudaErrorDeviceAlreadyInUse           = cudaError_t(54)
const cudaErrorProfilerDisabled             = cudaError_t(55)
const cudaErrorProfilerNotInitialized       = cudaError_t(56)
const cudaErrorProfilerAlreadyStarted       = cudaError_t(57)
const cudaErrorProfilerAlreadyStopped       = cudaError_t(58)
const cudaErrorAssert                       = cudaError_t(59)
const cudaErrorTooManyPeers                 = cudaError_t(60)
const cudaErrorHostMemoryAlreadyRegistered  = cudaError_t(61)
const cudaErrorHostMemoryNotRegistered      = cudaError_t(62)
const cudaErrorOperatingSystem              = cudaError_t(63)
const cudaErrorPeerAccessUnsupported        = cudaError_t(64)
const cudaErrorLaunchMaxDepthExceeded       = cudaError_t(65)
const cudaErrorLaunchFileScopedTex          = cudaError_t(66)
const cudaErrorLaunchFileScopedSurf         = cudaError_t(67)
const cudaErrorSyncDepthExceeded            = cudaError_t(68)
const cudaErrorLaunchPendingCountExceeded   = cudaError_t(69)
const cudaErrorNotPermitted                 = cudaError_t(70)
const cudaErrorNotSupported                 = cudaError_t(71)
const cudaErrorHardwareStackError           = cudaError_t(72)
const cudaErrorIllegalInstruction           = cudaError_t(73)
const cudaErrorMisalignedAddress            = cudaError_t(74)
const cudaErrorInvalidAddressSpace          = cudaError_t(75)
const cudaErrorInvalidPc                    = cudaError_t(76)
const cudaErrorIllegalAddress               = cudaError_t(77)
const cudaErrorInvalidPtx                   = cudaError_t(78)
const cudaErrorInvalidGraphicsContext       = cudaError_t(79)
const cudaErrorNvlinkUncorrectable          = cudaError_t(80)
const cudaErrorJitCompilerNotFound          = cudaError_t(81)
const cudaErrorCooperativeLaunchTooLarge    = cudaError_t(82)
const cudaErrorStartupFailure               = cudaError_t(0x7f)
const cudaErrorApiFailureBase               = cudaError_t(10000)

const cudaChannelFormatKind = Cuint

# possible cudaChannelFormatKind values
const cudaChannelFormatKindSigned   = cudaChannelFormatKind(0)
const cudaChannelFormatKindUnsigned = cudaChannelFormatKind(1)
const cudaChannelFormatKindFloat    = cudaChannelFormatKind(2)
const cudaChannelFormatKindNone     = cudaChannelFormatKind(3)

struct cudaChannelFormatDesc
    x::Cint
    y::Cint
    z::Cint
    w::Cint
    f::cudaChannelFormatKind
end

Base.zero(::Type{cudaChannelFormatDesc}) = cudaChannelFormatDesc(Cint(0), Cint(0), Cint(0), Cint(0), cudaChannelFormatKind(0))
Base.zero(x::cudaChannelFormatDesc) = zero(typeof(x))

const cudaArray_t = CUarray
const cudaArray_const_t = CUarray

const cudaMipmappedArray_t = CUmipmappedArray
const cudaMipmappedArray_const_t = CUmipmappedArray

const cudaMemoryType = Cuint

# possible cudaMemoryType values
const cudaMemoryTypeHost   = cudaMemoryType(1)
const cudaMemoryTypeDevice = cudaMemoryType(2)

const cudaMemcpyKind = Cuint

# possible cudaMemcpyKind values
const cudaMemcpyHostToHost      = cudaMemcpyKind(0)
const cudaMemcpyHostToDevice    = cudaMemcpyKind(1)
const cudaMemcpyDeviceToHost    = cudaMemcpyKind(2)
const cudaMemcpyDeviceToDevice  = cudaMemcpyKind(3)
const cudaMemcpyDefault         = cudaMemcpyKind(4)

struct cudaPitchedPtr
    ptr::Ptr{Nothing}
    pitch::Csize_t
    xsize::Csize_t
    ysize::Csize_t
end

Base.zero(::Type{cudaPitchedPtr}) = cudaPitchedPtr(C_NULL, Csize_t(0), Csize_t(0), Csize_t(0))
Base.zero(x::cudaPitchedPtr) = zero(typeof(x))

struct cudaExtent
    width::Csize_t
    height::Csize_t
    depth::Csize_t
end

Base.zero(::Type{cudaExtent}) = cudaExtent(Csize_t(0), Csize_t(0), Csize_t(0))
Base.zero(x::cudaExtent) = zero(typeof(x))

struct cudaPos
    x::Csize_t
    y::Csize_t
    z::Csize_t
end

Base.zero(::Type{cudaPos}) = cudaPos(Csize_t(0), Csize_t(0), Csize_t(0))
Base.zero(x::cudaPos) = zero(typeof(x))

struct cudaMemcpy3DParms
    srcArray::cudaArray_t
    srcPos::cudaPos
    srcPtr::cudaPitchedPtr

    dstArray::cudaArray_t
    dstPos::cudaPos
    dstPtr::cudaPitchedPtr

    extent::cudaExtent
    kind::cudaMemcpyKind
end

Base.zero(::Type{cudaMemcpy3DParms}) = cudaMemcpy3DParms(
    cudaArray_t(C_NULL),
    zero(cudaPos),
    zero(cudaPitchedPtr),
    cudaArray_t(C_NULL),
    zero(cudaPos),
    zero(cudaPitchedPtr),
    zero(cudaExtent),
    cudaMemcpyKind(0))
Base.zero(x::cudaMemcpy3DParms) = zero(typeof(x))

struct cudaMemcpy3DPeerParms
    srcArray::cudaArray_t
    srcPos::cudaPos
    srcPtr::cudaPitchedPtr
    srcDevice::Cint

    dstArray::cudaArray_t
    dstPos::cudaPos
    dstPtr::cudaPitchedPtr
    dstDevice::Cint

    extent::cudaExtent
end

Base.zero(::Type{cudaMemcpy3DPeerParms}) = cudaMemcpy3DPeerParms(
    cudaArray_t(C_NULL),
    zero(cudaPos),
    zero(cudaPitchedPtr),
    Cint(0),
    cudaArray_t(C_NULL),
    zero(cudaPos),
    zero(cudaPitchedPtr),
    Cint(0),
    zero(cudaExtent))
Base.zero(x::cudaMemcpy3DPeerParms) = zero(typeof(x))

const cudaGraphicsRegisterFlags = Cuint

# possible cudaGraphicsRegisterFlags values
const cudaGraphicsRegisterFlagsNone             = cudaGraphicsRegisterFlags(0)
const cudaGraphicsRegisterFlagsReadOnly         = cudaGraphicsRegisterFlags(1)
const cudaGraphicsRegisterFlagsWriteDiscard     = cudaGraphicsRegisterFlags(2)
const cudaGraphicsRegisterFlagsSurfaceLoadStore = cudaGraphicsRegisterFlags(4)
const cudaGraphicsRegisterFlagsTextureGather    = cudaGraphicsRegisterFlags(8)

const cudaGraphicsMapFlags = Cuint

# possible cudaGraphicsMapFlags values
const cudaGraphicsMapFlagsNone          = cudaGraphicsMapFlags(0)
const cudaGraphicsMapFlagsReadOnly      = cudaGraphicsMapFlags(1)
const cudaGraphicsMapFlagsWriteDiscard  = cudaGraphicsMapFlags(2)

const cudaGraphicsCubeFace = Cuint

# possible cudaGraphicsCubeFace values
const cudaGraphicsCubeFacePositiveX = cudaGraphicsCubeFace(0x00)
const cudaGraphicsCubeFaceNegativeX = cudaGraphicsCubeFace(0x01)
const cudaGraphicsCubeFacePositiveY = cudaGraphicsCubeFace(0x02)
const cudaGraphicsCubeFaceNegativeY = cudaGraphicsCubeFace(0x03)
const cudaGraphicsCubeFacePositiveZ = cudaGraphicsCubeFace(0x04)
const cudaGraphicsCubeFaceNegativeZ = cudaGraphicsCubeFace(0x05)

const cudaResourceType = Cuint

# possible cudaResourceType values
const cudaResourceTypeArray             = cudaResourceType(0x00)
const cudaResourceTypeMipmappedArray    = cudaResourceType(0x01)
const cudaResourceTypeLinear            = cudaResourceType(0x02)
const cudaResourceTypePitch2D           = cudaResourceType(0x03)

const cudaResourceViewFormat = Cuint

# possible cudaResourceViewFormat values
const cudaResViewFormatNone                         = cudaResourceViewFormat(0x00)
const cudaResViewFormatUnsignedChar1                = cudaResourceViewFormat(0x01)
const cudaResViewFormatUnsignedChar2                = cudaResourceViewFormat(0x02)
const cudaResViewFormatUnsignedChar4                = cudaResourceViewFormat(0x03)
const cudaResViewFormatSignedChar1                  = cudaResourceViewFormat(0x04)
const cudaResViewFormatSignedChar2                  = cudaResourceViewFormat(0x05)
const cudaResViewFormatSignedChar4                  = cudaResourceViewFormat(0x06)
const cudaResViewFormatUnsignedShort1               = cudaResourceViewFormat(0x07)
const cudaResViewFormatUnsignedShort2               = cudaResourceViewFormat(0x08)
const cudaResViewFormatUnsignedShort4               = cudaResourceViewFormat(0x09)
const cudaResViewFormatSignedShort1                 = cudaResourceViewFormat(0x0a)
const cudaResViewFormatSignedShort2                 = cudaResourceViewFormat(0x0b)
const cudaResViewFormatSignedShort4                 = cudaResourceViewFormat(0x0c)
const cudaResViewFormatUnsignedInt1                 = cudaResourceViewFormat(0x0d)
const cudaResViewFormatUnsignedInt2                 = cudaResourceViewFormat(0x0e)
const cudaResViewFormatUnsignedInt4                 = cudaResourceViewFormat(0x0f)
const cudaResViewFormatSignedInt1                   = cudaResourceViewFormat(0x10)
const cudaResViewFormatSignedInt2                   = cudaResourceViewFormat(0x11)
const cudaResViewFormatSignedInt4                   = cudaResourceViewFormat(0x12)
const cudaResViewFormatHalf1                        = cudaResourceViewFormat(0x13)
const cudaResViewFormatHalf2                        = cudaResourceViewFormat(0x14)
const cudaResViewFormatHalf4                        = cudaResourceViewFormat(0x15)
const cudaResViewFormatFloat1                       = cudaResourceViewFormat(0x16)
const cudaResViewFormatFloat2                       = cudaResourceViewFormat(0x17)
const cudaResViewFormatFloat4                       = cudaResourceViewFormat(0x18)
const cudaResViewFormatUnsignedBlockCompressed1     = cudaResourceViewFormat(0x19)
const cudaResViewFormatUnsignedBlockCompressed2     = cudaResourceViewFormat(0x1a)
const cudaResViewFormatUnsignedBlockCompressed3     = cudaResourceViewFormat(0x1b)
const cudaResViewFormatUnsignedBlockCompressed4     = cudaResourceViewFormat(0x1c)
const cudaResViewFormatSignedBlockCompressed4       = cudaResourceViewFormat(0x1d)
const cudaResViewFormatUnsignedBlockCompressed5     = cudaResourceViewFormat(0x1e)
const cudaResViewFormatSignedBlockCompressed5       = cudaResourceViewFormat(0x1f)
const cudaResViewFormatUnsignedBlockCompressed6H    = cudaResourceViewFormat(0x20)
const cudaResViewFormatSignedBlockCompressed6H      = cudaResourceViewFormat(0x21)
const cudaResViewFormatUnsignedBlockCompressed7     = cudaResourceViewFormat(0x22)

# 64-bit should be 56 bytes and 32-bit should be 48 bytes
if (Sys.WORD_SIZE == 64)
    struct cudaResourceDesc_res_array
        array::cudaArray_t
        pad::NTuple{12, Cuint}
    end
    Base.zero(::Type{cudaResourceDesc_res_array}) = cudaResourceDesc_res_array(
        cudaArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct cudaResourceDesc_res_array
        array::cudaArray_t
        pad::NTuple{11, Cuint}
    end
    Base.zero(::Type{cudaResourceDesc_res_array}) = cudaResourceDesc_res_array(
        cudaArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::cudaResourceDesc_res_array) = zero(typeof(x))

# 64-bit should be 56 bytes and 32-bit should be 48 bytes
if (Sys.WORD_SIZE == 64)
    struct cudaResourceDesc_res_mipmap
        array::cudaMipmappedArray_t
        pad::NTuple{12, Cuint}
    end
    Base.zero(::Type{cudaResourceDesc_res_mipmap}) = cudaResourceDesc_res_mipmap(
        cudaMipmappedArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
elseif (Sys.WORD_SIZE == 32)
    struct cudaResourceDesc_res_mipmap
        array::cudaMipmappedArray_t
        pad::NTuple{11, Cuint}
    end
    Base.zero(::Type{cudaResourceDesc_res_mipmap}) = cudaResourceDesc_res_mipmap(
        cudaMipmappedArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::cudaResourceDesc_res_mipmap) = zero(typeof(x))

# 64-bit should be 56 bytes and 32-bit should be 48 bytes
struct cudaResourceDesc_res_linear
    devPtr::Ptr{Nothing}
    desc::cudaChannelFormatDesc
    sizeInBytes::Csize_t
    pad::NTuple{4, Cuint}
end
Base.zero(::Type{cudaResourceDesc_res_linear}) = cudaResourceDesc_res_linear(
    C_NULL,
    zero(cudaChannelFormatDesc),
    Csize_t(0),
    (Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
Base.zero(x::cudaResourceDesc_res_linear) = zero(typeof(x))

# 64-bit should be 56 bytes and 32-bit should be 48 bytes
struct cudaResourceDesc_res_pitch2D
    devPtr::Ptr{Nothing}
    desc::cudaChannelFormatDesc
    width::Csize_t
    height::Csize_t
    pitchInBytes::Csize_t
end

Base.zero(::Type{cudaResourceDesc_res_pitch2D}) = cudaResourceDesc_res_pitch2D(
    C_NULL,
    zero(cudaChannelFormatDesc),
    Csize_t(0),
    Csize_t(0),
    Csize_t(0))
Base.zero(x::cudaResourceDesc_res_pitch2D) = zero(typeof(x))

const cudaResourceDesc_res = Union{cudaResourceDesc_res_array,
                                    cudaResourceDesc_res_mipmap,
                                    cudaResourceDesc_res_linear,
                                    cudaResourceDesc_res_pitch2D}

# 64-bit should be 56 bytes and 32-bit should be 48 bytes
if (Sys.WORD_SIZE == 64)
    @assert((sizeof(cudaResourceDesc_res_array) == 56) &&
            (sizeof(cudaResourceDesc_res_mipmap) == 56) &&
            (sizeof(cudaResourceDesc_res_linear) == 56) &&
            (sizeof(cudaResourceDesc_res_pitch2D) == 56))
    Base.zero(::Type{cudaResourceDesc_res}) = cudaResourceDesc_res_array(
        cudaArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))

elseif (Sys.WORD_SIZE == 32)
    @assert((sizeof(cudaResourceDesc_res_array) == 48) &&
            (sizeof(cudaResourceDesc_res_mipmap) == 48) &&
            (sizeof(cudaResourceDesc_res_linear) == 48) &&
            (sizeof(cudaResourceDesc_res_pitch2D) == 48))
    Base.zero(::Type{cudaResourceDesc_res}) = cudaResourceDesc_res_array(
        cudaArray_t(C_NULL),
        (Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0), Cuint(0)))
end
Base.zero(x::cudaResourceDesc_res) = zero(typeof(x))

# type casting between possible cudaResourceDesc_res data types
cudaResourceDesc_res_array(crdrm::cudaResourceDesc_res_mipmap) = Base.unsafe_load(Ptr{cudaResourceDesc_res_array}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_mipmap}, Base.cconvert(Ref{cudaResourceDesc_res_mipmap}, crdrm))))
cudaResourceDesc_res_array(crdrl::cudaResourceDesc_res_linear) = Base.unsafe_load(Ptr{cudaResourceDesc_res_array}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_linear}, Base.cconvert(Ref{cudaResourceDesc_res_linear}, crdrl))))
cudaResourceDesc_res_array(crdrp::cudaResourceDesc_res_pitch2D) = Base.unsafe_load(Ptr{cudaResourceDesc_res_array}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_pitch2D}, Base.cconvert(Ref{cudaResourceDesc_res_pitch2D}, crdrp))))

cudaResourceDesc_res_mipmap(crdra::cudaResourceDesc_res_array) = Base.unsafe_load(Ptr{cudaResourceDesc_res_mipmap}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_array}, Base.cconvert(Ref{cudaResourceDesc_res_array}, crdra))))
cudaResourceDesc_res_mipmap(crdrl::cudaResourceDesc_res_linear) = Base.unsafe_load(Ptr{cudaResourceDesc_res_mipmap}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_linear}, Base.cconvert(Ref{cudaResourceDesc_res_linear}, crdrl))))
cudaResourceDesc_res_mipmap(crdrp::cudaResourceDesc_res_pitch2D) = Base.unsafe_load(Ptr{cudaResourceDesc_res_mipmap}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_pitch2D}, Base.cconvert(Ref{cudaResourceDesc_res_pitch2D}, crdrp))))

cudaResourceDesc_res_linear(crdra::cudaResourceDesc_res_array) = Base.unsafe_load(Ptr{cudaResourceDesc_res_linear}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_array}, Base.cconvert(Ref{cudaResourceDesc_res_array}, crdra))))
cudaResourceDesc_res_linear(crdrm::cudaResourceDesc_res_mipmap) = Base.unsafe_load(Ptr{cudaResourceDesc_res_linear}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_mipmap}, Base.cconvert(Ref{cudaResourceDesc_res_mipmap}, crdrm))))
cudaResourceDesc_res_linear(crdrp::cudaResourceDesc_res_pitch2D) = Base.unsafe_load(Ptr{cudaResourceDesc_res_linear}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_pitch2D}, Base.cconvert(Ref{cudaResourceDesc_res_pitch2D}, crdrp))))

cudaResourceDesc_res_pitch2D(crdra::cudaResourceDesc_res_array) = Base.unsafe_load(Ptr{cudaResourceDesc_res_pitch2D}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_array}, Base.cconvert(Ref{cudaResourceDesc_res_array}, crdra))))
cudaResourceDesc_res_pitch2D(crdrm::cudaResourceDesc_res_mipmap) = Base.unsafe_load(Ptr{cudaResourceDesc_res_pitch2D}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_mipmap}, Base.cconvert(Ref{cudaResourceDesc_res_mipmap}, crdrm))))
cudaResourceDesc_res_pitch2D(crdrl::cudaResourceDesc_res_linear) = Base.unsafe_load(Ptr{cudaResourceDesc_res_pitch2D}(Base.unsafe_convert(Ptr{cudaResourceDesc_res_linear}, Base.cconvert(Ref{cudaResourceDesc_res_linear}, crdrl))))

struct cudaResourceDesc
    resType::Cuint
    res::cudaResourceDesc_res
end

Base.zero(::Type{cudaResourceDesc}) = cudaResourceDesc(
    cudaResourceType(0),
    zero(cudaResourceDesc_res))
Base.zero(x::cudaResourceDesc) = zero(typeof(x))

struct cudaResourceViewDesc
    format::cudaResourceViewFormat
    width::Csize_t
    height::Csize_t
    depth::Csize_t
    firstMipmapLevel::Cuint
    lastMipmapLevel::Cuint
    firstLayer::Cuint
    lastLayer::Cuint
end

Base.zero(::Type{cudaResourceViewDesc}) = cudaResourceViewDesc(
    cudaResourceViewFormat(0),
    Csize_t(0),
    Csize_t(0),
    Csize_t(0),
    Cuint(0),
    Cuint(0),
    Cuint(0),
    Cuint(0))
Base.zero(x::cudaResourceViewDesc) = zero(typeof(x))

struct cudaPointerAttributes
    memoryType::cudaMemoryType
    device::Cint
    devicePointer::Ptr{Nothing}
    hostPointer::Ptr{Nothing}
    isManaged::Cint
end

Base.zero(::Type{cudaPointerAttributes}) = cudaPointerAttributes(
    cudaMemoryType(0),
    Cint(0),
    C_NULL,
    C_NULL,
    Cint(0))
Base.zero(x::cudaPointerAttributes) = zero(typeof(x))

struct cudaFuncAttributes
    sharedSizeBytes::Csize_t
    constSizeBytes::Csize_t
    localSizeBytes::Csize_t
    maxThreadsPerBlock::Cint
    numRegs::Cint
    ptxVersion::Cint
    binaryVersion::Cint
    cacheModeCA::Cint
    maxDynamicSharedSizeBytes::Cint
    preferredShmemCarveout::Cint
end

Base.zero(::Type{cudaFuncAttributes}) = cudaFuncAttributes(
    Csize_t(0),
    Csize_t(0),
    Csize_t(0),
    Cint(0),
    Cint(0),
    Cint(0),
    Cint(0),
    Cint(0),
    Cint(0),
    Cint(0))
Base.zero(x::cudaFuncAttributes) = zero(typeof(x))

const cudaFuncAttribute = Cuint

# possible cudaFuncAttribute values
const cudaFuncAttributeMaxDynamicSharedMemorySize       = cudaFuncAttribute(8)
const cudaFuncAttributePreferredSharedMemoryCarveout    = cudaFuncAttribute(9)
#=
In CUDA 9.2, 'cudaFuncAttributeMax' is 10 by enumerating from
'cudaFuncAttributePreferredSharedMemoryCarveout'.
=#
const cudaFuncAttributeMax                              = cudaFuncAttribute(10)

const cudaFuncCache = Cuint

# possible cudaFuncCache values
const cudaFuncCachePreferNone   = cudaFuncCache(0)
const cudaFuncCachePreferShared = cudaFuncCache(1)
const cudaFuncCachePreferL1     = cudaFuncCache(2)
const cudaFuncCachePreferEqual  = cudaFuncCache(3)

const cudaSharedMemConfig = Cuint

# possible cudaSharedMemConfig values
const cudaSharedMemBankSizeDefault      = cudaSharedMemConfig(0)
const cudaSharedMemBankSizeFourByte     = cudaSharedMemConfig(1)
const cudaSharedMemBankSizeEightByte    = cudaSharedMemConfig(2)

# cudaSharedCarveout can be negative
const cudaSharedCarveout = Cint

# possible cudaSharedCarveout values
const cudaSharedmemCarveoutDefault      = cudaSharedCarveout(-1)
const cudaSharedmemCarveoutMaxShared    = cudaSharedCarveout(100)
const cudaSharedmemCarveoutMaxL1        = cudaSharedCarveout(0)

const cudaComputeMode = Cuint

# possible cudaComputeMode values
const cudaComputeModeDefault            = cudaComputeMode(0)
const cudaComputeModeExclusive          = cudaComputeMode(1)
const cudaComputeModeProhibited         = cudaComputeMode(2)
const cudaComputeModeExclusiveProcess   = cudaComputeMode(3)

const cudaLimit = Cuint

# possible cudaLimit values
const cudaLimitStackSize                    = cudaLimit(0x00)
const cudaLimitPrintfFifoSize               = cudaLimit(0x01)
const cudaLimitMallocHeapSize               = cudaLimit(0x02)
const cudaLimitDevRuntimeSyncDepth          = cudaLimit(0x03)
const cudaLimitDevRuntimePendingLaunchCount = cudaLimit(0x04)

const cudaMemoryAdvise = Cuint

# possible cudaMemoryAdvise values
const cudaMemAdviseSetReadMostly            = cudaMemoryAdvise(1)
const cudaMemAdviseUnsetReadMostly          = cudaMemoryAdvise(2)
const cudaMemAdviseSetPreferredLocation     = cudaMemoryAdvise(3)
const cudaMemAdviseUnsetPreferredLocation   = cudaMemoryAdvise(4)
const cudaMemAdviseSetAccessedBy            = cudaMemoryAdvise(5)
const cudaMemAdviseUnsetAccessedBy          = cudaMemoryAdvise(6)

const cudaMemRangeAttribute = Cuint

# possible cudaMemRangeAttribute values
const cudaMemRangeAttributeReadMostly           = cudaMemRangeAttribute(1)
const cudaMemRangeAttributePreferredLocation    = cudaMemRangeAttribute(2)
const cudaMemRangeAttributeAccessedBy           = cudaMemRangeAttribute(3)
const cudaMemRangeAttributeLastPrefetchLocation = cudaMemRangeAttribute(4)

const cudaOutputMode_t = Cuint

# possible cudaOutputMode values
const cudaKeyValuePair  = cudaOutputMode_t(0x00)
const cudaCSV           = cudaOutputMode_t(0x01)

const cudaDeviceAttr = Cuint

# possible cudaDeviceAttr values
const cudaDevAttrMaxThreadsPerBlock                     = cudaDeviceAttr(1)
const cudaDevAttrMaxBlockDimX                           = cudaDeviceAttr(2)
const cudaDevAttrMaxBlockDimY                           = cudaDeviceAttr(3)
const cudaDevAttrMaxBlockDimZ                           = cudaDeviceAttr(4)
const cudaDevAttrMaxGridDimX                            = cudaDeviceAttr(5)
const cudaDevAttrMaxGridDimY                            = cudaDeviceAttr(6)
const cudaDevAttrMaxGridDimZ                            = cudaDeviceAttr(7)
const cudaDevAttrMaxSharedMemoryPerBlock                = cudaDeviceAttr(8)
const cudaDevAttrTotalConstantMemory                    = cudaDeviceAttr(9)
const cudaDevAttrWarpSize                               = cudaDeviceAttr(10)
const cudaDevAttrMaxPitch                               = cudaDeviceAttr(11)
const cudaDevAttrMaxRegistersPerBlock                   = cudaDeviceAttr(12)
const cudaDevAttrClockRate                              = cudaDeviceAttr(13)
const cudaDevAttrTextureAlignment                       = cudaDeviceAttr(14)
const cudaDevAttrGpuOverlap                             = cudaDeviceAttr(15)
const cudaDevAttrMultiProcessorCount                    = cudaDeviceAttr(16)
const cudaDevAttrKernelExecTimeout                      = cudaDeviceAttr(17)
const cudaDevAttrIntegrated                             = cudaDeviceAttr(18)
const cudaDevAttrCanMapHostMemory                       = cudaDeviceAttr(19)
const cudaDevAttrComputeMode                            = cudaDeviceAttr(20)
const cudaDevAttrMaxTexture1DWidth                      = cudaDeviceAttr(21)
const cudaDevAttrMaxTexture2DWidth                      = cudaDeviceAttr(22)
const cudaDevAttrMaxTexture2DHeight                     = cudaDeviceAttr(23)
const cudaDevAttrMaxTexture3DWidth                      = cudaDeviceAttr(24)
const cudaDevAttrMaxTexture3DHeight                     = cudaDeviceAttr(25)
const cudaDevAttrMaxTexture3DDepth                      = cudaDeviceAttr(26)
const cudaDevAttrMaxTexture2DLayeredWidth               = cudaDeviceAttr(27)
const cudaDevAttrMaxTexture2DLayeredHeight              = cudaDeviceAttr(28)
const cudaDevAttrMaxTexture2DLayeredLayers              = cudaDeviceAttr(29)
const cudaDevAttrSurfaceAlignment                       = cudaDeviceAttr(30)
const cudaDevAttrConcurrentKernels                      = cudaDeviceAttr(31)
const cudaDevAttrEccEnabled                             = cudaDeviceAttr(32)
const cudaDevAttrPciBusId                               = cudaDeviceAttr(33)
const cudaDevAttrPciDeviceId                            = cudaDeviceAttr(34)
const cudaDevAttrTccDriver                              = cudaDeviceAttr(35)
const cudaDevAttrMemoryClockRate                        = cudaDeviceAttr(36)
const cudaDevAttrGlobalMemoryBusWidth                   = cudaDeviceAttr(37)
const cudaDevAttrL2CacheSize                            = cudaDeviceAttr(38)
const cudaDevAttrMaxThreadsPerMultiProcessor            = cudaDeviceAttr(39)
const cudaDevAttrAsyncEngineCount                       = cudaDeviceAttr(40)
const cudaDevAttrUnifiedAddressing                      = cudaDeviceAttr(41)
const cudaDevAttrMaxTexture1DLayeredWidth               = cudaDeviceAttr(42)
const cudaDevAttrMaxTexture1DLayeredLayers              = cudaDeviceAttr(43)
const cudaDevAttrMaxTexture2DGatherWidth                = cudaDeviceAttr(45)
const cudaDevAttrMaxTexture2DGatherHeight               = cudaDeviceAttr(46)
const cudaDevAttrMaxTexture3DWidthAlt                   = cudaDeviceAttr(47)
const cudaDevAttrMaxTexture3DHeightAlt                  = cudaDeviceAttr(48)
const cudaDevAttrMaxTexture3DDepthAlt                   = cudaDeviceAttr(49)
const cudaDevAttrPciDomainId                            = cudaDeviceAttr(50)
const cudaDevAttrTexturePitchAlignment                  = cudaDeviceAttr(51)
const cudaDevAttrMaxTextureCubemapWidth                 = cudaDeviceAttr(52)
const cudaDevAttrMaxTextureCubemapLayeredWidth          = cudaDeviceAttr(53)
const cudaDevAttrMaxTextureCubemapLayeredLayers         = cudaDeviceAttr(54)
const cudaDevAttrMaxSurface1DWidth                      = cudaDeviceAttr(55)
const cudaDevAttrMaxSurface2DWidth                      = cudaDeviceAttr(56)
const cudaDevAttrMaxSurface2DHeight                     = cudaDeviceAttr(57)
const cudaDevAttrMaxSurface3DWidth                      = cudaDeviceAttr(58)
const cudaDevAttrMaxSurface3DHeight                     = cudaDeviceAttr(59)
const cudaDevAttrMaxSurface3DDepth                      = cudaDeviceAttr(60)
const cudaDevAttrMaxSurface1DLayeredWidth               = cudaDeviceAttr(61)
const cudaDevAttrMaxSurface1DLayeredLayers              = cudaDeviceAttr(62)
const cudaDevAttrMaxSurface2DLayeredWidth               = cudaDeviceAttr(63)
const cudaDevAttrMaxSurface2DLayeredHeight              = cudaDeviceAttr(64)
const cudaDevAttrMaxSurface2DLayeredLayers              = cudaDeviceAttr(65)
const cudaDevAttrMaxSurfaceCubemapWidth                 = cudaDeviceAttr(66)
const cudaDevAttrMaxSurfaceCubemapLayeredWidth          = cudaDeviceAttr(67)
const cudaDevAttrMaxSurfaceCubemapLayeredLayers         = cudaDeviceAttr(68)
const cudaDevAttrMaxTexture1DLinearWidth                = cudaDeviceAttr(69)
const cudaDevAttrMaxTexture2DLinearWidth                = cudaDeviceAttr(70)
const cudaDevAttrMaxTexture2DLinearHeight               = cudaDeviceAttr(71)
const cudaDevAttrMaxTexture2DLinearPitch                = cudaDeviceAttr(72)
const cudaDevAttrMaxTexture2DMipmappedWidth             = cudaDeviceAttr(73)
const cudaDevAttrMaxTexture2DMipmappedHeight            = cudaDeviceAttr(74)
const cudaDevAttrComputeCapabilityMajor                 = cudaDeviceAttr(75)
const cudaDevAttrComputeCapabilityMinor                 = cudaDeviceAttr(76)
const cudaDevAttrMaxTexture1DMipmappedWidth             = cudaDeviceAttr(77)
const cudaDevAttrStreamPrioritiesSupported              = cudaDeviceAttr(78)
const cudaDevAttrGlobalL1CacheSupported                 = cudaDeviceAttr(79)
const cudaDevAttrLocalL1CacheSupported                  = cudaDeviceAttr(80)
const cudaDevAttrMaxSharedMemoryPerMultiprocessor       = cudaDeviceAttr(81)
const cudaDevAttrMaxRegistersPerMultiprocessor          = cudaDeviceAttr(82)
const cudaDevAttrManagedMemory                          = cudaDeviceAttr(83)
const cudaDevAttrIsMultiGpuBoard                        = cudaDeviceAttr(84)
const cudaDevAttrMultiGpuBoardGroupID                   = cudaDeviceAttr(85)
const cudaDevAttrHostNativeAtomicSupported              = cudaDeviceAttr(86)
const cudaDevAttrSingleToDoublePrecisionPerfRatio       = cudaDeviceAttr(87)
const cudaDevAttrPageableMemoryAccess                   = cudaDeviceAttr(88)
const cudaDevAttrConcurrentManagedAccess                = cudaDeviceAttr(89)
const cudaDevAttrComputePreemptionSupported             = cudaDeviceAttr(90)
const cudaDevAttrCanUseHostPointerForRegisteredMem      = cudaDeviceAttr(91)
const cudaDevAttrReserved92                             = cudaDeviceAttr(92)
const cudaDevAttrReserved93                             = cudaDeviceAttr(93)
const cudaDevAttrReserved94                             = cudaDeviceAttr(94)
const cudaDevAttrCooperativeLaunch                      = cudaDeviceAttr(95)
const cudaDevAttrCooperativeMultiDeviceLaunch           = cudaDeviceAttr(96)
const cudaDevAttrMaxSharedMemoryPerBlockOptin           = cudaDeviceAttr(97)
const cudaDevAttrCanFlushRemoteWrites                   = cudaDeviceAttr(98)
const cudaDevAttrHostRegisterSupported                  = cudaDeviceAttr(99)
const cudaDevAttrPageableMemoryAccessUsesHostPageTables = cudaDeviceAttr(100)
const cudaDevAttrDirectManagedMemAccessFromHost         = cudaDeviceAttr(101)

const cudaDeviceP2PAttr = Cuint

# possible cudaDeviceP2PAttr values
const cudaDevP2PAttrPerformanceRank             = cudaDeviceP2PAttr(1)
const cudaDevP2PAttrAccessSupported             = cudaDeviceP2PAttr(2)
const cudaDevP2PAttrNativeAtomicSupported       = cudaDeviceP2PAttr(3)
const cudaDevP2PAttrCudaArrayAccessSupported    = cudaDeviceP2PAttr(4)

struct cudaDeviceProp
    name::NTuple{256, UInt8}
    totalGlobalMem::Csize_t
    sharedMemPerBlock::Csize_t
    regsPerBlock::Cint
    warpSize::Cint
    memPitch::Csize_t
    maxThreadsPerBlock::Cint
    maxThreadsDim::NTuple{3, Cint}
    maxGridSize::NTuple{3, Cint}
    clockRate::Cint
    totalConstMem::Csize_t
    major::Cint
    minor::Cint
    textureAlignment::Csize_t
    texturePitchAlignment::Csize_t
    deviceOverlap::Cint
    multiProcessorCount::Cint
    kernelExecTimeoutEnabled::Cint
    integrated::Cint
    canMapHostMemory::Cint
    computeMode::Cint
    maxTexture1D::Cint
    maxTexture1DMipmap::Cint
    maxTexture1DLinear::Cint
    maxTexture2D::NTuple{2, Cint}
    maxTexture2DMipmap::NTuple{2, Cint}
    maxTexture2DLinear::NTuple{3, Cint}
    maxTexture2DGather::NTuple{2, Cint}
    maxTexture3D::NTuple{3, Cint}
    maxTexture3DAlt::NTuple{3, Cint}
    maxTextureCubemap::Cint
    maxTexture1DLayered::NTuple{2, Cint}
    maxTexture2DLayered::NTuple{3, Cint}
    maxTextureCubemapLayered::NTuple{2, Cint}
    maxSurface1D::Cint
    maxSurface2D::NTuple{2, Cint}
    maxSurface3D::NTuple{3, Cint}
    maxSurface1DLayered::NTuple{2, Cint}
    maxSurface2DLayered::NTuple{3, Cint}
    maxSurfaceCubemap::Cint
    maxSurfaceCubemapLayered::NTuple{2, Cint}
    surfaceAlignment::Csize_t
    concurrentKernels::Cint
    ECCEnabled::Cint
    pciBusID::Cint
    pciDeviceID::Cint
    pciDomainID::Cint
    tccDriver::Cint
    asyncEngineCount::Cint
    unifiedAddressing::Cint
    memoryClockRate::Cint
    memoryBusWidth::Cint
    l2CacheSize::Cint
    maxThreadsPerMultiProcessor::Cint
    streamPrioritiesSupported::Cint
    globalL1CacheSupported::Cint
    localL1CacheSupported::Cint
    sharedMemPerMultiprocessor::Csize_t
    regsPerMultiprocessor::Cint
    managedMemory::Cint
    isMultiGpuBoard::Cint
    multiGpuBoardGroupID::Cint
    hostNativeAtomicSupported::Cint
    singleToDoublePrecisionPerfRatio::Cint
    pageableMemoryAccess::Cint
    concurrentManagedAccess::Cint
    computePreemptionSupported::Cint
    canUseHostPointerForRegisteredMem::Cint
    cooperativeLaunch::Cint
    cooperativeMultiDeviceLaunch::Cint
    sharedMemPerBlockOptin::Csize_t
    pageableMemoryAccessUsesHostPageTables::Cint
    directManagedMemAccessFromHost::Cint
end

Base.zero(::Type{cudaDeviceProp}) = cudaDeviceProp(
    (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),    #= name::NTuple{256, UInt8} =#
    Csize_t(0),                     #= totalGlobalMem::Csize_t                      =#
    Csize_t(0),                     #= sharedMemPerBlock::Csize_t                   =#
    Cint(0),                        #= regsPerBlock::Cint                           =#
    Cint(0),                        #= warpSize::Cint                               =#
    Csize_t(0),                     #= memPitch::Csize_t                            =#
    Cint(0),                        #= maxThreadsPerBlock::Cint                     =#
    (Cint(0), Cint(0), Cint(0)),    #= maxThreadsDim::NTuple{3, Cint}               =#
    (Cint(0), Cint(0), Cint(0)),    #= maxGridSize::NTuple{3, Cint}                 =#
    Cint(0),                        #= clockRate::Cint                              =#
    Csize_t(0),                     #= totalConstMem::Csize_t                       =#
    Cint(-1),                       #= major::Cint                                  =#
    Cint(-1),                       #= minor::Cint                                  =#
    Csize_t(0),                     #= textureAlignment::Csize_t                    =#
    Csize_t(0),                     #= texturePitchAlignment::Csize_t               =#
    Cint(-1),                       #= deviceOverlap::Cint                          =#
    Cint(0),                        #= multiProcessorCount::Cint                    =#
    Cint(0),                        #= kernelExecTimeoutEnabled::Cint               =#
    Cint(0),                        #= integrated::Cint                             =#
    Cint(0),                        #= canMapHostMemory::Cint                       =#
    Cint(0),                        #= computeMode::Cint                            =#
    Cint(0),                        #= maxTexture1D::Cint                           =#
    Cint(0),                        #= maxTexture1DMipmap::Cint                     =#
    Cint(0),                        #= maxTexture1DLinear::Cint                     =#
    (Cint(0), Cint(0)),             #= maxTexture2D::NTuple{2, Cint}                =#
    (Cint(0), Cint(0)),             #= maxTexture2DMipmap::NTuple{2, Cint}          =#
    (Cint(0), Cint(0), Cint(0)),    #= maxTexture2DLinear::NTuple{3, Cint}          =#
    (Cint(0), Cint(0)),             #= maxTexture2DGather::NTuple{2, Cint}          =#
    (Cint(0), Cint(0), Cint(0)),    #= maxTexture3D::NTuple{3, Cint}                =#
    (Cint(0), Cint(0), Cint(0)),    #= maxTexture3DAlt::NTuple{3, Cint}             =#
    Cint(0),                        #= maxTextureCubemap::Cint                      =#
    (Cint(0), Cint(0)),             #= maxTexture1DLayered::NTuple{2, Cint}         =#
    (Cint(0), Cint(0), Cint(0)),    #= maxTexture2DLayered::NTuple{3, Cint}         =#
    (Cint(0), Cint(0)),             #= maxTextureCubemapLayered::NTuple{2, Cint}    =#
    Cint(0),                        #= maxSurface1D::Cint                           =#
    (Cint(0), Cint(0)),             #= maxSurface2D::NTuple{2, Cint}                =#
    (Cint(0), Cint(0), Cint(0)),    #= maxSurface3D::NTuple{3, Cint}                =#
    (Cint(0), Cint(0)),             #= maxSurface1DLayered::NTuple{2, Cint}         =#
    (Cint(0), Cint(0), Cint(0)),    #= maxSurface2DLayered::NTuple{3, Cint}         =#
    Cint(0),                        #= maxSurfaceCubemap::Cint                      =#
    (Cint(0), Cint(0)),             #= maxSurfaceCubemapLayered::NTuple{2, Cint}    =#
    Csize_t(0),                     #= surfaceAlignment::Csize_t                    =#
    Cint(0),                        #= concurrentKernels::Cint                      =#
    Cint(0),                        #= ECCEnabled::Cint                             =#
    Cint(0),                        #= pciBusID::Cint                               =#
    Cint(0),                        #= pciDeviceID::Cint                            =#
    Cint(0),                        #= pciDomainID::Cint                            =#
    Cint(0),                        #= tccDriver::Cint                              =#
    Cint(0),                        #= asyncEngineCount::Cint                       =#
    Cint(0),                        #= unifiedAddressing::Cint                      =#
    Cint(0),                        #= memoryClockRate::Cint                        =#
    Cint(0),                        #= memoryBusWidth::Cint                         =#
    Cint(0),                        #= l2CacheSize::Cint                            =#
    Cint(0),                        #= maxThreadsPerMultiProcessor::Cint            =#
    Cint(0),                        #= streamPrioritiesSupported::Cint              =#
    Cint(0),                        #= globalL1CacheSupported::Cint                 =#
    Cint(0),                        #= localL1CacheSupported::Cint                  =#
    Csize_t(0),                     #= sharedMemPerMultiprocessor::Csize_t          =#
    Cint(0),                        #= regsPerMultiprocessor::Cint                  =#
    Cint(0),                        #= managedMemory::Cint                          =#
    Cint(0),                        #= isMultiGpuBoard::Cint                        =#
    Cint(0),                        #= multiGpuBoardGroupID::Cint                   =#
    Cint(0),                        #= hostNativeAtomicSupported::Cint              =#
    Cint(0),                        #= singleToDoublePrecisionPerfRatio::Cint       =#
    Cint(0),                        #= pageableMemoryAccess::Cint                   =#
    Cint(0),                        #= concurrentManagedAccess::Cint                =#
    Cint(0),                        #= computePreemptionSupported::Cint             =#
    Cint(0),                        #= canUseHostPointerForRegisteredMem::Cint      =#
    Cint(0),                        #= cooperativeLaunch::Cint                      =#
    Cint(0),                        #= cooperativeMultiDeviceLaunch::Cint           =#
    Csize_t(0),                     #= sharedMemPerBlockOptin::Csize_t              =#
    Cint(0),                        #= pageableMemoryAccessUsesHostPageTables::Cint =#
    Cint(0))                        #= directManagedMemAccessFromHost::Cint         =#
Base.zero(x::cudaDeviceProp) = zero(typeof(x))

const cudaDevicePropDontCare = zero(cudaDeviceProp)

# both CUDA_IPC_HANDLE_SIZE and CU_IPC_HANDLE_SIZE are defined as 64
const CUDA_IPC_HANDLE_SIZE = CU_IPC_HANDLE_SIZE

const cudaIpcEventHandle_t = CUipcEventHandle

const cudaIpcMemHandle_t = CUipcMemHandle

const cudaCGScope = Cuint

# possible cudaCGScope values
const cudaCGScopeInvalid   = cudaCGScope(0)
const cudaCGScopeGrid      = cudaCGScope(1)
const cudaCGScopeMultiGrid = cudaCGScope(2)

struct cudaLaunchParams
    func::Ptr{Nothing}
    gridDim::dim3
    blockDim::dim3
    args::Ptr{Ptr{Nothing}}
    sharedMem::Csize_t
    stream::cudaStream_t
end

Base.zero(::Type{cudaLaunchParams}) = cudaLaunchParams(
    C_NULL,
    zero(dim3),
    zero(dim3),
    Ptr{Ptr{Nothing}}(C_NULL),
    Csize_t(0),
    cudaStream_t(C_NULL))
Base.zero(x::cudaLaunchParams) = zero(typeof(x))

# type association for CUDA runtime stream callback function
const cudaStreamCallback_t = Ptr{Nothing}

# CUDA runtime surface data types from 'cuda_surface_types.h'

const cudaSurfaceType1D             = 0x01
const cudaSurfaceType2D             = 0x02
const cudaSurfaceType3D             = 0x03
const cudaSurfaceTypeCubemap        = 0x0C
const cudaSurfaceType1DLayered      = 0xF1
const cudaSurfaceType2DLayered      = 0xF2
const cudaSurfaceTypeCubemapLayered = 0xFC

const cudaSurfaceBoundaryMode = Cuint

# possible cudaSurfaceBoundaryMode values
const cudaBoundaryModeZero  = cudaSurfaceBoundaryMode(0)
const cudaBoundaryModeClamp = cudaSurfaceBoundaryMode(1)
const cudaBoundaryModeTrap  = cudaSurfaceBoundaryMode(2)

const cudaSurfaceFormatMode = Cuint

# possible cudaSurfaceFormatMode values
const cudaFormatModeForced  = cudaSurfaceFormatMode(0)
const cudaFormatModeAuto    = cudaSurfaceFormatMode(1)

struct surfaceReference
    channelDesc::cudaChannelFormatDesc
end
Base.zero(::Type{surfaceReference}) = surfaceReference(zero(cudaChannelFormatDesc))
Base.zero(x::surfaceReference) = zero(typeof(x))

const cudaSurfaceObject_t = Culonglong

# CUDA runtime texture data types from 'cuda_texture_types.h'

const cudaTextureType1D             = 0x01
const cudaTextureType2D             = 0x02
const cudaTextureType3D             = 0x03
const cudaTextureTypeCubemap        = 0x0C
const cudaTextureType1DLayered      = 0xF1
const cudaTextureType2DLayered      = 0xF2
const cudaTextureTypeCubemapLayered = 0xFC

const cudaTextureAddressMode = Cuint

# possible cudaTextureAddressMode values
const cudaAddressModeWrap   = cudaTextureAddressMode(0)
const cudaAddressModeClamp  = cudaTextureAddressMode(1)
const cudaAddressModeMirror = cudaTextureAddressMode(2)
const cudaAddressModeBorder = cudaTextureAddressMode(3)

const cudaTextureFilterMode = Cuint

# possible cudaTextureFilterMode values
const cudaFilterModePoint   = cudaTextureFilterMode(0)
const cudaFilterModeLinear  = cudaTextureFilterMode(1)

const cudaTextureReadMode = Cuint

# possible cudaTextureReadMode values
const cudaReadModeElementType       = cudaTextureReadMode(0)
const cudaReadModeNormalizedFloat   = cudaTextureReadMode(1)

struct textureReference
    normalized::Cint
    filterMode::Cuint
    addressMode::NTuple{3, Cuint}
    channelDesc::cudaChannelFormatDesc
    sRGB::Cint
    maxAnisotropy::Cuint
    mipmapFilterMode::Cuint
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
    __cudaReserved::NTuple{15, Cint}
end
Base.zero(::Type{textureReference}) = textureReference(
    Cint(0),
    Cuint(0),
    (Cuint(0), Cuint(0), Cuint(0)),
    zero(cudaChannelFormatDesc),
    Cint(0),
    Cuint(0),
    Cuint(0),
    Cfloat(0),
    Cfloat(0),
    Cfloat(0),
    (Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0), Cint(0)))
Base.zero(x::textureReference) = zero(typeof(x))

struct cudaTextureDesc
    addressMode::NTuple{3, Cuint}
    filterMode::Cuint
    readMode::Cuint
    sRGB::Cint
    borderColor::NTuple{4, Cfloat}
    normalizedCoords::Cint
    maxAnisotropy::Cuint
    mipmapFilterMode::Cuint
    mipmapLevelBias::Cfloat
    minMipmapLevelClamp::Cfloat
    maxMipmapLevelClamp::Cfloat
end
Base.zero(::Type{cudaTextureDesc}) = cudaTextureDesc(
    (Cuint(0), Cuint(0), Cuint(0)),
    Cuint(0),
    Cuint(0),
    Cint(0),
    (Cfloat(0), Cfloat(0), Cfloat(0), Cfloat(0)),
    Cint(0),
    Cuint(0),
    Cuint(0),
    Cfloat(0),
    Cfloat(0),
    Cfloat(0))
Base.zero(x::cudaTextureDesc) = zero(typeof(x))

const cudaTextureObject_t = Culonglong





