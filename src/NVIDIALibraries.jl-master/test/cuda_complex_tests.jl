#=*
* CUDA Complex arithmetic function tests
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

using NVIDIALibraries.ComplexTypes

# cuConjf()
let x = cuComplex(111.1f0, 111.1f0)
    @test (pop!(collect(reinterpret(Complex{Float32}, [cuConjf(x)]))) == conj(Complex{Float32}(111.1f0, 111.1f0)))
end

# cuConj()
let x = cuDoubleComplex(111.1, 111.1)
    @test (pop!(collect(reinterpret(Complex{Float64}, [cuConj(x)]))) == conj(Complex{Float64}(111.1, 111.1)))
end

# cuCaddf()
let x = cuComplex(111.1f0, 111.1f0), y = cuComplex(111.1f0, 111.1f0)
    @test (pop!(collect(reinterpret(Complex{Float32}, [cuCaddf(x, y)]))) == (Complex{Float32}(111.1f0, 111.1f0) + Complex{Float32}(111.1f0, 111.1f0)))
end

# cuCadd()
let x = cuDoubleComplex(111.1, 111.1), y = cuDoubleComplex(111.1, 111.1)
    @test (pop!(collect(reinterpret(Complex{Float64}, [cuCadd(x, y)]))) == (Complex{Float64}(111.1, 111.1) + Complex{Float64}(111.1, 111.1)))
end

# cuCsubf()
let x = cuComplex(111.1f0, 111.1f0), y = cuComplex(11.1f0, 11.1f0)
    @test (pop!(collect(reinterpret(Complex{Float32}, [cuCsubf(x, y)]))) == (Complex{Float32}(111.1f0, 111.1f0) - Complex{Float32}(11.1f0, 11.1f0)))
end

# cuCsub()
let x = cuDoubleComplex(111.1, 111.1), y = cuDoubleComplex(11.1, 11.1)
    @test (pop!(collect(reinterpret(Complex{Float64}, [cuCsub(x, y)]))) == (Complex{Float64}(111.1, 111.1) - Complex{Float64}(11.1, 11.1)))
end

# cuCmulf()
let x = cuComplex(111.1f0, 111.1f0), y = cuComplex(111.1f0, 111.1f0)
    @test (pop!(collect(reinterpret(Complex{Float32}, [cuCmulf(x, y)]))) == (Complex{Float32}(111.1f0, 111.1f0) * Complex{Float32}(111.1f0, 111.1f0)))
end

# cuCmul()
let x = cuDoubleComplex(111.1, 111.1), y = cuDoubleComplex(111.1, 111.1)
    @test (pop!(collect(reinterpret(Complex{Float64}, [cuCmul(x, y)]))) == (Complex{Float64}(111.1, 111.1) * Complex{Float64}(111.1, 111.1)))
end

# cuCdivf()
let x = cuComplex(111.1f0, 111.1f0), y = cuComplex(11.1f0, 11.1f0)
    @test (pop!(collect(reinterpret(Complex{Float32}, [cuCdivf(x, y)]))) == (Complex{Float32}(111.1f0, 111.1f0) / Complex{Float32}(11.1f0, 11.1f0)))
end

# cuCdiv()
let x = cuDoubleComplex(111.1, 111.1), y = cuDoubleComplex(11.1, 11.1)
    @test (pop!(collect(reinterpret(Complex{Float64}, [cuCdiv(x, y)]))) == (Complex{Float64}(111.1, 111.1) / Complex{Float64}(11.1, 11.1)))
end

# cuCabsf()
let x = cuComplex(111.1f0, 111.1f0)
    @test (cuCabsf(x) == abs(Complex{Float32}(111.1f0, 111.1f0)))
end

# cuCabs()
let x = cuDoubleComplex(111.1, 111.1)
    @test (cuCabs(x) == abs(Complex{Float64}(111.1, 111.1)))
end

# fma() is not defined for Complex{Float32} or Complex{Float64} in julia 1.0

