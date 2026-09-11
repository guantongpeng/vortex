// Copyright © 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIP_VORTEX_VECTOR_TYPES_H
#define HIP_VORTEX_VECTOR_TYPES_H

// dim3 and the HIP vector types for the Vortex device compilation.
//
// dim3 mirrors vx_spawn.h's dim3_t layout (x/y/z or m[3]) but is a plain
// struct: dim3_t is a union, and unions cannot be base classes.

#include <stdint.h>

#ifndef __HIP_VORTEX_EXEC_SPACE
#define __HIP_VORTEX_EXEC_SPACE
#define __host__
#define __device__
// extern "C" keeps the entry name (and its __vx_kentry_<name> alias, which
// VOLT emits for annotate("vortex.kernel")) unmangled so kernels can be
// launched by name from a .vxbin symbol table. Same convention as
// vx_spawn2.h's __kernel.
#define __global__ extern "C" __attribute__((annotate("vortex.kernel"), used, retain))
#endif

struct dim3 {
    union {
        struct { uint32_t x, y, z; };
        uint32_t m[3];
    };
    __host__ __device__ dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

#define __HIP_VORTEX_VEC2(name, T) \
    struct name { \
        T x, y; \
        __host__ __device__ name(T x_ = 0, T y_ = 0) : x(x_), y(y_) {} \
    }

#define __HIP_VORTEX_VEC4(name, T) \
    struct name { \
        T x, y, z, w; \
        __host__ __device__ name(T x_ = 0, T y_ = 0, T z_ = 0, T w_ = 0) \
            : x(x_), y(y_), z(z_), w(w_) {} \
    }

__HIP_VORTEX_VEC2(char2, char);
__HIP_VORTEX_VEC2(uchar2, unsigned char);
__HIP_VORTEX_VEC2(short2, short);
__HIP_VORTEX_VEC2(ushort2, unsigned short);
__HIP_VORTEX_VEC2(int2, int);
__HIP_VORTEX_VEC2(uint2, unsigned int);
__HIP_VORTEX_VEC2(long2, long);
__HIP_VORTEX_VEC2(ulong2, unsigned long);
__HIP_VORTEX_VEC2(float2, float);
__HIP_VORTEX_VEC2(double2, double);

__HIP_VORTEX_VEC4(char4, char);
__HIP_VORTEX_VEC4(uchar4, unsigned char);
__HIP_VORTEX_VEC4(short4, short);
__HIP_VORTEX_VEC4(ushort4, unsigned short);
__HIP_VORTEX_VEC4(int4, int);
__HIP_VORTEX_VEC4(uint4, unsigned int);
__HIP_VORTEX_VEC4(long4, long);
__HIP_VORTEX_VEC4(ulong4, unsigned long);
__HIP_VORTEX_VEC4(float4, float);
__HIP_VORTEX_VEC4(double4, double);

// longlong/ulonglong variants keep the LLP64-style names used by HIP sources.
__HIP_VORTEX_VEC2(longlong2, long long);
__HIP_VORTEX_VEC2(ulonglong2, unsigned long long);
__HIP_VORTEX_VEC4(longlong4, long long);
__HIP_VORTEX_VEC4(ulonglong4, unsigned long long);

#undef __HIP_VORTEX_VEC2
#undef __HIP_VORTEX_VEC4

#endif // HIP_VORTEX_VECTOR_TYPES_H
