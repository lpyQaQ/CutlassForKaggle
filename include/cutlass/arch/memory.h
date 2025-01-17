/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Fragment type to store loaded data
        typename AccessType,
        /// The bytes of loading
        int LoadBytes>
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11)) &&                                  \
        defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) &&           \
        !(defined(__clang__) && defined(__CUDA__))
#define CUTLASS_ENABLE_L2_PREFETCH 1
#else
#define CUTLASS_ENABLE_L2_PREFETCH 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// initialize data to zero before ld.global
template <typename AccessType>
struct global_load<AccessType, 64> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        uint4* data = reinterpret_cast<uint4*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %16, 0;\n"
                "  mov.b32 %0, %17;\n"
                "  mov.b32 %1, %17;\n"
                "  mov.b32 %2, %17;\n"
                "  mov.b32 %3, %17;\n"
                "  mov.b32 %4, %17;\n"
                "  mov.b32 %5, %17;\n"
                "  mov.b32 %6, %17;\n"
                "  mov.b32 %7, %17;\n"
                "  mov.b32 %8, %17;\n"
                "  mov.b32 %9, %17;\n"
                "  mov.b32 %10, %17;\n"
                "  mov.b32 %11, %17;\n"
                "  mov.b32 %12, %17;\n"
                "  mov.b32 %13, %17;\n"
                "  mov.b32 %14, %17;\n"
                "  mov.b32 %15, %17;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%18];\n"
                "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%19];\n"
                "  @p ld.global.v4.u32 {%8, %9, %10, %11}, [%20];\n"
                "  @p ld.global.v4.u32 {%12, %13, %14, %15}, [%21];\n"
                "}\n"
                : "=r"(data[0].x), "=r"(data[0].y),  //  0,  1
                  "=r"(data[0].z), "=r"(data[0].w),  //  2,  3
                  "=r"(data[1].x), "=r"(data[1].y),  //  4,  5
                  "=r"(data[1].z), "=r"(data[1].w),  //  6,  7
                  "=r"(data[2].x), "=r"(data[2].y),  //  8,  9
                  "=r"(data[2].z), "=r"(data[2].w),  // 10, 11
                  "=r"(data[3].x), "=r"(data[3].y),  // 12, 13
                  "=r"(data[3].z), "=r"(data[3].w)   // 14, 15
                : "r"((int)pred_guard),              // 16
                  "r"(pack_pad),                     // 17
                  "l"(ptr),                          // 18
                  "l"(((uint8_t*)ptr) + 16),         // 19
                  "l"(((uint8_t*)ptr) + 32),         // 20
                  "l"(((uint8_t*)ptr) + 48));        // 21
    }
};

template <typename AccessType>
struct global_load<AccessType, 32> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        uint4* data = reinterpret_cast<uint4*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %8, 0;\n"
                "  mov.b32 %0, %9;\n"
                "  mov.b32 %1, %9;\n"
                "  mov.b32 %2, %9;\n"
                "  mov.b32 %3, %9;\n"
                "  mov.b32 %4, %9;\n"
                "  mov.b32 %5, %9;\n"
                "  mov.b32 %6, %9;\n"
                "  mov.b32 %7, %9;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%10];\n"
                "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%11];\n"
                "}\n"
                : "=r"(data[0].x), "=r"(data[0].y),  //  0,  1
                  "=r"(data[0].z), "=r"(data[0].w),  //  2,  3
                  "=r"(data[1].x), "=r"(data[1].y),  //  4,  5
                  "=r"(data[1].z), "=r"(data[1].w)   //  6,  7
                : "r"((int)pred_guard),              //  8
                  "r"(pack_pad),                     //  9
                  "l"(ptr),                          // 10
                  "l"(((uint8_t*)ptr) + 16));        // 11
    }
};

template <typename AccessType>
struct global_load<AccessType, 16> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        uint4& data = reinterpret_cast<uint4&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %4, 0;\n"
                "  mov.b32 %0, %5;\n"
                "  mov.b32 %1, %5;\n"
                "  mov.b32 %2, %5;\n"
                "  mov.b32 %3, %5;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%6];\n"
                "}\n"
                : "=r"(data.x), "=r"(data.y),  // 0, 1
                  "=r"(data.z), "=r"(data.w)   // 2, 3
                : "r"((int)pred_guard),        // 4
                  "r"(pack_pad),               // 5
                  "l"(ptr));                   // 6
    }
};

template <typename AccessType>
struct global_load<AccessType, 8> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        uint2& data = reinterpret_cast<uint2&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  mov.b32 %0, %3;\n"
                "  mov.b32 %1, %3;\n"
                "  @p ld.global.v2.u32 {%0, %1}, [%4];\n"
                "}\n"
                : "=r"(data.x), "=r"(data.y)                       // 0, 1
                : "r"((int)pred_guard), "r"(pack_pad), "l"(ptr));  // 2, 3, 4
    }
};

template <typename AccessType>
struct global_load<AccessType, 4> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        unsigned& data = reinterpret_cast<unsigned&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %1, 0;\n"
                "  mov.b32 %0, %2;\n"
                "  @p ld.global.u32 %0, [%3];\n"
                "}\n"
                : "=r"(data)                                       // 0
                : "r"((int)pred_guard), "r"(pack_pad), "l"(ptr));  // 1, 2, 3
    }
};

template <typename AccessType>
struct global_load<AccessType, 2> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        uint16_t& data = reinterpret_cast<uint16_t&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %1, 0;\n"
                "  mov.b16 %0, %2;\n"
                "  @p ld.global.u16 %0, [%3];\n"
                "}\n"
                : "=h"(data)                           // 0
                : "r"((int)pred_guard),                // 1
                  "h"(uint16_t(pack_pad)), "l"(ptr));  // 2, 3
    }
};

template <typename AccessType>
struct global_load<AccessType, 1> {
    CUTLASS_DEVICE
    global_load(AccessType& D, void const* ptr, bool pred_guard,
                uint32_t pack_pad = 0) {
        if (pred_guard) {
            D = *(reinterpret_cast<AccessType const*>(ptr));
        } else {
            uint8_t val = pack_pad & 0xFF;
            D = *(reinterpret_cast<AccessType const*>(&val));
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
        /// Fragment type to store loaded data
        typename AccessType,
        /// The bytes of loading
        int LoadBytes>
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_store<AccessType, 64> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const* data = reinterpret_cast<uint4 const*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
                "  @p st.global.v4.u32 [%11], {%12, %13, %14, %15};\n"
                "  @p st.global.v4.u32 [%16], {%17, %18, %19, %20};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
                  "r"(data[0].w), "r"((int)pred_guard),
                  "l"(((uint8_t*)ptr) + 16), "r"(data[1].x), "r"(data[1].y),
                  "r"(data[1].z), "r"(data[1].w), "l"(((uint8_t*)ptr) + 32),
                  "r"(data[2].x), "r"(data[2].y), "r"(data[2].z),
                  "r"(data[2].w), "l"(((uint8_t*)ptr) + 48), "r"(data[3].x),
                  "r"(data[3].y), "r"(data[3].z), "r"(data[3].w));
    }
};

template <typename AccessType>
struct global_store<AccessType, 32> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const* data = reinterpret_cast<uint4 const*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
                  "r"(data[0].w), "r"((int)pred_guard),
                  "l"(((uint8_t*)ptr) + 16), "r"(data[1].x), "r"(data[1].y),
                  "r"(data[1].z), "r"(data[1].w));
    }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const& data = reinterpret_cast<uint4 const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w),
                  "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint2 const& data = reinterpret_cast<uint2 const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %3, 0;\n"
                "  @p st.global.v2.u32 [%0], {%1, %2};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data.x), "r"(data.y), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint32_t const& data = reinterpret_cast<uint32_t const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  @p st.global.u32 [%0], %1;\n"
                "}\n"
                :
                : "l"(ptr), "r"(data), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint16_t const& data = reinterpret_cast<uint16_t const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  @p st.global.u16 [%0], %1;\n"
                "}\n"
                :
                : "l"(ptr), "h"(data), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
    CUTLASS_DEVICE
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        if (pred_guard)
            *(reinterpret_cast<AccessType*>(ptr)) = D;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "memory_sm75.h"
#include "memory_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
