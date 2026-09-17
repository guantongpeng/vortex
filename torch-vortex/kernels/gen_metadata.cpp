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

// Emits the VXKMDATA kernel metadata from *measured* struct sizes.
//
// The sizes used to be typed by hand into the Makefile, and had drifted from
// the real structs (conv 104 vs 88, pool 80 vs 72). The runtime copies exactly
// args_size bytes out of the host argument block, so a size that is too large
// is a host stack over-read. This program asks the host compiler for sizeof()
// on the same header the device compiler uses, so there is nothing left to
// hand-maintain or get wrong.
//
// Build:
//   $(CXX) -std=c++17 -I. -o gen_metadata gen_metadata.cpp
// Use:
//   ./gen_metadata --xlen=64 --meta all_meta.json --sidecar torch_all.vxbin.meta.json
//
// With no --meta the metadata JSON goes to stdout (the form VX_KERNEL_METADATA
// wants). --sidecar additionally writes the richer file the extension
// cross-checks itself against at init time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "torch_kernel_args.h"

// The host extension is always LP64: it is compiled by the host compiler, and
// its argument blocks are built with host pointers. So an rv32 image can never
// match what this program emits, and the build refuses rather than producing a
// half-width blob that the kernel would misread.
static_assert(sizeof(void*) == 8,
              "torch-vortex argument blocks are LP64; build with XLEN=64");

// Every argument type must be a plain, copyable, standard-layout aggregate:
// the runtime memcpy()s it between host and device, so anything with a vtable,
// padding quirks or a non-trivial copy would break silently.
#define TORCH_ASSERT_LAYOUT(name, type, mbx, lmem)                            \
    static_assert(std::is_standard_layout<type>::value,                       \
                  #type " must be standard layout");                          \
    static_assert(std::is_trivially_copyable<type>::value,                    \
                  #type " must be trivially copyable");                       \
    static_assert(alignof(type) <= 8, #type " aligns wider than 8 bytes");

TORCH_KERNEL_TABLE(TORCH_ASSERT_LAYOUT)

namespace {

int g_xlen = 64;

std::string json_escape(const char* s) {
    std::string out;
    for (const char* p = s; *p; ++p) {
        if (*p == '"' || *p == '\\') out.push_back('\\');
        out.push_back(*p);
    }
    return out;
}

void write_file(const std::string& path, const std::string& body) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "gen_metadata: cannot write %s\n", path.c_str());
        std::exit(1);
    }
    std::fwrite(body.data(), 1, body.size(), f);
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    std::string meta_path, sidecar_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&](const char* prefix) -> const char* {
            const size_t n = std::strlen(prefix);
            return a.compare(0, n, prefix) == 0 ? argv[i] + n : nullptr;
        };
        if (const char* v = value("--xlen=")) g_xlen = std::atoi(v);
        else if (const char* v = value("--meta=")) meta_path = v;
        else if (const char* v = value("--sidecar=")) sidecar_path = v;
        else {
            std::fprintf(stderr,
                         "usage: gen_metadata [--xlen=N] [--meta=PATH] "
                         "[--sidecar=PATH]\n");
            return 2;
        }
    }

    // VXKMDATA records. args_size is sizeof() of the very struct both
    // compilers parsed above.
    std::string meta = "[";
    bool first = true;
#define TORCH_EMIT_RECORD(name, type, mbx, lmem)                              \
    do {                                                                      \
        char buf[512];                                                        \
        int off = std::snprintf(buf, sizeof(buf),                             \
                                "%s{\"name\": \"%s\", \"args_size\": %zu",     \
                                first ? "" : ", ", #name, sizeof(type));      \
        if ((mbx) > 0)                                                        \
            off += std::snprintf(buf + off, sizeof(buf) - off,                \
                                 ", \"max_block\": [%d, 1, 1]", (int)(mbx));  \
        if ((lmem) > 0)                                                       \
            off += std::snprintf(buf + off, sizeof(buf) - off,                \
                                 ", \"static_lmem_bytes\": %d", (int)(lmem)); \
        std::snprintf(buf + off, sizeof(buf) - off, "}");                     \
        meta += buf;                                                          \
        first = false;                                                        \
    } while (0);

    TORCH_KERNEL_TABLE(TORCH_EMIT_RECORD)
    meta += "]\n";

    if (meta_path.empty()) {
        std::fputs(meta.c_str(), stdout);
    } else {
        write_file(meta_path, meta);
    }

    if (!sidecar_path.empty()) {
        std::string sc =
            "{\n  \"schema\": 1,\n  \"xlen\": " + std::to_string(g_xlen) +
            ",\n  \"args_sizes\": {\n";
        bool f2 = true;
#define TORCH_EMIT_SIDECAR(name, type, mbx, lmem)                             \
    sc += std::string(f2 ? "    " : ",\n    ") + "\"" + #name + "\": " +      \
          std::to_string(sizeof(type));                                       \
    f2 = false;
        TORCH_KERNEL_TABLE(TORCH_EMIT_SIDECAR)
        sc += "\n  }\n}\n";
        write_file(sidecar_path, sc);
    }
    return 0;
}
