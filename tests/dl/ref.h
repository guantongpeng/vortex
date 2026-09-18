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

#ifndef VORTEX_DL_TEST_REF_H
#define VORTEX_DL_TEST_REF_H

// Shared CPU reference implementations for tests/dl (plan §7.2: every
// operator is checked against a double-precision host reference with a
// fixed seed before any performance claim).

#include <cmath>
#include <cstdint>
#include <vector>

namespace ref {

inline double gelu(double x) {
    // tanh approximation — matches prim_kernels.hip's VX_PRIM_GELU_TANH
    return 0.5 * x *
           (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

// torch's default gelu, the erf form. The two are different functions, not
// two spellings of one, which is why the kernels name both.
inline double gelu_erf(double x) {
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}

inline double silu(double x) { return x / (1.0 + std::exp(-x)); }

inline void softmax(const std::vector<float>& in, uint32_t rows,
                    uint32_t cols, std::vector<double>& out) {
    out.assign((size_t)rows * cols, 0.0);
    for (uint32_t r = 0; r < rows; ++r) {
        double m = -INFINITY;
        for (uint32_t j = 0; j < cols; ++j) {
            m = std::max(m, (double)in[(size_t)r * cols + j]);
        }
        double s = 0.0;
        for (uint32_t j = 0; j < cols; ++j) {
            double e = std::exp((double)in[(size_t)r * cols + j] - m);
            out[(size_t)r * cols + j] = e;
            s += e;
        }
        for (uint32_t j = 0; j < cols; ++j) {
            out[(size_t)r * cols + j] /= s;
        }
    }
}

// The per-row mean and rstd, which is what aten::native_layer_norm returns
// alongside the normalised output. The variance is the mean squared deviation
// from the mean -- NOT E[x^2] - mean^2, which is the same number in exact
// arithmetic and a different one in floating point. Both forms are written in
// double here so the reference itself is not the limiting factor; the kernel
// runs in float32 and is compared to this.
inline void layer_stats(const std::vector<float>& in, uint32_t rows,
                        uint32_t cols, double eps, std::vector<double>& mean,
                        std::vector<double>& rstd) {
    mean.assign(rows, 0.0);
    rstd.assign(rows, 0.0);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (size_t)r * cols;
        double s1 = 0.0;
        for (uint32_t j = 0; j < cols; ++j) s1 += row[j];
        const double m = s1 / cols;
        double s2 = 0.0;
        for (uint32_t j = 0; j < cols; ++j) {
            const double d = (double)row[j] - m;
            s2 += d * d;
        }
        mean[r] = m;
        rstd[r] = 1.0 / std::sqrt(s2 / cols + eps);
    }
}

inline void layernorm(const std::vector<float>& in,
                      const std::vector<float>& gamma,
                      const std::vector<float>& beta, uint32_t rows,
                      uint32_t cols, double eps, std::vector<double>& out) {
    out.assign((size_t)rows * cols, 0.0);
    std::vector<double> mean, rstd;
    layer_stats(in, rows, cols, eps, mean, rstd);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (size_t)r * cols;
        for (uint32_t j = 0; j < cols; ++j) {
            out[(size_t)r * cols + j] =
                ((double)row[j] - mean[r]) * rstd[r] * gamma[j] + beta[j];
        }
    }
}

inline void rmsnorm(const std::vector<float>& in,
                    const std::vector<float>& gamma, uint32_t rows,
                    uint32_t cols, double eps, std::vector<double>& out) {
    out.assign((size_t)rows * cols, 0.0);
    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = in.data() + (size_t)r * cols;
        double s2 = 0.0;
        for (uint32_t j = 0; j < cols; ++j) {
            s2 += (double)row[j] * row[j];
        }
        double inv_rms = 1.0 / std::sqrt(s2 / cols + eps);
        for (uint32_t j = 0; j < cols; ++j) {
            out[(size_t)r * cols + j] = (double)row[j] * inv_rms * gamma[j];
        }
    }
}

} // namespace ref

#endif // VORTEX_DL_TEST_REF_H
