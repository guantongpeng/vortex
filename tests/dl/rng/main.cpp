// Philox 4x32-10 RNG acceptance test (plan P3 first tier).
//
// 1. The host reference below is an INDEPENDENT implementation of the same
//    published algorithm (Salmon et al. SC'11 / Random123 philox.h), and it
//    first self-checks the published test vector
//      Philox4x32-10(ctr={0,0,0,0}, key={0,0}) =
//          {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
//    so the constants (M=0xD2511F53/0xCD9E8D57, W=0x9E3779B9/0xBB67AE85)
//    and round structure are validated against the paper, not just against
//    the device kernel.
// 2. Device outputs are compared BIT-EXACTLY against the reference: both
//    sides compute (float)(w >> 8) * 2^-24, an exact IEEE op sequence
//    (24-bit int -> fp32 is exact, scaling by a power of two is exact), so
//    any bit difference is a real defect. The first 16 outputs are the
//    mandated gate; the whole array is also checked (stronger, no
//    tolerance weakened anywhere).
// 3. Statistical sanity (n = 4096): |mean - 0.5| <= 3/sqrt(n) (double
//    accumulation), min >= 0, max < 1.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <vortex2.h>

// --- self-contained public API of host.cpp --------------------------------
extern "C" {
typedef enum vx_rng_status {
    VX_RNG_OK = 0,
    VX_RNG_ERR_NOT_INITIALIZED = 1,
    VX_RNG_ERR_BAD_ARGS = 2,
    VX_RNG_ERR_LAUNCH = 3,
} vx_rng_status;
vx_rng_status vx_rng_init(vx_device_h dev, const char* vxbin_path);
vx_rng_status vx_rng_finalize(void);
vx_rng_status vx_rng_uniform_f32(vx_queue_h q, uint64_t out, uint32_t n,
                                 uint64_t seed, uint64_t offset);
}

#define CHECK(expr) do { \
    if ((expr) != 0) { \
        fprintf(stderr, "FAILED at %s:%d: '%s'\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

// --- independent host Philox4x32-10 ----------------------------------------
// mulhilo32 via a 64-bit product, exactly as the paper specifies.
static void mulhilo32(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo) {
    const uint64_t p = (uint64_t)a * (uint64_t)b;
    *hi = (uint32_t)(p >> 32);
    *lo = (uint32_t)p;
}

static void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t key0, uint32_t key1, uint32_t out[4]) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key0 += 0x9E3779B9u;
            key1 += 0xBB67AE85u;
        }
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo32(0xD2511F53u, c0, &hi0, &lo0);
        mulhilo32(0xCD9E8D57u, c2, &hi1, &lo1);
        const uint32_t n0 = hi1 ^ c1 ^ key0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ key1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// 24-bit round-down uniform, identical arithmetic to the kernel.
static float bits_to_u01(uint32_t bits) {
    return (float)(bits >> 8) * (1.0f / 16777216.0f);
}

// Reference stream: block b uses ctr = offset + b (64-bit), key = seed.
static void host_gen(std::vector<float>& out, uint32_t n, uint64_t seed,
                     uint64_t offset) {
    out.assign(n, 0.0f);
    for (uint32_t i = 0; i < n; i += 4) {
        const uint64_t ctr = offset + (uint64_t)(i / 4);
        uint32_t w[4];
        philox4x32_10((uint32_t)ctr, (uint32_t)(ctr >> 32), 0, 0,
                      (uint32_t)seed, (uint32_t)(seed >> 32), w);
        for (int j = 0; j < 4 && i + (uint32_t)j < n; ++j)
            out[i + j] = bits_to_u01(w[j]);
    }
}

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

struct DevBuf {
    vx_buffer_h h = nullptr;
    uint64_t addr = 0;
};

static DevBuf make_buf(vx_device_h dev, size_t bytes) {
    DevBuf b;
    vx_buffer_create(dev, bytes, 0, &b.h);
    vx_buffer_address(b.h, &b.addr);
    return b;
}

static void download(vx_queue_h q, std::vector<float>& dst, const DevBuf& b) {
    vx_event_h ev = nullptr;
    vx_enqueue_read(q, dst.data(), b.h, 0, dst.size() * 4, 0, nullptr, &ev);
    vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
    vx_event_release(ev);
}

// Runs one device case and applies the gates. Returns 0 on pass.
static int run_case(vx_queue_h q, vx_device_h dev, const char* name,
                    uint32_t n, uint64_t seed, uint64_t offset,
                    int stats_gate) {
    std::vector<float> ref;
    host_gen(ref, n, seed, offset);

    DevBuf bo = make_buf(dev, (size_t)n * 4);
    CHECK(vx_rng_uniform_f32(q, bo.addr, n, seed, offset));
    CHECK(vx_queue_flush(q));
    std::vector<float> got(n);
    download(q, got, bo);
    vx_buffer_release(bo.h);

    uint32_t first16_bad = 0, all_bad = 0;
    double max_rel = 0.0;
    const uint32_t first = n < 16 ? n : 16;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t gb = fbits(got[i]), rb = fbits(ref[i]);
        if (gb != rb) {
            ++all_bad;
            if (i < first) ++first16_bad;
        }
        const double rel = fabs((double)got[i] - (double)ref[i]) /
                           ((double)ref[i] > 0.0 ? (double)ref[i] : 1.0);
        if (rel > max_rel) max_rel = rel;
    }
    double mean = 0.0, minv = 1.0, maxv = -1.0;
    for (float x : got) {
        mean += (double)x;                 // double accumulation
        if ((double)x < minv) minv = (double)x;
        if ((double)x > maxv) maxv = (double)x;
    }
    mean /= (double)n;
    const double bound = 3.0 / sqrt((double)n);
    const double dev_mean = fabs(mean - 0.5);

    printf("%-8s n=%-5u first16_bad=%u all_bad=%u max_rel=%.2e\n", name, n,
           first16_bad, all_bad, max_rel);
    printf("         mean=%.6f |mean-0.5|=%.6f bound=3/sqrt(n)=%.6f "
           "min=%.6f max=%.6f\n",
           mean, dev_mean, bound, minv, maxv);

    int failures = 0;
    if (first16_bad) {
        printf("         FAILED: first 16 outputs not bit-exact\n");
        ++failures;
    }
    if (all_bad) {
        printf("         FAILED: %u/%u outputs not bit-exact\n", all_bad, n);
        ++failures;
    }
    if (stats_gate) {
        if (dev_mean > bound) {
            printf("         FAILED: |mean-0.5| > 3/sqrt(n)\n");
            ++failures;
        }
        if (!(minv >= 0.0) || !(maxv < 1.0)) {
            printf("         FAILED: values outside [0,1)\n");
            ++failures;
        }
    }
    return failures;
}

int main(int argc, char** argv) {
    const char* vxbin = (argc > 1) ? argv[1] : "rng.vxbin";

    // Gate 0: reference self-check against the published Random123 vector.
    {
        uint32_t w[4];
        philox4x32_10(0, 0, 0, 0, 0, 0, w);
        const uint32_t expect[4] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu,
                                    0x9b00dbd8u};
        const int ok = memcmp(w, expect, 16) == 0;
        printf("vector:  philox4x32-10(ctr=0,key=0) = "
               "{%08x,%08x,%08x,%08x} expect {6627e8d5,e169c58d,bc57ac4c,"
               "9b00dbd8} -> %s\n",
               w[0], w[1], w[2], w[3], ok ? "match" : "MISMATCH");
        if (!ok) {
            fprintf(stderr, "FAILED: host reference does not reproduce the "
                            "Random123 test vector\n");
            return 1;
        }
    }

    vx_device_h dev = nullptr;
    CHECK(vx_device_open(0, &dev));
    vx_queue_h q = nullptr;
    CHECK(vx_queue_create(dev, nullptr, &q));
    if (vx_rng_init(dev, vxbin) != VX_RNG_OK) {
        fprintf(stderr, "FAILED: vx_rng_init(%s)\n", vxbin);
        return 1;
    }

    int failures = 0;
    // Case A: n=4096, both key words nonzero, offset 0.
    failures += run_case(q, dev, "caseA", 4096, 0x9E3779B97F4A7C15ull, 0, 1);
    // Case B: 64-bit offset whose low word wraps into the high word at
    // block 3 (ctr crosses 2^32: 0xFFFFFFFD + b), different seed.
    failures += run_case(q, dev, "caseB", 4096, 0x0123456789ABCDEFull,
                         0xFFFFFFFDull, 1);
    // Case C: n not a multiple of 4 (tail masking), small offset.
    failures += run_case(q, dev, "caseC", 101, 1, 3, 0);

    vx_rng_finalize();
    vx_queue_release(q);
    vx_device_release(dev);

    if (failures == 0) {
        printf("PASSED\n");
        return 0;
    }
    fprintf(stderr, "FAILED (%d)\n", failures);
    return 1;
}
