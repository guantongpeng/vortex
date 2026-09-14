// P2.3 item 6: hiprtc is explicitly not implemented — the stubs must
// return hipErrorNotSupported (never silently succeed or crash).

#include <hip/hip_runtime_api.h>

#include <cstdio>

int main() {
    hipError_t e = hipInit(0);
    if (e != hipSuccess) { fprintf(stderr, "hipInit failed\n"); return 1; }

    hiprtcProgram prog = nullptr;
    if (hiprtcCreateProgram(&prog, "__global__ void k(){}", "k.hip", 0,
                            nullptr, nullptr) != hipErrorNotSupported) {
        fprintf(stderr, "FAILED: hiprtcCreateProgram must return not-supported\n");
        return 1;
    }
    if (hiprtcCompileProgram(prog, 0, nullptr) != hipErrorNotSupported ||
        hiprtcGetCodeSize(prog, nullptr) != hipErrorNotSupported ||
        hiprtcGetCode(prog, nullptr) != hipErrorNotSupported ||
        hiprtcDestroyProgram(&prog) != hipErrorNotSupported) {
        fprintf(stderr, "FAILED: hiprtc stubs must all return not-supported\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
