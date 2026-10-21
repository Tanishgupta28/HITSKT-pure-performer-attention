#include <stddef.h>
#include <stdio.h>
#include <nvml.h>

/* CPU layout check only: no NVML/CUDA linking or function calls. */
int main(void) {
    printf("{\"size\":%zu,\"pid\":%zu,\"usedGpuMemory\":%zu,"
           "\"gpuInstanceId\":%zu,\"computeInstanceId\":%zu}\n",
           sizeof(nvmlProcessInfo_v2_t), offsetof(nvmlProcessInfo_v2_t,pid),
           offsetof(nvmlProcessInfo_v2_t,usedGpuMemory),
           offsetof(nvmlProcessInfo_v2_t,gpuInstanceId),
           offsetof(nvmlProcessInfo_v2_t,computeInstanceId));
    return 0;
}
