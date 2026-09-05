/* Device-capability probe for antirez/h3.c, weights not required.
 *
 * h3_metal_probe() is standalone: it queries the Metal device and process
 * memory without mapping any checkpoint. That makes it the one part of h3
 * we can run before committing to a ~130-160GB weight download, and it
 * answers the question the h3 README does not: what does this machine
 * actually offer, given the docs only ever name M3 Max and M5 Max.
 *
 * Build: see tools/Makefile (links against h3.c's libh3.a).
 */
#include <stdio.h>
#include <string.h>
#include "h3.h"
#include "h3_metal.h"

static double gib(unsigned long long bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

int main(void) {
    h3_device_info d;
    char error[256] = {0};

    if (!h3_metal_probe(&d, error, sizeof(error))) {
        fprintf(stderr, "probe failed: %s\n", error[0] ? error : "unknown");
        return 1;
    }

    printf("Device: %s (%s)\n", d.name, d.architecture);
    printf("  physical memory       %.1f GiB\n", gib(d.physical_memory));
    printf("  recommended GPU set   %.1f GiB\n", gib(d.recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(d.max_buffer_length));
    printf("  Apple GPU family      %d\n", d.apple_gpu_family);
    printf("  Metal 4               %s\n", d.metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", d.unified_memory ? "yes" : "no");

    /* h3 gates its TensorOps/int8 path on a LITERAL SUBSTRING MATCH of the
     * device name against "M5" (h3_gpu.m):
     *
     *     BOOL m5 = [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
     *     BOOL wantsTensorOps = m5 && (!nax || !*nax || strcmp(nax,"0") != 0);
     *
     * It is NOT a capability query. The `metal4` field above comes from
     * [device supportsFamily:MTLGPUFamilyMetal4], which macOS 26 reports "yes"
     * for far older GPUs than the ones that have tensor hardware, and h3 never
     * consults it for gating. Reading metal4 as "int8 available" is wrong. */
    int tensor_ops = strstr(d.name, "M5") != NULL;

    printf("\nImplications for h3.c on this machine:\n");
    printf("  TensorOps / int8 MLP  %s\n",
           tensor_ops ? "available"
                      : "UNAVAILABLE, BF16 path only (name has no \"M5\")");
    printf("    note: Metal 4 reads \"%s\" above but h3 ignores it for gating\n",
           d.metal4 ? "yes" : "no");
    printf("  max single tensor     %.1f GiB (hard per-buffer cap)\n",
           gib(d.max_buffer_length));
    printf("  GPU set vs 36.5 GiB full-residency DiT   %s\n",
           gib(d.recommended_working_set) >= 36.5
               ? "fits without --ssd-streaming"
               : "DOES NOT FIT, --ssd-streaming required");
    printf("  GPU set vs 2.0 GiB streamed DiT          %s\n",
           gib(d.recommended_working_set) >= 2.0 ? "fits" : "does not fit");
    return 0;
}
