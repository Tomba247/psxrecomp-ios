/* Large zero-filled buffers taken from ordinary memory at startup rather than
 * from the program's own data. Henge loads a program's whole image, zero-fill
 * included, into its limited executable region. Pages are used when touched,
 * as with .bss. The address is valid before main and before other ordinary
 * constructors (priority 101). */
#ifndef PSXRECOMP_PSX_BIG_H
#define PSXRECOMP_PSX_BIG_H
#if defined(__unix__) || defined(__APPLE__)
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
static inline void *psx_big_take(size_t bytes, const char *what) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) { fprintf(stderr, "psxrecomp: no memory for %s\n", what); abort(); }
    return p;
}
#define PSX_BIG(linkage, type, name, count) \
    linkage type *name; \
    __attribute__((constructor(101))) static void name##_take(void) { \
        name = (type *)psx_big_take(sizeof(type) * (size_t)(count), #name); \
    }
#else /* elsewhere: an ordinary zero-filled array behind the same pointer */
#include "psx_bss.h"
#define PSX_BIG(linkage, type, name, count) \
    static PSX_BSS type name##_storage[count]; \
    linkage type *name = name##_storage;
#endif
#endif
