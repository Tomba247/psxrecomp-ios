#ifndef PSX_IOS_BOOT_H
#define PSX_IOS_BOOT_H
#include "psx_ios.h"
#ifdef PSX_IOS
#ifdef __cplusplus
extern "C" {
#endif
/* iOS has no command line and a read-only app bundle. Before anything else in
 * main(): send stdout/stderr to Documents/psxrecomp.log, make Documents the
 * working directory, and build the argument list the desktop CLI would get. */
void psx_ios_prepare(int *argc, char ***argv);
#ifdef __cplusplus
}
#endif
#endif
#endif
