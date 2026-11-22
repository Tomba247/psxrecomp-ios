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
/* iOS refuses GPU work from an app that is not active (MoltenVK then reports
 * the device lost for good). Call before acquiring/submitting: returns at once
 * while the app is active, otherwise holds the caller, pumping events, until
 * the app is active again. */
void psx_ios_wait_foreground(void);
#ifdef __cplusplus
}
#endif
#endif
#endif
