/* psx_ios.h — one place that decides "this is an iOS/iPadOS build".
 * iOS forbids spawning processes and loading unsigned code, and offers
 * OpenGL ES instead of desktop OpenGL. */
#ifndef PSX_IOS_H
#define PSX_IOS_H
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define PSX_IOS 1
#endif
#endif
#endif
