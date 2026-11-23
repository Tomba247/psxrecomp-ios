/* host_touch_ui.h -- on-screen touch controls, settings button and settings
 * panel for touch-screen hosts (iOS). Host presentation only: nothing here is
 * part of guest VRAM or save states.
 *
 * Enabled on iOS, or anywhere with PSX_TOUCH_UI=1 in the environment. When
 * disabled every call is inert.
 *
 * The UI is drawn on the CPU into one premultiplied ARGB8888 canvas that the
 * renderer alpha-blends over the finished frame (Vulkan backend).
 */
#ifndef PSX_HOST_TOUCH_UI_H
#define PSX_HOST_TOUCH_UI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int  touch_ui_enabled(void);

/* Every host SDL event (main thread). Returns 1 when the event is consumed. */
int  touch_ui_event(const void *sdl_event);

/* Test/debug injection: phase 0 = down, 1 = move, 2 = up; x,y in 0..1. */
void touch_ui_inject(int finger, int phase, float x, float y);

/* The virtual pad: returns 1 while the touch pad is on screen and fills the
 * PSX active-low button word and centred sticks. */
int  touch_ui_get_pad(uint16_t *buttons, uint8_t st[4]);

/* Renderer side. Returns 1 when there is something to draw. The canvas keeps
 * the aspect of the sc_w x sc_h drawable; *version changes when the pixels
 * do. The pointer stays valid until the next call. */
int  touch_ui_canvas(int sc_w, int sc_h, const uint32_t **px, int *w, int *h,
                     unsigned *version);
/* One call per presented frame (FPS counter). */
void touch_ui_presented(void);
/* 1 = stretch the picture over the whole screen instead of letterboxing. */
int  touch_ui_fit_screen(void);
/* 1 = generate in-between pictures for 60 fps (Vulkan frame generation). */
int  touch_ui_framegen(void);
/* Internal resolution chosen in the panel for THIS run: 0 = the game's
 * default, else a 1..6 x scale. A change applies at the next start. */
int  touch_ui_resolution(void);

#ifdef __cplusplus
}
#endif
#endif
