/* host_touch_ui.c -- on-screen touch pad, settings button and settings panel
 * (see host_touch_ui.h).
 *
 * Units: everything is laid out in "u", where the screen is 100 u tall and
 * 100 * aspect u wide, so the layout is the same on every screen size.
 */
#include "host_touch_ui.h"
#include "host_font8x8.h"
#include "psx_ios.h"
#include "psx_sdl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(PSX_SDL3)

int  touch_ui_enabled(void) { return 0; }
int  touch_ui_event(const void *e) { (void)e; return 0; }
void touch_ui_inject(int f, int p, float x, float y) { (void)f; (void)p; (void)x; (void)y; }
int  touch_ui_get_pad(uint16_t *b, uint8_t st[4]) { (void)b; (void)st; return 0; }
int  touch_ui_canvas(int a, int b, const uint32_t **px, int *w, int *h, unsigned *v) {
    (void)a; (void)b; (void)px; (void)w; (void)h; (void)v; return 0;
}
void touch_ui_presented(void) {}
int  touch_ui_fit_screen(void) { return 0; }
int  touch_ui_framegen(void) { return 0; }
int  touch_ui_resolution(void) { return 0; }
int  touch_ui_tex_smooth(void) { return 0; }
int  touch_ui_msaa(void) { return 1; }
int  touch_ui_hd_pack(void) { return 1; }

#else

/* PSX pad word, active low. */
#define B_SELECT 0x0001u
#define B_START  0x0008u
#define B_UP     0x0010u
#define B_RIGHT  0x0020u
#define B_DOWN   0x0040u
#define B_LEFT   0x0080u
#define B_L2     0x0100u
#define B_R2     0x0200u
#define B_L1     0x0400u
#define B_R1     0x0800u
#define B_TRI    0x1000u
#define B_CIRCLE 0x2000u
#define B_CROSS  0x4000u
#define B_SQUARE 0x8000u

enum { MODE_AUTO = 0, MODE_ALWAYS = 1, MODE_OFF = 2 };
enum { ROW_SMOOTH = 0, ROW_FPS, ROW_FIT, ROW_RES, ROW_AA, ROW_TEX, ROW_DEPTH, ROW_GFX, ROW_TOUCH, ROW_OPACITY, ROW_CLOSE, ROW_COUNT };

#define MAX_FINGERS 12
typedef struct { int used; long long id; float x, y; int ui; } Finger;

typedef struct {
    int kind;            /* 0 circle, 1 pill */
    float cx, cy, r, hw, hh;
    uint16_t bit;
    const char *label;   /* pills */
} Ctl;
#define CTL_COUNT 10

static SDL_Mutex *s_mx;
static int s_init, s_enabled;
static Finger s_f[MAX_FINGERS];
static int s_menu_open;
static int s_show_fps = 0, s_fit = 0, s_mode = MODE_AUTO, s_opacity = 1, s_smooth = 1;
static int s_tex = 2;        /* textures: 0 = sharp, 1 = smooth, 2 = HD pack (applies at start; smooth where the pack has nothing) */
static int s_tex_boot = -1;
static int s_res = 0;        /* internal resolution: 0 = the game default, else 1..6 x */
static int s_aa = 0, s_aa_boot = -1;   /* MSAA: 0 off, 1 = 2x, 2 = 4x (applies at start) */
static int s_depth = 0, s_depth_boot = -1;   /* depth sorting (applies at start) */
static int s_metal = 0, s_metal_boot = -1;   /* graphics: 0 MoltenVK, 1 Metal (applies at start) */
static int s_res_boot = -1;  /* value this run started with */
static int s_has_gamepad;
static uint16_t s_buttons = 0xFFFFu;
static float s_aspect = 2868.0f / 1320.0f;
static int s_dirty = 1;
static unsigned s_version = 1;
static uint32_t *s_px;
static int s_w, s_h;
static int s_fps, s_fps_frames;
static unsigned long long s_fps_t0;
static char s_cfg_path[512] = "touch_ui.ini";

/* ---- settings file ------------------------------------------------------ */
static void cfg_load(void) {
    FILE *f = fopen(s_cfg_path, "r");
    if (!f) return;
    char line[128];
    while (fgets(line, sizeof line, f)) {
        int v;
        if (sscanf(line, "show_fps=%d", &v) == 1) s_show_fps = v != 0;
        else if (sscanf(line, "fit_screen=%d", &v) == 1) s_fit = v != 0;
        else if (sscanf(line, "touch_mode=%d", &v) == 1) s_mode = v < 0 || v > 2 ? 0 : v;
        else if (sscanf(line, "opacity=%d", &v) == 1) s_opacity = v < 0 || v > 2 ? 1 : v;
        else if (sscanf(line, "smooth_60=%d", &v) == 1) s_smooth = v != 0;
        else if (sscanf(line, "frame_rate=%d", &v) == 1) s_smooth = v == 2 ? 3 : (v < 0 || v > 2 ? 1 : v);
        else if (sscanf(line, "fps_mode=%d", &v) == 1) s_smooth = v < 0 || v > 3 ? 1 : v;
        else if (sscanf(line, "tex_smooth=%d", &v) == 1) s_tex = v != 0;
        else if (sscanf(line, "textures=%d", &v) == 1) s_tex = v < 0 || v > 2 ? 2 : v;
        else if (sscanf(line, "msaa=%d", &v) == 1) s_aa = v < 0 || v > 2 ? 0 : v;
        else if (sscanf(line, "depth_sort=%d", &v) == 1) s_depth = v != 0;
        else if (sscanf(line, "metal=%d", &v) == 1) s_metal = v != 0;
        else if (sscanf(line, "resolution=%d", &v) == 1) s_res = v < 0 || v > 8 ? 0 : v;
    }
    fclose(f);
}

static void cfg_save(void) {
    FILE *f = fopen(s_cfg_path, "w");
    if (!f) return;
    fprintf(f, "show_fps=%d\nfit_screen=%d\ntouch_mode=%d\nopacity=%d\nfps_mode=%d\nresolution=%d\ntextures=%d\nmsaa=%d\ndepth_sort=%d\nmetal=%d\n",
            s_show_fps, s_fit, s_mode, s_opacity, s_smooth, s_res, s_tex, s_aa, s_depth, s_metal);
    fclose(f);
}

/* Metal that does not get through its first pictures is left off at the next
 * start: the marker is there while it starts and gone once it has run. */
static void metal_marker(char *out, size_t n) { snprintf(out, n, "%s.metal-starting", s_cfg_path); }
static void metal_boot(void) {
    char m[560]; metal_marker(m, sizeof m);
    FILE *f = fopen(m, "r");
    if (f) {
        fclose(f); remove(m);
        if (s_metal) { s_metal = 0; cfg_save(); fprintf(stdout, "psxrecomp: metal did not get through its last start: off\n"); }
    }
    s_metal_boot = s_metal;
    if (s_metal_boot && (f = fopen(m, "w"))) { fputs("1\n", f); fclose(f); }
}

static void ui_init(void) {
    if (s_init) return;
    s_init = 1;
#ifdef PSX_IOS
    s_enabled = 1;
#endif
    const char *e = getenv("PSX_TOUCH_UI");
    if (e && *e) s_enabled = *e != '0';
    if (!s_enabled) return;
    e = getenv("PSX_TOUCH_UI_FILE");
    if (e && *e) snprintf(s_cfg_path, sizeof s_cfg_path, "%s", e);
    s_mx = SDL_CreateMutex();
    cfg_load();
    s_res_boot = s_res;
    s_aa_boot = s_aa;
    s_tex_boot = s_tex;
    s_depth_boot = s_depth;
    metal_boot();
    s_has_gamepad = SDL_HasGamepad() ? 1 : 0;
    fprintf(stdout, "psxrecomp: touch ui on (fps %d, fit %d, touch mode %d, opacity %d, gamepad %d)\n",
            s_show_fps, s_fit, s_mode, s_opacity, s_has_gamepad);
}

int touch_ui_enabled(void) { ui_init(); return s_enabled; }
int touch_ui_fit_screen(void) { ui_init(); return s_enabled && s_fit; }
/* 0 = the game's own 30, 1 = 60 (one in-between picture), 2 = 90 (two), 3 = 120 (three). */
int touch_ui_framegen(void) { ui_init(); return s_enabled ? s_smooth : 0; }
int touch_ui_tex_smooth(void) { ui_init(); return s_enabled && s_tex; }
/* 1 = load the HD texture pack this run (also 1 when the touch UI is off, so
 * scripted desktop runs follow the environment alone). */
int touch_ui_hd_pack(void) { ui_init(); return s_enabled ? s_tex_boot == 2 : 1; }
int touch_ui_msaa(void) { ui_init(); return s_enabled ? (s_aa_boot == 2 ? 4 : s_aa_boot == 1 ? 2 : 1) : 1; }
int touch_ui_depth_sort(void) { ui_init(); return s_enabled && s_depth_boot == 1; }
int touch_ui_metal(void) { ui_init(); return s_enabled && s_metal_boot == 1; }
void touch_ui_metal_ran(void) { char m[560]; metal_marker(m, sizeof m); remove(m); }
int touch_ui_resolution(void) { ui_init(); return s_enabled ? s_res_boot : 0; }

/* ---- layout ------------------------------------------------------------- */
#define DPAD_CX 32.0f
#define DPAD_CY 68.0f
#define DPAD_R  17.0f     /* thumbstick base ring */
#define NUB_R   8.5f      /* the part that follows the thumb */
#define NUB_MAX 11.0f     /* how far the nub travels from the centre */
static float s_nub_x, s_nub_y;   /* nub offset from the base centre, in u */
#define COG_CY  5.0f
#define COG_R   3.4f
#define PANEL_HW 55.0f
#define PANEL_Y0 9.0f
#define ROW_Y0   (PANEL_Y0 + 9.0f)
#define ROW_STEP 7.0f
#define ROW_H    6.3f

static float wu(void) { return 100.0f * s_aspect; }

static void layout(Ctl c[CTL_COUNT]) {
    const float W = wu();
    const float fx = W - 32.0f, fy = 66.0f, fo = 12.0f, fr = 7.5f;
    Ctl t[CTL_COUNT] = {
        { 0, fx,      fy - fo, fr, 0, 0, B_TRI,    NULL },
        { 0, fx + fo, fy,      fr, 0, 0, B_CIRCLE, NULL },
        { 0, fx,      fy + fo, fr, 0, 0, B_CROSS,  NULL },
        { 0, fx - fo, fy,      fr, 0, 0, B_SQUARE, NULL },
        { 1, 22.0f,     12.0f, 0, 9.0f, 5.0f, B_L2, "L2" },
        { 1, 43.0f,     12.0f, 0, 9.0f, 5.0f, B_L1, "L1" },
        { 1, W - 43.0f, 12.0f, 0, 9.0f, 5.0f, B_R1, "R1" },
        { 1, W - 22.0f, 12.0f, 0, 9.0f, 5.0f, B_R2, "R2" },
        { 1, W * 0.5f - 11.0f, 89.0f, 0, 8.5f, 3.8f, B_SELECT, "SELECT" },
        { 1, W * 0.5f + 11.0f, 89.0f, 0, 8.5f, 3.8f, B_START,  "START" },
    };
    memcpy(c, t, sizeof t);
}

static int pad_active(void) {
    return s_mode == MODE_ALWAYS || (s_mode == MODE_AUTO && !s_has_gamepad);
}
static int pad_visible(void) { return pad_active() && !s_menu_open; }

static void recompute_pad(void) {
    uint16_t b = 0xFFFFu;
    if (pad_visible()) {
        Ctl c[CTL_COUNT];
        layout(c);
        float nubx = 0.0f, nuby = 0.0f;
        for (int i = 0; i < MAX_FINGERS; i++) {
            if (!s_f[i].used || s_f[i].ui == 1) continue;
            const float x = s_f[i].x * wu(), y = s_f[i].y * 100.0f;
            float dx = x - DPAD_CX, dy = y - DPAD_CY;
            float d = sqrtf(dx * dx + dy * dy);
            if (s_f[i].ui == 2) {
                /* The thumbstick: this finger owns it until it lifts, however
                 * far it drifts. Eight directions, small dead zone. */
                if (d > 3.5f) {
                    if (dx >  0.38f * d) b &= (uint16_t)~B_RIGHT;
                    if (dx < -0.38f * d) b &= (uint16_t)~B_LEFT;
                    if (dy >  0.38f * d) b &= (uint16_t)~B_DOWN;
                    if (dy < -0.38f * d) b &= (uint16_t)~B_UP;
                }
                const float k = d > NUB_MAX ? NUB_MAX / d : 1.0f;
                nubx = dx * k; nuby = dy * k;
                continue;
            }
            for (int k = 0; k < CTL_COUNT; k++) {
                dx = x - c[k].cx; dy = y - c[k].cy;
                int hit = c[k].kind == 0
                    ? (dx * dx + dy * dy) < (c[k].r * 1.4f) * (c[k].r * 1.4f)
                    : fabsf(dx) < c[k].hw * 1.2f && fabsf(dy) < c[k].hh * 1.6f;
                if (hit) b &= (uint16_t)~c[k].bit;
            }
        }
        if (fabsf(nubx - s_nub_x) > 0.25f || fabsf(nuby - s_nub_y) > 0.25f ||
            ((nubx == 0.0f && nuby == 0.0f) && (s_nub_x != 0.0f || s_nub_y != 0.0f))) {
            s_nub_x = nubx; s_nub_y = nuby; s_dirty = 1;
        }
    }
    if (b != s_buttons) { s_buttons = b; s_dirty = 1; }
}

static void menu_tap(float x, float y) {
    const float W = wu();
    const float px0 = W * 0.5f - PANEL_HW, px1 = W * 0.5f + PANEL_HW;
    const float py1 = ROW_Y0 + ROW_STEP * ROW_COUNT + 2.0f;
    if (x < px0 || x > px1 || y < PANEL_Y0 || y > py1) {
        s_menu_open = 0; s_dirty = 1;
        return;
    }
    for (int r = 0; r < ROW_COUNT; r++) {
        const float y0 = ROW_Y0 + ROW_STEP * r;
        if (y < y0 || y > y0 + ROW_H) continue;
        switch (r) {
        case ROW_SMOOTH:  s_smooth = (s_smooth + 1) % 4; break;
        case ROW_TEX:     s_tex = (s_tex + 1) % 3; break;
        case ROW_AA:      s_aa = (s_aa + 1) % 3; break;
        case ROW_DEPTH:   s_depth = !s_depth; break;
        case ROW_GFX:     s_metal = !s_metal; break;
        case ROW_FPS:     s_show_fps = !s_show_fps; break;
        case ROW_FIT:     s_fit = !s_fit; break;
        case ROW_RES:     s_res = (s_res + 1) % 9; break;
        case ROW_TOUCH:   s_mode = (s_mode + 1) % 3; break;
        case ROW_OPACITY: s_opacity = (s_opacity + 1) % 3; break;
        case ROW_CLOSE:   s_menu_open = 0; break;
        }
        if (r != ROW_CLOSE) cfg_save();
        s_dirty = 1;
        return;
    }
}

static void finger(long long id, int phase, float nx, float ny) {
    int slot = -1;
    for (int i = 0; i < MAX_FINGERS; i++)
        if (s_f[i].used && s_f[i].id == id) { slot = i; break; }
    if (phase == 0) {
        if (slot < 0)
            for (int i = 0; i < MAX_FINGERS; i++)
                if (!s_f[i].used) { slot = i; break; }
        if (slot < 0) return;
        s_f[slot].used = 1; s_f[slot].id = id;
        s_f[slot].x = nx; s_f[slot].y = ny; s_f[slot].ui = 0;
        const float x = nx * wu(), y = ny * 100.0f;
        if (s_menu_open) {
            s_f[slot].ui = 1;
            menu_tap(x, y);
        } else {
            const float dx = x - wu() * 0.5f, dy = y - COG_CY;
            if (dx * dx + dy * dy < 7.0f * 7.0f) {
                s_f[slot].ui = 1;
                s_menu_open = 1; s_dirty = 1;
            } else if (pad_visible()) {
                const float sx = x - DPAD_CX, sy = y - DPAD_CY;
                int taken = 0;
                for (int i = 0; i < MAX_FINGERS; i++) taken |= (i != slot && s_f[i].used && s_f[i].ui == 2);
                if (!taken && sx * sx + sy * sy < (DPAD_R * 1.7f) * (DPAD_R * 1.7f)) s_f[slot].ui = 2;
            }
        }
    } else if (slot >= 0) {
        if (phase == 1) { s_f[slot].x = nx; s_f[slot].y = ny; }
        else s_f[slot].used = 0;
    }
    recompute_pad();
}

void touch_ui_inject(int f, int phase, float x, float y) {
    if (!touch_ui_enabled()) return;
    SDL_LockMutex(s_mx);
    finger(1000 + f, phase, x, y);
    SDL_UnlockMutex(s_mx);
}

int touch_ui_event(const void *sdl_event) {
    if (!sdl_event || !touch_ui_enabled()) return 0;
    const SDL_Event *e = (const SDL_Event *)sdl_event;
    int phase;
    switch (e->type) {
    case SDL_EVENT_FINGER_DOWN:     phase = 0; break;
    case SDL_EVENT_FINGER_MOTION:   phase = 1; break;
    case SDL_EVENT_FINGER_UP:
    case SDL_EVENT_FINGER_CANCELED: phase = 2; break;
    case SDL_EVENT_GAMEPAD_ADDED:
    case SDL_EVENT_GAMEPAD_REMOVED: {
        /* The removed pad is still listed while its event is delivered. */
        int n = 0;
        SDL_JoystickID *ids = SDL_GetGamepads(&n);
        if (ids) SDL_free(ids);
        if (e->type == SDL_EVENT_GAMEPAD_REMOVED && n > 0) {
            int still = 0;
            ids = SDL_GetGamepads(&n);
            for (int i = 0; ids && i < n; i++) if (ids[i] != e->gdevice.which) still++;
            if (ids) SDL_free(ids);
            n = still;
        }
        SDL_LockMutex(s_mx);
        if ((n > 0) != s_has_gamepad) {
            s_has_gamepad = n > 0;
            s_dirty = 1;
            recompute_pad();
            fprintf(stdout, "psxrecomp: touch ui: controller %s\n",
                    s_has_gamepad ? "connected, touch pad hidden" : "gone, touch pad shown");
        }
        SDL_UnlockMutex(s_mx);
        return 0;
    }
    default: return 0;
    }
    SDL_LockMutex(s_mx);
    finger((long long)e->tfinger.fingerID, phase, e->tfinger.x, e->tfinger.y);
    SDL_UnlockMutex(s_mx);
    return 1;
}

int touch_ui_get_pad(uint16_t *buttons, uint8_t st[4]) {
    if (!touch_ui_enabled()) return 0;
    SDL_LockMutex(s_mx);
    const int on = pad_active();
    const uint16_t b = s_menu_open ? 0xFFFFu : s_buttons;
    SDL_UnlockMutex(s_mx);
    if (!on) return 0;
    if (buttons) *buttons = b;
    if (st) st[0] = st[1] = st[2] = st[3] = 0x80;
    return 1;
}

void touch_ui_presented(void) {
    if (!touch_ui_enabled()) return;
    const unsigned long long now = SDL_GetTicks();
    if (!s_fps_t0) s_fps_t0 = now;
    s_fps_frames++;
    if (now - s_fps_t0 >= 1000) {
        const int fps = (int)((s_fps_frames * 1000ull + (now - s_fps_t0) / 2) / (now - s_fps_t0));
        s_fps_frames = 0; s_fps_t0 = now;
        SDL_LockMutex(s_mx);
        if (fps != s_fps) { s_fps = fps; if (s_show_fps) s_dirty = 1; }
        SDL_UnlockMutex(s_mx);
    }
}

/* ---- drawing (premultiplied ARGB8888) ----------------------------------- */
static float s_k;   /* canvas pixels per u */

static inline void px_over(int x, int y, uint32_t rgb, float a) {
    if (a <= 0.0f || x < 0 || y < 0 || x >= s_w || y >= s_h) return;
    if (a > 1.0f) a = 1.0f;
    const unsigned a8 = (unsigned)(a * 255.0f + 0.5f), inv = 255u - a8;
    uint32_t *d = &s_px[(size_t)y * (size_t)s_w + (size_t)x];
    const uint32_t o = *d;
    const unsigned r = (((rgb >> 16) & 255u) * a8 + ((o >> 16) & 255u) * inv + 127u) / 255u;
    const unsigned g = (((rgb >> 8) & 255u) * a8 + ((o >> 8) & 255u) * inv + 127u) / 255u;
    const unsigned b = ((rgb & 255u) * a8 + (o & 255u) * inv + 127u) / 255u;
    const unsigned al = a8 + ((o >> 24) * inv + 127u) / 255u;
    *d = (al > 255u ? 255u : al) << 24 | r << 16 | g << 8 | b;
}

static inline float cov(float d) { return d <= -0.5f ? 1.0f : d >= 0.5f ? 0.0f : 0.5f - d; }

/* Filled disc / ring; coordinates in u. thick <= 0 fills. */
static void circle(float cx, float cy, float r, float thick, uint32_t rgb, float a) {
    cx *= s_k; cy *= s_k; r *= s_k; thick *= s_k;
    const int x0 = (int)(cx - r - thick - 2), x1 = (int)(cx + r + thick + 2);
    const int y0 = (int)(cy - r - thick - 2), y1 = (int)(cy + r + thick + 2);
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            const float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
            float d = sqrtf(dx * dx + dy * dy) - r;
            if (thick > 0.0f) d = fabsf(d) - thick * 0.5f;
            px_over(x, y, rgb, a * cov(d));
        }
}

/* Rounded rectangle centred at (cx,cy) with half size (hw,hh). */
static void rrect(float cx, float cy, float hw, float hh, float rad, float thick,
                  uint32_t rgb, float a) {
    cx *= s_k; cy *= s_k; hw *= s_k; hh *= s_k; rad *= s_k; thick *= s_k;
    const int x0 = (int)(cx - hw - thick - 2), x1 = (int)(cx + hw + thick + 2);
    const int y0 = (int)(cy - hh - thick - 2), y1 = (int)(cy + hh + thick + 2);
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            const float qx = fabsf(x + 0.5f - cx) - (hw - rad);
            const float qy = fabsf(y + 0.5f - cy) - (hh - rad);
            const float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
            const float in = qx > qy ? qx : qy;
            float d = sqrtf(ox * ox + oy * oy) + (in < 0 ? in : 0) - rad;
            if (thick > 0.0f) d = fabsf(d) - thick * 0.5f;
            px_over(x, y, rgb, a * cov(d));
        }
}

/* Line segment with round caps. */
static void seg(float ax, float ay, float bx, float by, float thick, uint32_t rgb, float a) {
    ax *= s_k; ay *= s_k; bx *= s_k; by *= s_k; thick *= s_k;
    const float h = thick * 0.5f;
    const int x0 = (int)((ax < bx ? ax : bx) - h - 2), x1 = (int)((ax > bx ? ax : bx) + h + 2);
    const int y0 = (int)((ay < by ? ay : by) - h - 2), y1 = (int)((ay > by ? ay : by) + h + 2);
    const float vx = bx - ax, vy = by - ay;
    const float len2 = vx * vx + vy * vy;
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            const float px = x + 0.5f - ax, py = y + 0.5f - ay;
            float t = len2 > 0 ? (px * vx + py * vy) / len2 : 0;
            t = t < 0 ? 0 : t > 1 ? 1 : t;
            const float dx = px - vx * t, dy = py - vy * t;
            px_over(x, y, rgb, a * cov(sqrtf(dx * dx + dy * dy) - h));
        }
}

static int text_scale(void) { int s = (int)(s_h / 220.0f + 0.5f); return s < 1 ? 1 : s; }

/* Text with its left edge at x (u) and vertically centred on cy (u). */
static void text_at(float x, float cy, const char *s, int scale, uint32_t rgb, float a) {
    int px = (int)(x * s_k), py = (int)(cy * s_k) - 4 * scale;
    for (; *s; s++, px += 8 * scale) {
        const unsigned ch = (unsigned char)*s;
        if (ch < 32 || ch > 126) continue;
        const uint8_t *g = PSX_FONT8X8[ch - 32];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                if (g[r] & (1u << c))
                    for (int yy = 0; yy < scale; yy++)
                        for (int xx = 0; xx < scale; xx++)
                            px_over(px + c * scale + xx, py + r * scale + yy, rgb, a);
    }
}
static float text_w(const char *s, int scale) { return (float)strlen(s) * 8.0f * scale / s_k; }
static void text_mid(float cx, float cy, const char *s, int scale, uint32_t rgb, float a) {
    text_at(cx - text_w(s, scale) * 0.5f, cy, s, scale, rgb, a);
}
static void text_right(float x1, float cy, const char *s, int scale, uint32_t rgb, float a) {
    text_at(x1 - text_w(s, scale), cy, s, scale, rgb, a);
}

static void draw_cog(float cx, float cy, float a) {
    circle(cx, cy, COG_R + 1.4f, 0, 0x000000, 0.28f * a);
    for (int i = 0; i < 8; i++) {
        const float t = (float)i * 0.78539816f;
        const float c = cosf(t), s = sinf(t);
        seg(cx + c * 1.9f, cy + s * 1.9f, cx + c * COG_R, cy + s * COG_R, 1.15f, 0xFFFFFF, a);
    }
    circle(cx, cy, 1.9f, 1.3f, 0xFFFFFF, a);
}

static void draw_pad(void) {
    static const float kAlpha[3] = { 0.20f, 0.36f, 0.58f };
    const float a = kAlpha[s_opacity];
    const uint16_t held = (uint16_t)~s_buttons;
    Ctl c[CTL_COUNT];
    layout(c);

    /* Thumbstick: a base ring and a nub that follows the thumb. */
    circle(DPAD_CX, DPAD_CY, DPAD_R, 0, 0x000000, a * 0.40f);   /* dark backing: readable on bright scenes */
    circle(DPAD_CX, DPAD_CY, DPAD_R, 0, 0xFFFFFF, a * 0.16f);
    circle(DPAD_CX, DPAD_CY, DPAD_R, 0.5f, 0xFFFFFF, a);
    {
        const int moving = (held & (B_UP | B_DOWN | B_LEFT | B_RIGHT)) != 0;
        const float nx = DPAD_CX + s_nub_x, ny = DPAD_CY + s_nub_y;
        circle(nx, ny, NUB_R, 0, 0x000000, a * 0.55f);
        circle(nx, ny, NUB_R, 0, 0xFFFFFF, moving ? 0.55f : a * 0.9f);
        circle(nx, ny, NUB_R, 0.5f, 0xFFFFFF, moving ? 1.0f : a * 1.5f);
        circle(nx, ny, NUB_R * 0.45f, 0.4f, 0xFFFFFF, a * 1.2f);
    }

    for (int i = 0; i < CTL_COUNT; i++) {
        const int on = (held & c[i].bit) != 0;
        if (c[i].kind == 1) {
            rrect(c[i].cx, c[i].cy, c[i].hw, c[i].hh, c[i].hh, 0, 0x000000, a * 0.45f);
            rrect(c[i].cx, c[i].cy, c[i].hw, c[i].hh, c[i].hh, 0, 0xFFFFFF, on ? 0.50f : a * 0.22f);
            rrect(c[i].cx, c[i].cy, c[i].hw, c[i].hh, c[i].hh, 0.5f, 0xFFFFFF, a);
            int sc = text_scale();
            if (strlen(c[i].label) > 2 && sc > 1) sc--;
            text_mid(c[i].cx, c[i].cy, c[i].label, sc, 0xFFFFFF, on ? 1.0f : a * 1.6f);
            continue;
        }
        const float x = c[i].cx, y = c[i].cy, r = c[i].r;
        circle(x, y, r, 0, 0x000000, a * 0.45f);
        circle(x, y, r, 0, 0xFFFFFF, on ? 0.50f : a * 0.22f);
        circle(x, y, r, 0.55f, 0xFFFFFF, a);
        const float al = on ? 1.0f : a * 1.7f, g = r * 0.42f, t = 1.0f;
        if (c[i].bit == B_TRI) {
            const uint32_t col = 0x40E8B8;
            seg(x, y - g * 1.15f, x + g * 1.2f, y + g * 0.85f, t, col, al);
            seg(x + g * 1.2f, y + g * 0.85f, x - g * 1.2f, y + g * 0.85f, t, col, al);
            seg(x - g * 1.2f, y + g * 0.85f, x, y - g * 1.15f, t, col, al);
        } else if (c[i].bit == B_CIRCLE) {
            circle(x, y, g * 1.1f, t, 0xFF6868, al);
        } else if (c[i].bit == B_CROSS) {
            seg(x - g, y - g, x + g, y + g, t, 0x88AAFF, al);
            seg(x - g, y + g, x + g, y - g, t, 0x88AAFF, al);
        } else {
            rrect(x, y, g, g, 0.3f, t, 0xFF98D8, al);
        }
    }
}

static void draw_menu(void) {
    const float W = wu(), cx = W * 0.5f;
    const float py1 = ROW_Y0 + ROW_STEP * ROW_COUNT + 2.0f;
    const float pcy = (PANEL_Y0 + py1) * 0.5f, phh = (py1 - PANEL_Y0) * 0.5f;
    const int sc = text_scale();
    rrect(cx, pcy, PANEL_HW, phh, 4.0f, 0, 0x141A22, 0.93f);
    rrect(cx, pcy, PANEL_HW, phh, 4.0f, 0.4f, 0xFFFFFF, 0.35f);
    text_mid(cx, PANEL_Y0 + 5.5f, "Settings", sc + 1, 0xFFFFFF, 1.0f);

    static const char *const kOnOff[2] = { "Off", "On" };
    static const char *const kMode[3] = { "Auto", "Always", "Off" };
    static const char *const kOpac[3] = { "Low", "Medium", "High" };
    static const char *const kRes[9] = { "Default", "1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x" };
    static const char *const kResNew[9] = { "Default (restart)", "1x (restart)", "2x (restart)", "3x (restart)",
                                            "4x (restart)", "5x (restart)", "6x (restart)", "7x (restart)", "8x (restart)" };
    static const char *const kRate[4] = { "30 (original)", "60", "90", "120" };
    static const char *const kTex[3] = { "Sharp", "Smooth", "HD pack" };
    static const char *const kTexNew[3] = { "Sharp (restart)", "Smooth (restart)", "HD pack (restart)" };
    static const char *const kAa[3] = { "Off", "MSAA 2x", "MSAA 4x" };
    static const char *const kAaNew[3] = { "Off (restart)", "MSAA 2x (restart)", "MSAA 4x (restart)" };
    static const char *const kDepth[4] = { "Off", "On", "Off (restart)", "On (restart)" };
    static const char *const kGfx[4] = { "Vulkan", "Metal", "Vulkan (restart)", "Metal (restart)" };
    const char *label[ROW_COUNT] = { "Frame rate", "Show FPS", "Fit to screen", "Resolution", "Anti-aliasing", "Textures", "Depth sorting", "Graphics", "Touch controls",
                                     "Button opacity", "Close" };
    const char *value[ROW_COUNT] = { kRate[s_smooth], kOnOff[s_show_fps], kOnOff[s_fit],
                                     s_res == s_res_boot ? kRes[s_res] : kResNew[s_res], s_aa == s_aa_boot ? kAa[s_aa] : kAaNew[s_aa],
                                     (s_tex == 2) != (s_tex_boot == 2) ? kTexNew[s_tex] : kTex[s_tex],
                                     kDepth[s_depth + (s_depth != s_depth_boot ? 2 : 0)], kGfx[s_metal + (s_metal != s_metal_boot ? 2 : 0)], kMode[s_mode],
                                     kOpac[s_opacity], NULL };
    for (int r = 0; r < ROW_COUNT; r++) {
        const float cy = ROW_Y0 + ROW_STEP * r + ROW_H * 0.5f;
        rrect(cx, cy, PANEL_HW - 4.0f, ROW_H * 0.5f, 2.5f, 0, 0xFFFFFF, r == ROW_CLOSE ? 0.16f : 0.08f);
        if (!value[r]) { text_mid(cx, cy, label[r], sc, 0xFFFFFF, 1.0f); continue; }
        text_at(cx - PANEL_HW + 8.0f, cy, label[r], sc, 0xFFFFFF, 0.95f);
        const int on = (r == ROW_SMOOTH && s_smooth) || (r == ROW_FPS && s_show_fps) || (r == ROW_FIT && s_fit) ||
                       (r == ROW_TOUCH && s_mode != MODE_OFF) || r == ROW_OPACITY || r == ROW_RES || (r == ROW_TEX && s_tex) || (r == ROW_AA && s_aa) || (r == ROW_DEPTH && s_depth) || (r == ROW_GFX && s_metal);
        text_right(cx + PANEL_HW - 8.0f, cy, value[r], sc, on ? 0xFFD060 : 0xA0A8B0, 1.0f);
    }
}

static void redraw(void) {
    memset(s_px, 0, (size_t)s_w * (size_t)s_h * 4u);
    s_k = (float)s_h / 100.0f;
    if (pad_visible()) draw_pad();
    if (s_show_fps) {
        char t[24];
        snprintf(t, sizeof t, "%d FPS", s_fps);
        const int sc = text_scale();
        const float w = text_w(t, sc);
        const float x = 16.0f, cy = pad_visible() ? 24.0f : 5.0f;
        rrect(x + w * 0.5f, cy, w * 0.5f + 1.6f, 2.9f, 1.4f, 0, 0x000000, 0.55f);
        text_at(x, cy, t, sc, 0x60FF90, 1.0f);
    }
    draw_cog(wu() * 0.5f, COG_CY, s_menu_open ? 1.0f : 0.55f);
    if (s_menu_open) draw_menu();
}

int touch_ui_canvas(int sc_w, int sc_h, const uint32_t **px, int *w, int *h,
                    unsigned *version) {
    if (!touch_ui_enabled() || sc_w < 64 || sc_h < 64) return 0;
    const int div = (sc_h + 719) / 720;
    const int cw = sc_w / div, ch = sc_h / div;
    SDL_LockMutex(s_mx);
    if (cw != s_w || ch != s_h || !s_px) {
        uint32_t *n = (uint32_t *)realloc(s_px, (size_t)cw * (size_t)ch * 4u);
        if (!n) { SDL_UnlockMutex(s_mx); return 0; }
        s_px = n; s_w = cw; s_h = ch;
        s_aspect = (float)sc_w / (float)sc_h;
        s_dirty = 1;
    }
    if (s_dirty) {
        s_dirty = 0;
        redraw();
        s_version++;
    }
    *px = s_px; *w = s_w; *h = s_h; *version = s_version;
    SDL_UnlockMutex(s_mx);
    return 1;
}

#endif /* PSX_SDL3 */
