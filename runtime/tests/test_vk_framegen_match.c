/* Frame-generation pairing (runtime/src/vk_framegen_match.h) against scenes
 * whose motion is known. Each scene is projected for two consecutive frames
 * the way the game submits it: back faces dropped, sorted by depth, look-alike
 * pieces everywhere. The generated picture must put every primitive half way
 * between its two true positions, and corners shared by two primitives must
 * stay together (a corner that does not is a visible hole).
 *
 * Build: cc -O1 -I runtime/src runtime/tests/test_vk_framegen_match.c -lm */
#ifdef FG_BASELINE
#include FG_BASELINE          /* compare another implementation (developer use) */
#else
#include "vk_framegen_match.h"
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct { double x, y, z; } V3;
typedef struct { V3 p; double yaw, pitch; } Cam;
typedef struct { V3 v[3]; int uv[6], tp, two_sided, group, obj, vid[3]; double z0; } Tri;
#define MAXT 12000
static Tri T[MAXT]; static int tn;
static int next_vid = 1;
static unsigned rs = 12345;
static double rnd(void) { rs = rs * 1664525u + 1013904223u; return (double)(rs >> 8) / 16777216.0; }

enum { WALL, GROUND, ROPE, AWNING, WATER, PILLAR, HERO, PICKUP, LEVER, HUD, GROUPS };
static const char *gname[GROUPS] = { "wall", "ground", "rope", "awning", "waterfall", "pillars", "hero", "pickups", "levers", "hud" };

static void quad(V3 a, V3 b, V3 c, V3 d, int va, int vb, int vc, int vd, int u, int v, int w, int h, int tp, int g, int obj, int two) {
    if (tn + 2 > MAXT) return;
    Tri t1 = { { a, b, c }, { u, v, u + w, v, u, v + h }, tp, two, g, obj, { va, vb, vc }, 0 };
    Tri t2 = { { c, b, d }, { u, v + h, u + w, v, u + w, v + h }, tp, two, g, obj, { vc, vb, vd }, 0 };
    T[tn++] = t1; T[tn++] = t2;
}
/* nx * ny quads over a parallelogram; every quad shows the same cut-out when
 * looks == 1, else one of `looks`. */
static void sheet(V3 o, V3 ex, V3 ey, int nx, int ny, int looks, int u, int v, int w, int h, int tp, int g, int obj, int two) {
    const int base = next_vid; next_vid += (nx + 1) * (ny + 1);
    for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
        V3 q[4]; int id[4];
        for (int k = 0; k < 4; k++) {
            const double s = (double)(i + (k & 1)) / nx, t = (double)(j + (k >> 1)) / ny;
            q[k].x = o.x + ex.x * s + ey.x * t; q[k].y = o.y + ex.y * s + ey.y * t; q[k].z = o.z + ex.z * s + ey.z * t;
            id[k] = base + (j + (k >> 1)) * (nx + 1) + i + (k & 1);
        }
        const int look = looks > 1 ? (int)(rnd() * looks) : 0;
        quad(q[0], q[1], q[2], q[3], id[0], id[1], id[2], id[3], u, v + look * 8, w, h, tp, g, obj, two);
    }
}
static V3 vec(double x, double y, double z) { V3 r = { x, y, z }; return r; }
static void build(void) {
    tn = 0; next_vid = 1; rs = 12345;
    sheet(vec(-2400, -700, 2600), vec(4800, 0, 0), vec(0, 1400, 0), 30, 9, 1, 0, 0, 32, 32, 1, WALL, 0, 1);
    sheet(vec(-2400, 700, 400), vec(4800, 0, 0), vec(0, 0, 2400), 20, 12, 6, 64, 0, 32, 8, 2, GROUND, 0, 1);
    sheet(vec(-40, -1200, 900), vec(30, 0, 0), vec(0, 1700, 0), 1, 26, 1, 0, 64, 16, 16, 3, ROPE, 0, 1);
    sheet(vec(-10, -1200, 915), vec(0, 0, 30), vec(0, 1700, 0), 1, 26, 1, 0, 64, 16, 16, 3, ROPE, 0, 1);
    sheet(vec(-900, -300, 1300), vec(500, 0, 60), vec(0, 60, -160), 12, 2, 1, 32, 64, 16, 32, 4, AWNING, 0, 1);
    sheet(vec(500, -900, 2300), vec(700, 0, 0), vec(0, 1600, 0), 7, 16, 1, 0, 128, 32, 16, 5, WATER, 0, 1);
    for (int k = 0; k < 4; k++) {
        const double px = -700 + 420 * k, pz = 1500 + 150 * k;
        sheet(vec(px, -800, pz), vec(90, 0, 0), vec(0, 1500, 0), 1, 14, 1, 64, 64, 16, 16, 6, PILLAR, 0, 1);
        sheet(vec(px + 90, -800, pz), vec(0, 0, 90), vec(0, 1500, 0), 1, 14, 1, 64, 64, 16, 16, 6, PILLAR, 0, 1);
    }
    for (int k = 0; k < 24; k++)
        sheet(vec(-60 + 25 * (k % 6), 100 + 40 * (k / 6), 1000 - 8 * (k % 3)), vec(26, 0, 0), vec(0, 42, 0), 1, 1, 1, k * 8, 200, 8, 8, 7, HERO, 1, 1);
    for (int k = 0; k < 5; k++)
        sheet(vec(-700 + 300 * k, 300, 1100 + 90 * k), vec(50, 0, 0), vec(0, 50, 0), 2, 2, 1, 0, 220, 16, 16, 8, PICKUP, 2 + k, 1);
    /* Identical levers one behind the other. Their look follows the distance in
     * steps (a coarser model, a darker palette), so in the frame a step is
     * crossed each looks exactly as its neighbour did one frame before. */
    for (int k = 0; k < 8; k++) {
        const int from = tn;
        sheet(vec(-420 + 150 * (k & 1), -260 + 88 * k, 1330 + 100 * k), vec(116, 0, 0), vec(0, 26, 0), 4, 1, 1, 0, 160, 8, 16, 40, LEVER, 0, 1);
        sheet(vec(-300 + 150 * (k & 1), -300 + 88 * k, 1330 + 100 * k), vec(24, 0, 0), vec(0, 66, 0), 1, 3, 1, 32, 160, 8, 8, 40, LEVER, 0, 1);
        for (int i = from; i < tn; i++) T[i].z0 = 1330 + 100 * k;
    }
}

typedef struct { float x, y; } P2;
static int project(V3 p, const Cam *c, P2 *o, double *depth) {
    const double X = p.x - c->p.x, Y = p.y - c->p.y, Z = p.z - c->p.z;
    const double x1 = X * cos(c->yaw) - Z * sin(c->yaw), z1 = X * sin(c->yaw) + Z * cos(c->yaw);
    const double y2 = Y * cos(c->pitch) - z1 * sin(c->pitch), z2 = Y * sin(c->pitch) + z1 * cos(c->pitch);
    *depth = z2;
    if (z2 < 60) return 0;
    o->x = (float)(160.0 + x1 * 350.0 / z2); o->y = (float)(120.0 + y2 * 350.0 / z2);
    return 1;
}
static V3 animate(const Tri *t, int v, double time) {
    V3 p = t->v[v];
    if (t->obj == 1) { p.x += 70 * time; p.y -= 18 * time; }
    if (t->obj >= 2) { p.y += 14 * sin(time * 0.9 + t->obj); p.x += 6 * sin(time + 2 * t->obj); }
    return p;
}
typedef struct { int tri; } Meta;
static FgCmd bufA[FG_MAX_CMDS], bufB[FG_MAX_CMDS];
static Meta metaA[FG_MAX_CMDS], metaB[FG_MAX_CMDS];
typedef struct { int t; double z; } Sort;
static int by_depth(const void *a, const void *b) {
    const double x = ((const Sort *)a)->z, y = ((const Sort *)b)->z;
    return x < y ? 1 : x > y ? -1 : ((const Sort *)a)->t - ((const Sort *)b)->t;
}
static int with_depth;
static void put(FgList *l, Meta *m, int tri, const P2 *q, const double *z, const int *uv, int tp, int precise, int buffer, int scroll) {
    FgCmd *c = &l->c[l->n]; memset(c, 0, sizeof *c);
    c->op = FG_STT; c->next = -1; c->offx = 0; c->offy = (int16_t)(buffer ? 240 : 0);
    for (int v = 0; v < 3; v++) {
        c->a[v * 5] = (int32_t)floorf(q[v].x) + c->offx; c->a[v * 5 + 1] = (int32_t)floorf(q[v].y) + c->offy;
        c->a[v * 5 + 2] = uv[v * 2]; c->a[v * 5 + 3] = uv[v * 2 + 1] + scroll; c->a[v * 5 + 4] = (int32_t)(rnd() * 255);
        c->pcx[v] = q[v].x + (float)c->offx; c->pcy[v] = q[v].y + (float)c->offy;
#ifndef FG_BASELINE
        if (z && with_depth) c->iz[v] = 1.0f / (float)floor(z[v] + 0.5);   /* GTE SZ is an integer */
#endif
    }
    (void)z;
    c->pc_valid = (uint8_t)precise; c->a[15] = tp; c->a[16] = tp * 3; c->a[17] = tp;
    fg_set_keys(c);
    m[l->n].tri = tri; l->n++; l->prims++; l->wide_prims++;
}
static void render(FgList *l, Meta *m, const Cam *c, double time, int precise, int buffer, int scroll) {
    static Sort s[MAXT]; int n = 0;
    l->n = l->prims = l->wide_prims = 0;
    for (int i = 0; i < tn; i++) {
        P2 q[3]; double d[3], z = 0; int ok = 1;
        for (int v = 0; v < 3; v++) { ok &= project(animate(&T[i], v, time), c, &q[v], &d[v]); z += d[v] / 3; }
        if (!ok) continue;
        float lx = 1e9f, hx = -1e9f, ly = 1e9f, hy = -1e9f;
        for (int v = 0; v < 3; v++) { lx = fminf(lx, q[v].x); hx = fmaxf(hx, q[v].x); ly = fminf(ly, q[v].y); hy = fmaxf(hy, q[v].y); }
        if (hx < -110 || lx > 430 || hy < 0 || ly > 240 || hx - lx > 1000) continue;
        s[n].t = i; s[n].z = floor(z / 8.0); n++;       /* ordering-table granularity */
    }
    qsort(s, (size_t)n, sizeof s[0], by_depth);
    for (int k = 0; k < n; k++) {
        const Tri *t = &T[s[k].t]; P2 q[3]; double d[3];
        for (int v = 0; v < 3; v++) project(animate(t, v, time), c, &q[v], &d[v]);
        const int level = t->group == LEVER ? (int)floor((t->z0 - c->p.z) / 100.0) : 0;
        put(l, m, s[k].t, q, d, t->uv, t->tp + level, precise, buffer, t->group == WATER ? scroll : 0);
    }
    for (int k = 0; k < 8; k++) {                        /* score digits: alike, fixed to the screen */
        const float x = 20.0f + 10.0f * (float)k; const int uv[6] = { 8 * (k % 3), 240, 8 * (k % 3) + 8, 240, 8 * (k % 3), 250 };
        P2 q[3] = { { x, 10 }, { x + 8, 10 }, { x, 20 } };
        put(l, m, -1 - k, q, NULL, uv, 9, precise, buffer, 0);
    }
}

typedef struct { int prims, broken, corners, torn, worst; double sum; int gp[GROUPS], gb[GROUPS]; } Stat;
static void measure(const FgList *cur, const Meta *m, const Cam *c0, double t0, Stat *st) {
    static float ex[FG_MAX_CMDS * 3], ey[FG_MAX_CMDS * 3]; static int first[200000];
    memset(first, 0xff, sizeof first);
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i]; const int tri = m[i].tri;
        float e = 0; int have = 1;
        for (int v = 0; v < 3; v++) {
            float x, y; fg_corner(c, v, &x, &y);
            const float gx = x - (c->moved ? c->vdx[v] : 0.0f) * 0.5f, gy = y - (c->moved ? c->vdy[v] : 0.0f) * 0.5f;
            ex[i * 3 + v] = gx; ey[i * 3 + v] = gy;
            float tx = x, ty = y;
            if (tri >= 0) {
                P2 q; double d;
                if (!project(animate(&T[tri], v, t0), c0, &q, &d)) { have = 0; break; }
                tx = (x + q.x) * 0.5f; ty = (y + q.y) * 0.5f;
            }
            const float dd = hypotf(gx - tx, gy - ty); if (dd > e) e = dd;
        }
        if (!have) continue;
        const int g = tri < 0 ? HUD : T[tri].group;
        st->prims++; st->gp[g]++; st->sum += e;
        if (e > 2.5f) { st->broken++; st->gb[g]++; }
        if ((int)e > st->worst) st->worst = (int)e;
        if (tri < 0) continue;
        for (int v = 0; v < 3; v++) {
            const int id = T[tri].vid[v];
            if (first[id] < 0) { first[id] = i * 3 + v; continue; }
            st->corners++;
            if (hypotf(ex[i * 3 + v] - ex[first[id]], ey[i * 3 + v] - ey[first[id]]) > 1.5f) st->torn++;
        }
    }
}

static Stat run(const char *name, Cam step, int precise, int scroll, int frames, int verbose) {
    FgList a = { bufA, 0, 0, 0, 0, { 0 } }, b = { bufB, 0, 0, 0, 0, { 0 } };
    Stat st; memset(&st, 0, sizeof st);
    for (int f = 0; f < frames; f++) {
        Cam c0 = { { -300 + step.p.x * f, -100 + step.p.y * f, -300 + step.p.z * f }, step.yaw * (f - frames / 2), step.pitch * (f - frames / 2) };
        Cam c1 = { { c0.p.x + step.p.x, c0.p.y + step.p.y, c0.p.z + step.p.z }, c0.yaw + step.yaw, c0.pitch + step.pitch };
        FgList *prev = f & 1 ? &b : &a, *cur = f & 1 ? &a : &b;
        Meta *mp = f & 1 ? metaB : metaA, *mc = f & 1 ? metaA : metaB;
        if (!f) render(prev, mp, &c0, f, precise, f & 1, scroll * f);     /* later it is the last `cur`, solved */
        render(cur, mc, &c1, f + 1, precise, !(f & 1), scroll * (f + 1));
        fg_solve(prev, cur);
        measure(cur, mc, &c0, f, &st);
    }
    if (verbose) {
        printf("%-28s %s  misplaced %5.2f%%  torn corners %5.2f%%  mean error %5.2f px  worst %3d px |", name, !with_depth ? "no depth" : precise ? "sub-px  " : "int     ",
               100.0 * st.broken / st.prims, 100.0 * st.torn / st.corners, st.sum / st.prims, st.worst);
        for (int g = 0; g < GROUPS; g++) if (st.gb[g]) printf(" %s %.0f%%", gname[g], 100.0 * st.gb[g] / st.gp[g]);
        printf("\n");
    }
    return st;
}

#ifndef FG_BASELINE
/* A tile grid scrolling by more than half a tile, with a few distinct tiles. */
static void test_backdrop(void) {
    FgList a = { bufA, 0, 0, 0, 0, { 0 } }, b = { bufB, 0, 0, 0, 0, { 0 } };
    for (int f = 0; f < 2; f++) {
        FgList *l = f ? &b : &a; l->n = 0;
        for (int y = -1; y < 9; y++) for (int x = -1; x < 12; x++) {
            FgCmd *c = &l->c[l->n++]; memset(c, 0, sizeof *c);
            c->op = FG_TR; c->bd = 1; c->next = -1; c->offy = (int16_t)(f ? 240 : 0);
            c->a[0] = x * 32 - f * 3; c->a[1] = y * 32 + f * 20 + c->offy; c->a[2] = c->a[3] = 32;
            c->a[4] = ((x * 7 + y * 13) % 9 == 0) ? 64 : 0; c->a[8] = 5;
            fg_set_keys(c);
        }
    }
    FgCmd *bar = &b.c[b.n++]; memset(bar, 0, sizeof *bar); bar->op = FG_FR; bar->bd = 2; bar->a[2] = 320; bar->a[3] = 20; bar->a[1] = 240; bar->offy = 240; fg_set_keys(bar);
    bar = &a.c[a.n++]; memset(bar, 0, sizeof *bar); bar->op = FG_FR; bar->bd = 2; bar->a[2] = 320; bar->a[3] = 20; fg_set_keys(bar);
    fg_solve(&a, &b);
    const int tiles = fg_solve_backdrop(&a, &b);
    assert(s_fg_bd_have && fabsf(s_fg_bd_dx + 3.0f) < 0.01f && fabsf(s_fg_bd_dy - 20.0f) < 0.01f);
    assert(tiles > 80 && b.c[b.n - 1].next >= 0);          /* letterbox bar: paired, did not vote */
}
/* A near face crossing the camera plane pairs by its homogeneous corners. */
static void test_projected(void) {
    FgList a = { bufA, 0, 0, 0, 0, { 0 } }, b = { bufB, 0, 0, 0, 0, { 0 } };
    for (int f = 0; f < 2; f++) {
        FgList *l = f ? &b : &a; l->n = 0;
        for (int k = 0; k < 4; k++) {
            FgCmd *c = &l->c[l->n++]; memset(c, 0, sizeof *c);
            c->op = FG_PROJ; c->next = -1; c->a[17] = 3;
            for (int v = 0; v < 3; v++) { c->pcx[v] = 9000.0f * (float)(k + (v & 1)) + 700.0f * (float)f; c->pcy[v] = 5000.0f * (float)v; c->pq[v] = (v == 2 ? -40.0f : 90.0f) - 6.0f * (float)f; }
            fg_set_keys(c);
        }
    }
    fg_solve(&a, &b);
    for (int k = 0; k < 4; k++) assert(b.c[k].next == k);
}
#endif

int main(int argc, char **argv) {
    const int verbose = argc > 1 || getenv("FG_VERBOSE") != NULL; (void)argv;
    build();
    static const struct { const char *n; Cam s; int scroll; } k[] = {
        { "still camera",             { { 0, 0, 0 }, 0, 0 }, 3 },
        { "slow pan",                 { { 12, 0, 0 }, 0, 0 }, 3 },
        { "run sideways",             { { 60, 0, 0 }, 0, 0 }, 4 },
        { "fall",                     { { 0, 70, 0 }, 0, 0 }, 5 },
        { "fast fall and run",        { { 55, 85, 10 }, 0, 0 }, 8 },
        { "turn",                     { { 0, 0, 0 }, 0.05, 0 }, 3 },
        { "turn and look down",       { { 10, 5, 0 }, 0.04, 0.03 }, 3 },
        { "orbit (strong parallax)",  { { 80, 0, 10 }, -0.035, 0 }, 4 },
        { "rush forward",             { { 5, 0, 70 }, 0.0, 0 }, 4 },
        { "walk in (looks change)",   { { 2, 0, 26 }, 0.0, 0 }, 4 },
        { "walk out (looks change)",  { { 2, 1, -26 }, 0.0, 0 }, 4 },
        { "bob (looks change)",       { { 3, 6, 18 }, 0.002, 0 }, 4 },
    };
    double worst_broken = 0, worst_torn = 0;
    double flat_broken = 0, flat_torn = 0, worst_lever = 0;   /* the levers apart from the rest of the scene */
    for (int mode = 0; mode < 3; mode++) {       /* sub-pixel and depth; whole pixels and depth; sub-pixel only */
        with_depth = mode < 2;
        for (unsigned i = 0; i < sizeof k / sizeof k[0]; i++) {
            const Stat s = run(k[i].n, k[i].s, mode != 1, k[i].scroll, 8, verbose);
            const double b = 100.0 * (s.broken - s.gb[LEVER]) / (s.prims - s.gp[LEVER]), t = 100.0 * s.torn / s.corners;
            const double lv = s.gp[LEVER] ? 100.0 * s.gb[LEVER] / s.gp[LEVER] : 0;
            if (strstr(k[i].n, "looks change") && lv > worst_lever) worst_lever = lv;
            double *wb = with_depth ? &worst_broken : &flat_broken, *wt = with_depth ? &worst_torn : &flat_torn;
            if (b > *wb) *wb = b;
            if (t > *wt) *wt = t;
        }
    }
    printf("worst scene: %.2f%% misplaced, %.2f%% torn corners; without depth %.2f%%, %.2f%%; look-alike levers %.2f%%\n",
           worst_broken, worst_torn, flat_broken, flat_torn, worst_lever);
#ifndef FG_BASELINE
    test_backdrop();
    test_projected();
    assert(worst_broken < 2.0);
    assert(worst_torn < 1.0);
    assert(worst_lever < 5.0);
    printf("Vulkan frame-generation pairing test passed\n");
#endif
    return 0;
}
