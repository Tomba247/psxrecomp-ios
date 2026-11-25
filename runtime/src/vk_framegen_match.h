/* vk_framegen_match.h - motion of every primitive between two guest frames.
 *
 * Frame generation draws the newest frame's primitives part of the way back to
 * where they were one frame earlier, so it needs each primitive's motion. The
 * draw lists carry no identity: a primitive is known only by what it draws
 * with (kind, texture page, CLUT, UVs), and world geometry repeats. A rope, a
 * row of planks or a waterfall is one texture cut-out drawn dozens of times.
 * Pairing each piece with the nearest piece of the same look picks a
 * neighbour as soon as the picture moves by more than half a piece; that one
 * piece then sits a piece's length away in the generated picture, leaving a
 * hole that flickers at half the display rate.
 *
 * What a wrong neighbour cannot fake is the mesh. Adjacent pieces share their
 * corners, and a corner has one motion, whichever piece reports it. So:
 *
 * and scenery obeys the camera: the screen motion of a still point follows from
 * its position, its depth and six numbers shared by the whole picture. So:
 *
 *   1. Camera: where corners carry depth, find the one camera motion under
 *      which the most scenery finds partners, to a fraction of a pixel.
 *      Meshes of different piece size and depth cannot all agree on a wrong
 *      one. Pieces that fit it are paired.
 *   2. Anchors: pieces that can only pair one way (a look that occurs once, or
 *      once within reach in both frames, or an exactly unmoved primitive).
 *   3. Growth: a piece touching corners of known motion pairs only with a
 *      same-look piece that gives those corners that same motion.
 *   4. Meshes still unpaired (no depth; moving objects made of look-alikes)
 *      are paired as a whole: the shift under which most of the mesh finds
 *      partners wins, the least ambiguous meshes first. Ties go to the motion
 *      of what is drawn near it; draw order stands in for depth, so the hint
 *      comes from scenery at a similar distance.
 *   5. Scrolled textures change UVs every frame; what is left repeats 1, 3 and
 *      4 with a look that leaves the UVs out.
 *   6. A piece with no partner (just appeared) moves with its corners'
 *      neighbours, else with the camera motion at its depth, else with the
 *      motion fitted around it. It never stays behind while the surface
 *      around it moves.
 *
 * Pure CPU and free of graphics API, so runtime/tests/test_vk_framegen_match.c
 * checks it against scenes with known motion.
 */
#ifndef PSX_VK_FRAMEGEN_MATCH_H
#define PSX_VK_FRAMEGEN_MATCH_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { FG_SEMI, FG_MASK, FG_TWIN, FG_MOD, FG_DA, FG_OFF, FG_VIEW, FG_WSET, FG_WDIS,
       FG_WCLR, FG_WCLRM,
       FG_FT, FG_GT, FG_TT, FG_STT, FG_PROJ, FG_FR, FG_TR, FG_TRS, FG_LN, FG_SLN, FG_OPS };
#define FG_FIRST_PRIM FG_FT
#define FG_MAX_CMDS 24576
#define FG_HASH 8192

typedef struct {
    uint8_t op, pc_valid, pq_valid, used, bd;
    uint8_t moved;               /* vdx/vdy hold each corner's motion since the previous frame */
    uint8_t how;                 /* FG_HOW_*: diagnostics */
    int16_t offx, offy;
    float vdx[3], vdy[3];
    int32_t a[20];
    /* Sub-pixel corners and perspective weights. FG_PROJ: the homogeneous
     * x, y and signed depth of each corner (see fg_proj_pairs). */
    float pcx[3], pcy[3], pq[3];
    float iz[3];                 /* 1 / camera depth of each corner, 0 = unknown */
    uint32_t key, loose;         /* look with and without UVs */
    int32_t next;                /* as prev: hash chain; as cur: partner in prev, -1 = none */
} FgCmd;

enum { FG_HOW_NONE, FG_HOW_STILL, FG_HOW_CAMERA, FG_HOW_ANCHOR, FG_HOW_MESH, FG_HOW_SEED, FG_HOW_LOOSE, FG_HOW_CORNER, FG_HOW_FIELD };

typedef struct {
    int semi_en, semi_mode, mask_set, mask_check;
    int tw[4], mod[4], da[4], off[2], view[4];
    int wide_on, wide_base;
} FgState;

typedef struct {
    FgCmd *c;
    int n, prims, wide_prims, overflow;
    FgState init;
} FgList;

/* Vertex count and the a[] slots of each vertex's x and y, per op. */
static const struct { int nv; int xi[3], yi[3]; } kFgPos[FG_OPS] = {
    [FG_FT]   = { 3, { 0, 2, 4 },  { 1, 3, 5 } },
    [FG_GT]   = { 3, { 0, 3, 6 },  { 1, 4, 7 } },
    [FG_TT]   = { 3, { 0, 4, 8 },  { 1, 5, 9 } },
    [FG_STT]  = { 3, { 0, 5, 10 }, { 1, 6, 11 } },
    [FG_PROJ] = { 3, { 0, 5, 10 }, { 1, 6, 11 } },   /* a[] as FG_STT */
    [FG_FR]   = { 1, { 0 }, { 1 } },
    [FG_TR]   = { 1, { 0 }, { 1 } },
    [FG_TRS]  = { 1, { 0 }, { 1 } },
    [FG_LN]   = { 2, { 0, 2 }, { 1, 3 } },
    [FG_SLN]  = { 2, { 0, 3 }, { 1, 4 } },
};
static const int kFgArgs[FG_OPS] = {
    [FG_FT] = 7, [FG_GT] = 9, [FG_TT] = 15, [FG_STT] = 19, [FG_PROJ] = 19, [FG_FR] = 5,
    [FG_TR] = 9, [FG_TRS] = 11, [FG_LN] = 5, [FG_SLN] = 6,
};

static int fg_is_textured(int op) {
    return op == FG_TT || op == FG_STT || op == FG_PROJ || op == FG_TR || op == FG_TRS;
}
static int fg_is_backdrop(const FgCmd *c) {
    if (c->op == FG_LN || c->op == FG_SLN || c->op == FG_PROJ) return 0;
    return (c->bd & 1) || (!fg_is_textured(c->op) && (c->bd & 2));
}
/* Primitives whose motion the mesh matcher solves. */
static int fg_is_world(const FgCmd *c) {
    return c->op >= FG_FIRST_PRIM && c->op != FG_PROJ && !fg_is_backdrop(c);
}

/* What the primitive draws with, minus where it is. Lit colours of textured
 * polygons change from frame to frame, so they stay out; `loose` also leaves
 * out texture coordinates and the colours of untextured shapes. */
static void fg_set_keys(FgCmd *c) {
    const int op = c->op, n = kFgArgs[op];
    uint32_t h = 2166136261u ^ (uint32_t)op, l = h;
    for (int i = 0; i < n; i++) {
        int pos = 0, uv = 0;
        for (int v = 0; v < kFgPos[op].nv; v++) {
            if (i == kFgPos[op].xi[v] || i == kFgPos[op].yi[v]) pos = 1;
            if ((op == FG_TT || op == FG_STT || op == FG_PROJ) &&
                (i == kFgPos[op].yi[v] + 1 || i == kFgPos[op].yi[v] + 2)) uv = 1;
        }
        if ((op == FG_STT || op == FG_PROJ) && (i == 4 || i == 9 || i == 14)) continue;
        if (pos) continue;
        h = (h ^ (uint32_t)c->a[i]) * 16777619u;
        if (!uv && fg_is_textured(op)) l = (l ^ (uint32_t)c->a[i]) * 16777619u;
    }
    c->key = h; c->loose = l ^ 0x5bd1e995u;
}

/* Corner position relative to the draw offset (independent of which of the
 * two frame buffers the frame was drawn into). */
static void fg_corner(const FgCmd *c, int v, float *x, float *y) {
    const int xi = kFgPos[c->op].xi[v], yi = kFgPos[c->op].yi[v];
    *x = (c->pc_valid ? c->pcx[v] : (float)c->a[xi]) - (float)c->offx;
    *y = (c->pc_valid ? c->pcy[v] : (float)c->a[yi]) - (float)c->offy;
}
static void fg_centre(const FgCmd *c, float *cx, float *cy) {
    const int nv = kFgPos[c->op].nv;
    float x = 0, y = 0;
    for (int i = 0; i < nv; i++) { float px, py; fg_corner(c, i, &px, &py); x += px; y += py; }
    *cx = x / (float)nv; *cy = y / (float)nv;
}

/* ---- solver state (one frame pair at a time; single thread) -------------- */
#define FG_VERTS   (FG_MAX_CMDS * 3)
#define FG_VHASH   131072
#define FG_MODES   3             /* distinct motions remembered per corner */
#define FG_REACH   72.0f         /* px an unaided pairing may span */
#define FG_REACH_MESH 192.0f     /* px a pairing confirmed by its corners may span */
typedef struct { float dx[FG_MODES], dy[FG_MODES]; uint16_t n[FG_MODES]; int16_t x16, y16; int32_t first; } FgVert;
static FgVert  s_fgv[FG_VERTS];
static int     s_fgv_n;
static int32_t s_fgv_hash[FG_VHASH];
static int32_t s_fgv_of[FG_MAX_CMDS][3];       /* corner ids of each cur primitive, -1 = not tracked */
static int32_t s_fgv_next[FG_VERTS];           /* chain of (primitive*3 + corner) per corner id */
static int32_t s_fg_kh[FG_HASH], s_fg_lh[FG_HASH], s_fg_lnext[FG_MAX_CMDS];
/* prev's primitives again, small enough to walk: corners, centre, and a grid
 * of FG_CELL px cells (by look and by the place of corner 0). */
#define FG_CELL  16
#define FG_GHASH 32768
typedef struct { float x[3], y[3], cx, cy; } FgGeo;
static FgGeo   s_fg_pg[FG_MAX_CMDS];
static int32_t s_fg_gh[2][FG_GHASH], s_fg_gnext[2][FG_MAX_CMDS];
static int32_t s_fg_queue[FG_MAX_CMDS];
static uint8_t s_fg_inq[FG_MAX_CMDS];
static int     s_fgq_head, s_fgq_len;
static int32_t s_fg_comp[FG_MAX_CMDS], s_fg_stack[FG_MAX_CMDS];
/* Paired primitives as samples of the motion field. */
typedef struct { float x, y, dx, dy, o; } FgSample;
#define FG_SAMPLES 1024
static FgSample s_fg_smp[FG_SAMPLES];
static int      s_fg_smp_n;

static uint32_t fg_cell_hash(uint32_t look, int ix, int iy) {
    uint32_t h = look ^ ((uint32_t)ix * 73856093u) ^ ((uint32_t)iy * 19349663u);
    h *= 2654435761u;
    return h >> 17;                            /* FG_GHASH entries */
}
static int fg_cell(float v) { return (int)floorf(v / (float)FG_CELL); }
static void fg_index_prev(const FgList *prev) {
    memset(s_fg_gh, 0xff, sizeof s_fg_gh);
    for (int i = prev->n - 1; i >= 0; i--) {
        const FgCmd *p = &prev->c[i]; FgGeo *g = &s_fg_pg[i];
        if (p->op < FG_FIRST_PRIM) continue;
        const int nv = kFgPos[p->op].nv;
        g->cx = g->cy = 0.0f;
        for (int v = 0; v < nv; v++) { fg_corner(p, v, &g->x[v], &g->y[v]); g->cx += g->x[v]; g->cy += g->y[v]; }
        g->cx /= (float)nv; g->cy /= (float)nv;
        if (nv != 3) continue;
        for (int l = 0; l < 2; l++) {
            const uint32_t h = fg_cell_hash(l ? p->loose : p->key, fg_cell(g->x[0]), fg_cell(g->y[0]));
            s_fg_gnext[l][i] = s_fg_gh[l][h]; s_fg_gh[l][h] = i;
        }
    }
}
static int32_t fg_vert_id(float x, float y) {
    /* 1/16 px: a shared corner is the same value in every primitive, while
     * two unrelated corners rarely agree to a sixteenth. */
    const int ix = (int)floorf(x * 16.0f + 0.5f), iy = (int)floorf(y * 16.0f + 0.5f);
    if (ix < -16000 || ix > 16000 || iy < -16000 || iy > 16000) return -1;   /* GTE clamp range */
    uint32_t h = (((uint32_t)ix * 73856093u) ^ ((uint32_t)iy * 19349663u)) * 2654435761u;
    for (h >>= 15;; h = (h + 1) & (FG_VHASH - 1)) {
        const int32_t id = s_fgv_hash[h & (FG_VHASH - 1)];
        if (id < 0) {
            FgVert *q = &s_fgv[s_fgv_n];
            memset(q, 0, sizeof *q);
            q->x16 = (int16_t)ix; q->y16 = (int16_t)iy; q->first = -1;
            s_fgv_hash[h & (FG_VHASH - 1)] = s_fgv_n;
            return s_fgv_n++;
        }
        if (s_fgv[id].x16 == ix && s_fgv[id].y16 == iy) return id;
    }
}
static float fg_tol(float dx, float dy) { return 1.25f + 0.02f * (fabsf(dx) + fabsf(dy)); }
static void fg_vert_add(int32_t id, float dx, float dy) {
    if (id < 0) return;
    FgVert *q = &s_fgv[id];
    const float t = fg_tol(dx, dy);
    for (int m = 0; m < FG_MODES; m++) {
        if (!q->n[m]) { q->dx[m] = dx; q->dy[m] = dy; q->n[m] = 1; return; }
        if (fabsf(q->dx[m] - dx) <= t && fabsf(q->dy[m] - dy) <= t) { if (q->n[m] < 65535) q->n[m]++; return; }
    }
}
static int fg_vert_main(int32_t id) {          /* best supported motion, -1 = unknown */
    if (id < 0 || !s_fgv[id].n[0]) return -1;
    int b = 0;
    for (int m = 1; m < FG_MODES; m++) if (s_fgv[id].n[m] > s_fgv[id].n[b]) b = m;
    return b;
}

static void fg_queue_around(const FgList *cur, int i) {
    for (int v = 0; v < 3; v++) {
        const int32_t id = s_fgv_of[i][v];
        if (id < 0) continue;
        for (int32_t k = s_fgv[id].first; k >= 0; k = s_fgv_next[k]) {
            const int j = k / 3;
            if (cur->c[j].next >= 0 || s_fg_inq[j] || s_fgq_len >= FG_MAX_CMDS) continue;
            s_fg_inq[j] = 1;
            s_fg_queue[(s_fgq_head + s_fgq_len++) % FG_MAX_CMDS] = j;
        }
    }
}
static void fg_accept(FgList *prev, FgList *cur, int i, int j, int how) {
    FgCmd *c = &cur->c[i]; FgCmd *p = &prev->c[j];
    p->used = 1; c->used = 1; c->next = j; c->how = (uint8_t)how; c->moved = 1;
    for (int v = 0; v < kFgPos[c->op].nv; v++) {
        float cx, cy, px, py;
        fg_corner(c, v, &cx, &cy); fg_corner(p, v, &px, &py);
        c->vdx[v] = cx - px; c->vdy[v] = cy - py;
        if (kFgPos[c->op].nv == 3) fg_vert_add(s_fgv_of[i][v], c->vdx[v], c->vdy[v]);
    }
    if (kFgPos[c->op].nv == 3) fg_queue_around(cur, i);
}

static int32_t fg_chain_first(const FgCmd *c, int loose) {
    return loose ? s_fg_lh[c->loose & (FG_HASH - 1)] : s_fg_kh[c->key & (FG_HASH - 1)];
}
static int32_t fg_chain_next(const FgList *prev, int32_t j, int loose) {
    return loose ? s_fg_lnext[j] : prev->c[j].next;
}
static int fg_same_look(const FgCmd *c, const FgCmd *p, int loose) {
    if (p->op != c->op || p->used) return 0;
    return loose ? p->loose == c->loose : p->key == c->key;
}
/* Two faces of one model can share a look; their outlines differ. Summed edge
 * difference in pixels, and whether it is small for one frame of motion. */
static int fg_same_shape(const FgCmd *c, const FgCmd *p, float *diff) {
    float cx[3], cy[3], px[3], py[3], sum = 0.0f; int ok = 1;
    for (int v = 0; v < 3; v++) { fg_corner(c, v, &cx[v], &cy[v]); fg_corner(p, v, &px[v], &py[v]); }
    for (int v = 0; v < 3; v++) {
        const int w = (v + 1) % 3;
        const float ex = cx[w] - cx[v], ey = cy[w] - cy[v], fx = px[w] - px[v], fy = py[w] - py[v];
        const float d = fabsf(ex - fx) + fabsf(ey - fy), s = fabsf(ex) + fabsf(ey);
        sum += d;
        if (d > 3.0f + 0.3f * s) ok = 0;
    }
    if (diff) *diff = sum;
    return ok;
}

/* Growth: pair primitive i only if its known corners agree. */
static int fg_try_mesh(FgList *prev, FgList *cur, int i, int loose, int how) {
    const FgCmd *c = &cur->c[i];
    int known = 0;
    for (int v = 0; v < 3; v++) if (fg_vert_main(s_fgv_of[i][v]) >= 0) known++;
    if (!known) return 0;
    float cx[3], cy[3];
    for (int v = 0; v < 3; v++) fg_corner(c, v, &cx[v], &cy[v]);
    int best = -1, best_agree = 0, steps = 0; float best_err = 1e9f;
    for (int32_t j = fg_chain_first(c, loose); j >= 0 && steps < 1024; j = fg_chain_next(prev, j, loose), steps++) {
        const FgCmd *p = &prev->c[j];
        if (!fg_same_look(c, p, loose)) continue;
        int agree = 0, differ = 0; float err = 0.0f;
        for (int v = 0; v < 3; v++) {
            const float dx = cx[v] - s_fg_pg[j].x[v], dy = cy[v] - s_fg_pg[j].y[v];
            if (fabsf(dx) > FG_REACH_MESH || fabsf(dy) > FG_REACH_MESH) { differ = 3; break; }
            const int32_t id = s_fgv_of[i][v];
            if (id < 0 || !s_fgv[id].n[0]) continue;
            float bd = 1e9f;
            for (int m = 0; m < FG_MODES && s_fgv[id].n[m]; m++) {
                const float d = fabsf(s_fgv[id].dx[m] - dx) + fabsf(s_fgv[id].dy[m] - dy);
                if (d < bd) bd = d;
            }
            if (bd <= fg_tol(dx, dy)) { agree++; err += bd; } else differ++;
        }
        if (!agree || (differ && agree < 2)) continue;
        if (loose && !fg_same_shape(c, p, NULL)) continue;
        if (agree > best_agree || (agree == best_agree && err < best_err)) { best = j; best_agree = agree; best_err = err; }
    }
    if (best < 0) return 0;
    fg_accept(prev, cur, i, best, how);
    return 1;
}
static int fg_grow(FgList *prev, FgList *cur, int loose, int how) {
    int made = 0;
    while (s_fgq_len > 0) {
        const int i = s_fg_queue[s_fgq_head];
        s_fgq_head = (s_fgq_head + 1) % FG_MAX_CMDS; s_fgq_len--;
        s_fg_inq[i] = 0;
        if (cur->c[i].next < 0) made += fg_try_mesh(prev, cur, i, loose, how);
    }
    return made;
}

/* ---- motion field around a point ----------------------------------------- */
static void fg_sample_all(const FgList *cur) {
    /* What is fixed to the screen says nothing about the scenery around it,
     * unless all of the picture stands still. */
    int total = 0, still = 0;
    for (int i = 0; i < cur->n; i++)
        if (fg_is_world(&cur->c[i]) && cur->c[i].next >= 0) { if (cur->c[i].how == FG_HOW_STILL) still++; else total++; }
    const int use_still = total < still / 4;
    if (use_still) total += still;
    s_fg_smp_n = 0;
    if (!total) return;
    const int step = (total + FG_SAMPLES - 1) / FG_SAMPLES;
    int k = 0;
    for (int i = 0; i < cur->n && s_fg_smp_n < FG_SAMPLES; i++) {
        const FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || c->next < 0 || (c->how == FG_HOW_STILL && !use_still) || (k++ % step) != 0) continue;
        FgSample *s = &s_fg_smp[s_fg_smp_n++];
        fg_centre(c, &s->x, &s->y);
        const int nv = kFgPos[c->op].nv;
        s->dx = s->dy = 0.0f; s->o = (float)i;
        for (int v = 0; v < nv; v++) { s->dx += c->vdx[v]; s->dy += c->vdy[v]; }
        s->dx /= (float)nv; s->dy /= (float)nv;
    }
}
typedef struct { float x, y, u[3], v[3]; int ok; } FgField;   /* u = u0 + u1*(x-x0) + u2*(y-y0) */
/* Weighted affine fit of the samples near (x, y) in the picture and near
 * position `order` in the draw list; two rounds, the second ignoring samples
 * that disagree with the first (another object passing in front). */
static FgField fg_field(float x, float y, float order, int count, int flat) {
    FgField f; memset(&f, 0, sizeof f); f.x = x; f.y = y;
    const float os = 10.0f / (float)(count > 0 ? count : 1);
    for (int pass = 0; pass < 3; pass++) {
        double m[6] = { 0 }, bu[3] = { 0 }, bv[3] = { 0 };
        for (int k = 0; k < s_fg_smp_n; k++) {
            const FgSample *s = &s_fg_smp[k];
            const float rx = s->x - x, ry = s->y - y, ro = (s->o - order) * os;
            float w = 1.0f / (1.0f + (rx * rx + ry * ry) / (40.0f * 40.0f) + ro * ro);
            w *= w;
            if (pass) {
                const float eu = s->dx - (f.u[0] + f.u[1] * rx + f.u[2] * ry);
                const float ev = s->dy - (f.v[0] + f.v[1] * rx + f.v[2] * ry);
                w /= 1.0f + (eu * eu + ev * ev) / 9.0f;
            }
            m[0] += w; m[1] += w * rx; m[2] += w * ry; m[3] += w * rx * rx; m[4] += w * rx * ry; m[5] += w * ry * ry;
            bu[0] += w * s->dx; bu[1] += w * s->dx * rx; bu[2] += w * s->dx * ry;
            bv[0] += w * s->dy; bv[1] += w * s->dy * rx; bv[2] += w * s->dy * ry;
        }
        if (m[0] < 1e-9) return f;
        f.ok = 1;
        /* Ridge: without spread in the samples the slope stays zero. */
        const double r = flat ? 1e9 : m[0] * 400.0;
        const double a = m[3] + r, d = m[5] + r, b = m[4];
        const double det = m[0] * (a * d - b * b) - m[1] * (m[1] * d - b * m[2]) + m[2] * (m[1] * b - a * m[2]);
        if (fabs(det) < 1e-9) { f.u[0] = (float)(bu[0] / m[0]); f.v[0] = (float)(bv[0] / m[0]); continue; }
        for (int q = 0; q < 2; q++) {
            const double *t = q ? bv : bu;
            float *o = q ? f.v : f.u;
            o[0] = (float)((t[0] * (a * d - b * b) - m[1] * (t[1] * d - b * t[2]) + m[2] * (t[1] * b - a * t[2])) / det);
            o[1] = (float)((m[0] * (t[1] * d - b * t[2]) - t[0] * (m[1] * d - b * m[2]) + m[2] * (m[1] * t[2] - t[1] * m[2])) / det);
            o[2] = (float)((m[0] * (a * t[2] - t[1] * b) - m[1] * (m[1] * t[2] - t[1] * m[2]) + t[0] * (m[1] * b - a * m[2])) / det);
        }
    }
    return f;
}
static void fg_field_at(const FgField *f, float x, float y, float *dx, float *dy) {
    *dx = f->u[0] + f->u[1] * (x - f->x) + f->u[2] * (y - f->y);
    *dy = f->v[0] + f->v[1] * (x - f->x) + f->v[2] * (y - f->y);
}

/* ---- camera motion ------------------------------------------------------- */
#define FG_NB 8
#define FG_FLOW_SAMPLES 256
typedef struct { double u[FG_NB], v[FG_NB], rs; int ok, inliers, n; } FgFlow;
static FgFlow s_fg_flow_last;                  /* start value: cameras move smoothly */
static int fg_has_depth(const FgCmd *c) {
    return kFgPos[c->op].nv == 3 && c->iz[0] > 0.0f && c->iz[1] > 0.0f && c->iz[2] > 0.0f;
}
/* First-order motion field of a rigid scene, u = (a + b x) / z + rotation
 * (quadratic in x, y), with the next term of a fast approach. */
static void fg_basis(const FgFlow *f, float x, float y, float iz, double *bu, double *bv) {
    const double X = (x - 160.0) / 160.0, Y = (y - 120.0) / 160.0, R = iz / f->rs;
    bu[0] = bv[0] = 1.0; bu[1] = bv[1] = X; bu[2] = bv[2] = Y;
    bu[3] = X * X; bu[4] = X * Y; bu[5] = R; bu[6] = X * R; bu[7] = X * R * R;
    bv[3] = X * Y; bv[4] = Y * Y; bv[5] = R; bv[6] = Y * R; bv[7] = Y * R * R;
}
static void fg_flow_at(const FgFlow *f, float x, float y, float iz, float *dx, float *dy) {
    double bu[FG_NB], bv[FG_NB], u = 0, v = 0;
    fg_basis(f, x, y, iz, bu, bv);
    for (int k = 0; k < FG_NB; k++) { u += f->u[k] * bu[k]; v += f->v[k] * bv[k]; }
    *dx = (float)u; *dy = (float)v;
}
static int fg_solve_sym(double a[FG_NB][FG_NB + 1], double *x) {      /* Gaussian elimination, pivoted */
    for (int c = 0; c < FG_NB; c++) {
        int p = c;
        for (int r = c + 1; r < FG_NB; r++) if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return 0;
        if (p != c) for (int k = 0; k <= FG_NB; k++) { const double t = a[c][k]; a[c][k] = a[p][k]; a[p][k] = t; }
        for (int r = c + 1; r < FG_NB; r++) {
            const double m = a[r][c] / a[c][c];
            for (int k = c; k <= FG_NB; k++) a[r][k] -= m * a[c][k];
        }
    }
    for (int c = FG_NB - 1; c >= 0; c--) {
        double t = a[c][FG_NB];
        for (int k = c + 1; k < FG_NB; k++) t -= a[c][k] * x[k];
        x[c] = t / a[c][c];
    }
    return 1;
}
typedef struct { float x[3], y[3], iz[3], dx[3], dy[3], w; } FgFlowObs;
static FgFlowObs s_fg_obs[FG_FLOW_SAMPLES * 2];
static int fg_flow_fit(FgFlow *f, const FgFlowObs *o, int n) {
    static double nu[FG_NB][FG_NB + 1], nv[FG_NB][FG_NB + 1];
    memset(nu, 0, sizeof nu); memset(nv, 0, sizeof nv);
    double wt = 0;
    for (int i = 0; i < n; i++) {
        if (o[i].w <= 0.0f) continue;
        for (int c = 0; c < 3; c++) {
            double bu[FG_NB], bv[FG_NB];
            fg_basis(f, o[i].x[c], o[i].y[c], o[i].iz[c], bu, bv);
            for (int r = 0; r < FG_NB; r++) {
                for (int k = r; k < FG_NB; k++) { nu[r][k] += o[i].w * bu[r] * bu[k]; nv[r][k] += o[i].w * bv[r] * bv[k]; }
                nu[r][FG_NB] += o[i].w * bu[r] * o[i].dx[c]; nv[r][FG_NB] += o[i].w * bv[r] * o[i].dy[c];
            }
            wt += o[i].w;
        }
    }
    if (wt < 12.0) return 0;
    for (int r = 0; r < FG_NB; r++) {
        for (int k = 0; k < r; k++) { nu[r][k] = nu[k][r]; nv[r][k] = nv[k][r]; }
        const double ridge = wt * (r == 0 ? 1e-9 : r == FG_NB - 1 ? 1e-3 : 1e-5);
        nu[r][r] += ridge; nv[r][r] += ridge;
    }
    return fg_solve_sym(nu, f->u) && fg_solve_sym(nv, f->v);
}
/* Partner of cur primitive c nearest to where the flow expects it, with no
 * corner further than `reach` from there; *err = the worst corner's distance. */
static int fg_flow_partner(const FgList *prev, const FgCmd *c, const FgFlow *f, int loose, float reach, float *err) {
    float ex[3], ey[3];                        /* expected corners one frame ago */
    for (int v = 0; v < 3; v++) {
        float x, y, dx, dy; fg_corner(c, v, &x, &y); fg_flow_at(f, x, y, c->iz[v], &dx, &dy);
        ex[v] = x - dx; ey[v] = y - dy;
    }
    int best = -1; *err = reach;
    const int grid = reach <= (float)FG_CELL;
    const int gx0 = grid ? fg_cell(ex[0] - reach) : 0, gx1 = grid ? fg_cell(ex[0] + reach) : 0;
    const int gy0 = grid ? fg_cell(ey[0] - reach) : 0, gy1 = grid ? fg_cell(ey[0] + reach) : 0;
    for (int gy = gy0; gy <= gy1; gy++) for (int gx = gx0; gx <= gx1; gx++) {
        int steps = 0;
        for (int32_t j = grid ? s_fg_gh[loose][fg_cell_hash(loose ? c->loose : c->key, gx, gy)] : fg_chain_first(c, loose);
             j >= 0 && steps < 1024; j = grid ? s_fg_gnext[loose][j] : fg_chain_next(prev, j, loose), steps++) {
            const FgGeo *g = &s_fg_pg[j];
            float e = fabsf(g->x[0] - ex[0]) + fabsf(g->y[0] - ey[0]);
            if (e >= *err || !fg_same_look(c, &prev->c[j], loose)) continue;
            for (int v = 1; v < 3 && e < *err; v++) {
                const float d = fabsf(g->x[v] - ex[v]) + fabsf(g->y[v] - ey[v]);
                if (d > e) e = d;
            }
            if (e < *err) { *err = e; best = j; }
        }
    }
    return best;
}
/* Alternate "pair with the nearest expected partner" and "refit". */
static void fg_flow_refine(const FgList *prev, const FgList *cur, const int32_t *smp, int n, FgFlow *f, int from) {
    static const float gate[7] = { 40.0f, 24.0f, 14.0f, 8.0f, 5.0f, 3.0f, 2.0f };
    for (int it = from; it < 7; it++) {
        f->inliers = 0; f->n = n;
        for (int k = 0; k < n; k++) {
            const FgCmd *c = &cur->c[smp[k]]; FgFlowObs *o = &s_fg_obs[k];
            float e; const int j = fg_flow_partner(prev, c, f, 0, gate[it] * 2.0f, &e);
            o->w = 0.0f;
            if (j < 0) continue;
            if (e <= 1.5f) f->inliers++;
            o->w = 1.0f / (1.0f + (e * e) / (gate[it] * gate[it] * 0.25f));
            for (int v = 0; v < 3; v++) {
                float px, py; fg_corner(c, v, &o->x[v], &o->y[v]); fg_corner(&prev->c[j], v, &px, &py);
                o->iz[v] = c->iz[v]; o->dx[v] = o->x[v] - px; o->dy[v] = o->y[v] - py;
            }
        }
        FgFlow t = *f;
        if (!fg_flow_fit(&t, s_fg_obs, n)) return;
        *f = t;
    }
    f->inliers = 0;
    for (int k = 0; k < n; k++) { float e; if (fg_flow_partner(prev, &cur->c[smp[k]], f, 0, 1.5f, &e) >= 0) f->inliers++; }
}
/* `anch`: primitives already paired one way only, a start value. */
static FgFlow fg_flow_find(const FgList *prev, const FgList *cur, const int32_t *anch, int an) {
    FgFlow best; memset(&best, 0, sizeof best);
    static int32_t smp[FG_FLOW_SAMPLES]; int total = 0, n = 0; double rs = 0;
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (fg_is_world(c) && c->how != FG_HOW_STILL && fg_has_depth(c)) { total++; rs += c->iz[0]; }
    }
    if (total < 24) return best;
    rs /= total;
    double at = 0.0; int seen = 0;
    const double step = total > FG_FLOW_SAMPLES ? (double)total / FG_FLOW_SAMPLES : 1.0;
    for (int i = 0; i < cur->n && n < FG_FLOW_SAMPLES; i++) {
        const FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || c->how == FG_HOW_STILL || !fg_has_depth(c)) continue;
        if (seen++ >= (int)at) { smp[n++] = i; at += step; }
    }
    for (int init = 0; init < 3; init++) {              /* last frame's motion, the anchors', none */
        FgFlow f; memset(&f, 0, sizeof f); f.rs = rs;
        if (init == 0) {
            if (!s_fg_flow_last.ok) continue;
            f = s_fg_flow_last;                /* depth scale differs a little: restate the 1/z terms */
            const double k = rs / f.rs;
            f.u[5] *= k; f.u[6] *= k; f.u[7] *= k * k; f.v[5] *= k; f.v[6] *= k; f.v[7] *= k * k; f.rs = rs;
        }
        if (init == 1) {
            int m = 0;
            for (int k = 0; k < an && m < FG_FLOW_SAMPLES; k++) {
                const FgCmd *c = &cur->c[anch[k]];
                if (c->next < 0 || c->how != FG_HOW_ANCHOR || !fg_has_depth(c)) continue;
                FgFlowObs *o = &s_fg_obs[m++]; o->w = 1.0f;
                for (int v = 0; v < 3; v++) { fg_corner(c, v, &o->x[v], &o->y[v]); o->iz[v] = c->iz[v]; o->dx[v] = c->vdx[v]; o->dy[v] = c->vdy[v]; }
            }
            if (m < 8) continue;
            int good = 1;
            for (int it = 0; it < 6 && good; it++) {       /* moving objects drop out */
                good = fg_flow_fit(&f, s_fg_obs, m);
                for (int k = 0; k < m && good; k++) {
                    float ex, ey; fg_flow_at(&f, s_fg_obs[k].x[0], s_fg_obs[k].y[0], s_fg_obs[k].iz[0], &ex, &ey);
                    const float d = fabsf(s_fg_obs[k].dx[0] - ex) + fabsf(s_fg_obs[k].dy[0] - ey);
                    s_fg_obs[k].w = 1.0f / (1.0f + d * d / 4.0f);
                }
            }
            if (!good) continue;
        }
        fg_flow_refine(prev, cur, smp, n, &f, init == 0 ? 3 : init == 1 ? 2 : 0);
        if (f.inliers > best.inliers) best = f;
        if (best.inliers * 10 >= n * 8) break;             /* nothing left to find */
    }
    if (best.inliers * 4 < n * 3) {
        /* No start value led there (a cut, a sudden turn): try plain shifts of
         * the whole picture on a subset and refine the three best. */
        struct { int score; float dx, dy; } top[3]; memset(top, 0, sizeof top);
        const int stride = n > 96 ? n / 96 : 1;
        for (int gy = -60; gy <= 60; gy += 12) for (int gx = -60; gx <= 60; gx += 12) {
            FgFlow f; memset(&f, 0, sizeof f); f.rs = rs; f.u[0] = gx; f.v[0] = gy;
            int sc = 0;
            for (int k = 0; k < n; k += stride) {
                float e;
                if (fg_flow_partner(prev, &cur->c[smp[k]], &f, 0, 12.0f, &e) >= 0) sc += e < 6.0f ? 2 : 1;
            }
            for (int q = 0; q < 3; q++) if (sc > top[q].score) {
                for (int r = 2; r > q; r--) top[r] = top[r - 1];
                top[q].score = sc; top[q].dx = (float)gx; top[q].dy = (float)gy;
                break;
            }
        }
        for (int q = 0; q < 3 && top[q].score; q++) {
            FgFlow f; memset(&f, 0, sizeof f); f.rs = rs; f.u[0] = top[q].dx; f.v[0] = top[q].dy;
            fg_flow_refine(prev, cur, smp, n, &f, 2);
            if (f.inliers > best.inliers) best = f;
        }
    }
    best.ok = best.inliers >= 20 && best.inliers * 5 >= n * 2;
    return best;
}
static int fg_flow_accept(FgList *prev, FgList *cur, const FgFlow *f, int loose) {
    int made = 0;
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || c->next >= 0 || !fg_has_depth(c)) continue;
        if (loose && !fg_is_textured(c->op)) continue;
        float e, sp = 0.0f;
        for (int v = 0; v < 3; v++) {
            float x, y, dx, dy; fg_corner(c, v, &x, &y); fg_flow_at(f, x, y, c->iz[v], &dx, &dy);
            if (fabsf(dx) + fabsf(dy) > sp) sp = fabsf(dx) + fabsf(dy);
        }
        const int j = fg_flow_partner(prev, c, f, loose, 1.6f + 0.03f * (sp > 200.0f ? 200.0f : sp), &e);
        if (j < 0 || (loose && !fg_same_shape(c, &prev->c[j], NULL))) continue;
        fg_accept(prev, cur, i, j, loose ? FG_HOW_LOOSE : FG_HOW_CAMERA); made++;
    }
    return made;
}

/* ---- look-alikes ---------------------------------------------------------- */
/* Two identical objects side by side, and one of them changes its look (a
 * darker palette, a coarser model) or leaves the picture: for that frame the
 * other one is the only partner in reach, and the object would be drawn half
 * way across to it. What gives the mistake away is the sudden motion: real
 * objects move nearly as they did one frame before, the camera's own change of
 * speed aside, which everything shares. */
#define FG_JUMP 10.0f            /* px of unexplained change of motion */
static struct { float dx, dy, spread; int n; } s_fg_accel;
static int fg_cmp_float(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}
static void fg_mean_motion(const FgCmd *c, float *dx, float *dy) {
    const int nv = kFgPos[c->op].nv;
    *dx = *dy = 0.0f;
    for (int v = 0; v < nv; v++) { *dx += c->vdx[v]; *dy += c->vdy[v]; }
    *dx /= (float)nv; *dy /= (float)nv;
}
/* The change of motion the pairs `at` have in common. */
static void fg_accel_find(const FgList *prev, const FgList *cur, const int32_t *at, int n) {
    enum { CAP = 512 };
    static float ax[CAP], ay[CAP], d[CAP], sx[CAP], sy[CAP];
    int m = 0;
    s_fg_accel.n = 0;
    const int step = n > CAP ? (n + CAP - 1) / CAP : 1;
    for (int k = 0; k < n && m < CAP; k += step) {
        const FgCmd *c = &cur->c[at[k]];
        if (c->next < 0 || c->how != FG_HOW_ANCHOR) continue;
        const FgCmd *p = &prev->c[c->next];
        if (!p->moved || p->how >= FG_HOW_CORNER) continue;
        float nx, ny, ox, oy; fg_mean_motion(c, &nx, &ny); fg_mean_motion(p, &ox, &oy);
        ax[m] = sx[m] = nx - ox; ay[m] = sy[m] = ny - oy; m++;
    }
    if (m < 12) return;                        /* too few to tell */
    qsort(sx, (size_t)m, sizeof sx[0], fg_cmp_float); qsort(sy, (size_t)m, sizeof sy[0], fg_cmp_float);
    s_fg_accel.dx = sx[m / 2]; s_fg_accel.dy = sy[m / 2];
    for (int k = 0; k < m; k++) d[k] = fabsf(ax[k] - s_fg_accel.dx) + fabsf(ay[k] - s_fg_accel.dy);
    qsort(d, (size_t)m, sizeof d[0], fg_cmp_float);
    s_fg_accel.spread = d[m / 2];
    s_fg_accel.n = m;
}
/* How far a motion (dx, dy) is from going on as p did; 0 = p's is not known. */
static float fg_jump(const FgCmd *p, float dx, float dy) {
    if (!s_fg_accel.n || !p->moved) return 0.0f;
    float ox, oy; fg_mean_motion(p, &ox, &oy);
    return fabsf(dx - ox - s_fg_accel.dx) + fabsf(dy - oy - s_fg_accel.dy);
}
/* Can c be where p was, (dx, dy) away? */
static int fg_plausible(const FgCmd *c, const FgCmd *p, const FgFlow *f, float dx, float dy) {
    const float e = fg_jump(p, dx, dy);
    if (e <= FG_JUMP) return 1;
    float ox, oy; fg_mean_motion(p, &ox, &oy);
    /* A guessed motion says less about the next one. */
    const float lim = (p->how >= FG_HOW_CORNER ? 2.0f : 1.0f) * FG_JUMP + 3.0f * s_fg_accel.spread + 0.5f * (fabsf(ox) + fabsf(oy));
    if (e <= lim) return 1;
    if (!f || !fg_has_depth(c)) return 0;
    float ex = 0.0f, ey = 0.0f;                /* what the camera does may be as sudden as it likes */
    for (int v = 0; v < 3; v++) {
        float x, y, ux, uy; fg_corner(c, v, &x, &y); fg_flow_at(f, x, y, c->iz[v], &ux, &uy);
        ex += ux / 3.0f; ey += uy / 3.0f;
    }
    return fabsf(dx - ex) + fabsf(dy - ey) <= 2.0f + 0.06f * (fabsf(ex) + fabsf(ey));
}

/* ---- meshes without an anchor -------------------------------------------- */
static int fg_component(const FgList *cur, int seed, int mark) {   /* unpaired primitives joined by corners */
    int n = 0, top = 0;
    s_fg_stack[top++] = seed; s_fg_comp[seed] = mark;
    static int32_t out[FG_MAX_CMDS];
    while (top > 0) {
        const int i = s_fg_stack[--top];
        out[n++] = i;
        for (int v = 0; v < 3; v++) {
            const int32_t id = s_fgv_of[i][v];
            if (id < 0) continue;
            for (int32_t k = s_fgv[id].first; k >= 0; k = s_fgv_next[k]) {
                const int j = k / 3;
                if (cur->c[j].next >= 0 || s_fg_comp[j] == mark) continue;
                s_fg_comp[j] = mark;
                s_fg_stack[top++] = j;
            }
        }
    }
    memcpy(s_fg_stack, out, sizeof(int32_t) * (size_t)n);
    return n;
}
/* How many of the mesh's primitives near (sx, sy) have a partner when that part
 * moved by (hx, hy) on top of the field's variation across it. */
static int fg_score(const FgList *prev, const FgList *cur, const int32_t *m, int n, const FgField *f,
                    float sx, float sy, float hx, float hy, int loose, int *of) {
    float bx, by; fg_field_at(f, sx, sy, &bx, &by);
    int ok = 0; *of = 0;
    for (int k = 0; k < n && *of < 96; k++) {
        const FgCmd *c = &cur->c[m[k]];
        float cx, cy, ex, ey; fg_centre(c, &cx, &cy);
        if (fabsf(cx - sx) > 70.0f || fabsf(cy - sy) > 70.0f) continue;
        fg_field_at(f, cx, cy, &ex, &ey);
        ex += hx - bx; ey += hy - by;
        const float t = 2.0f + 0.03f * (fabsf(cx - sx) + fabsf(cy - sy));
        int steps = 0; (*of)++;
        for (int32_t j = fg_chain_first(c, loose); j >= 0 && steps < 512; j = fg_chain_next(prev, j, loose), steps++) {
            const FgCmd *p = &prev->c[j];
            if (!fg_same_look(c, p, loose)) continue;
            if (fabsf(cx - s_fg_pg[j].cx - ex) <= t && fabsf(cy - s_fg_pg[j].cy - ey) <= t && fg_same_shape(c, p, NULL)) { ok++; break; }
        }
    }
    return ok;
}
/* sure: accept only a mesh whose best shift is clearly better than any other,
 * wherever the surroundings move; else the field breaks ties. */
static void fg_seed_components(FgList *prev, FgList *cur, const FgFlow *flow, int loose, int how, int sure) {
    for (int i = 0; i < cur->n; i++) s_fg_comp[i] = 0;
    int mark = 0;
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || kFgPos[c->op].nv != 3 || c->next >= 0 || s_fg_comp[i]) continue;
        if (loose && !fg_is_textured(c->op)) continue;
        const int n = fg_component(cur, i, ++mark);
        if (sure && n < 6) continue;
        static int32_t mem[FG_MAX_CMDS];
        memcpy(mem, s_fg_stack, sizeof(int32_t) * (size_t)n);
        float mx = 0, my = 0, mo = 0;
        for (int k = 0; k < n; k++) { float x, y; fg_centre(&cur->c[mem[k]], &x, &y); mx += x; my += y; mo += (float)mem[k]; }
        mx /= (float)n; my /= (float)n; mo /= (float)n;
        const FgField f = fg_field(mx, my, mo, cur->n, 0);
        float best = 0.0f, second = 0.0f, bhx = 0, bhy = 0, bfrac = 0; int bi = -1, bj = -1;
        const int tries = sure ? 1 : 3;
        for (int s = 0; s < tries && s < n; s++) {       /* pieces spread over the mesh propose shifts */
            int si = mem[(int)(((int64_t)n * (2 * s + 1)) / (2 * tries)) % n];
            if (sure) {                                  /* the piece nearest the mesh centre */
                float bd = 1e9f;
                for (int k = 0; k < n; k++) { float x, y; fg_centre(&cur->c[mem[k]], &x, &y); const float d = hypotf(x - mx, y - my); if (d < bd) { bd = d; si = mem[k]; } }
            }
            const FgCmd *d = &cur->c[si];
            float cx, cy, ex, ey; fg_centre(d, &cx, &cy); fg_field_at(&f, cx, cy, &ex, &ey);
            struct { int j; float d; } top[24]; const int cap = sure ? 24 : 8; int tn = 0, steps = 0;
            for (int32_t j = fg_chain_first(d, loose); j >= 0 && steps < 512; j = fg_chain_next(prev, j, loose), steps++) {
                const FgCmd *p = &prev->c[j];
                float sd;
                if (!fg_same_look(d, p, loose) || !fg_same_shape(d, p, &sd)) continue;
                float px, py; fg_centre(p, &px, &py);
                const float dd = hypotf(cx - px - ex, cy - py - ey) + sd;
                if (dd > (sure ? FG_REACH : 24.0f + 0.5f * (fabsf(ex) + fabsf(ey)))) continue;
                /* Both the surroundings and its own last motion speak against it. */
                if (!sure && dd > 8.0f && !fg_plausible(d, p, flow, cx - px, cy - py)) continue;
                int at = tn;                             /* the nearest, sorted */
                while (at > 0 && top[at - 1].d > dd) at--;
                if (at >= cap) continue;
                for (int q = tn < cap ? tn : cap - 1; q > at; q--) top[q] = top[q - 1];
                top[at].j = j; top[at].d = dd;
                if (tn < cap) tn++;
            }
            for (int k = 0; k < tn; k++) {
                float px, py; fg_centre(&prev->c[top[k].j], &px, &py);
                int of; const int sc = fg_score(prev, cur, mem, n, &f, cx, cy, cx - px, cy - py, loose, &of);
                /* Ties go to what the surroundings do and, for a little, to what the
                 * piece itself did; more than that and a mistake would keep itself alive. */
                const FgCmd *was = &prev->c[top[k].j];
                const float off = top[k].d + (was->how >= FG_HOW_CORNER ? 0.0f : 0.25f * fg_jump(was, cx - px, cy - py));
                const float rank = sure ? (float)sc : (float)sc * (1.0f - (off > 60.0f ? 60.0f : off) / 100.0f);
                const int same = bi >= 0 && fabsf(cx - px - bhx) < 4.0f && fabsf(cy - py - bhy) < 4.0f;
                if (rank > best) {
                    if (!same) second = best;
                    best = rank; bi = si; bj = top[k].j; bhx = cx - px; bhy = cy - py; bfrac = of ? (float)sc / (float)of : 0.0f;
                } else if (!same && rank > second) second = rank;
            }
        }
        if (bi < 0 || (sure && (best < 6.0f || bfrac < 0.6f || second > 0.6f * best || !fg_plausible(&cur->c[bi], &prev->c[bj], flow, bhx, bhy)))) continue;
        fg_accept(prev, cur, bi, bj, how);
        fg_grow(prev, cur, loose, how);
    }
}

static void fg_unpair(FgList *prev, FgCmd *c) {
    prev->c[c->next].used = 0;
    c->used = 0; c->moved = 0; c->how = FG_HOW_NONE; c->next = -1;
}
/* Drop an anchor that contradicts two or more others at its own corners. */
static void fg_validate_anchors(FgList *prev, FgList *cur, const int32_t *anch, int n) {
    for (int k = 0; k < n; k++) {
        const FgCmd *c = &cur->c[anch[k]];
        if (kFgPos[c->op].nv != 3) continue;
        for (int v = 0; v < 3; v++) fg_vert_add(s_fgv_of[anch[k]][v], c->vdx[v], c->vdy[v]);
    }
    for (int k = 0; k < n; k++) {
        FgCmd *c = &cur->c[anch[k]];
        if (kFgPos[c->op].nv != 3) continue;
        int bad = 0, good = 0;
        for (int v = 0; v < 3; v++) {
            const int32_t id = s_fgv_of[anch[k]][v];
            if (id < 0) continue;
            for (int m = 0; m < FG_MODES && s_fgv[id].n[m]; m++) {
                const float t = fg_tol(c->vdx[v], c->vdy[v]);
                const int mine = fabsf(s_fgv[id].dx[m] - c->vdx[v]) <= t && fabsf(s_fgv[id].dy[m] - c->vdy[v]) <= t;
                if (mine) good += s_fgv[id].n[m] - 1; else bad += s_fgv[id].n[m];
            }
        }
        if (bad >= 2 && bad > good) fg_unpair(prev, c);
    }
    for (int i = 0; i < s_fgv_n; i++) memset(s_fgv[i].n, 0, sizeof s_fgv[i].n);
    for (int k = 0; k < n; k++) {
        const FgCmd *c = &cur->c[anch[k]];
        if (c->next < 0 || kFgPos[c->op].nv != 3) continue;
        for (int v = 0; v < 3; v++) fg_vert_add(s_fgv_of[anch[k]][v], c->vdx[v], c->vdy[v]);
    }
}

/* Faces that cross the camera plane (FG_PROJ) have no screen position to
 * pair by; their homogeneous corners do. x and y are pixels times depth. */
static void fg_proj_pairs(FgList *prev, FgList *cur) {
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (c->op != FG_PROJ) continue;
        float ext = 0.0f, best = 1e30f; int bj = -1, steps = 0;
        for (int v = 0; v < 3; v++) {
            const int w = (v + 1) % 3;
            ext += fabsf(c->pcx[v] - c->pcx[w]) + fabsf(c->pcy[v] - c->pcy[w]) + 160.0f * fabsf(c->pq[v] - c->pq[w]);
        }
        for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512; j = fg_chain_next(prev, j, 0), steps++) {
            const FgCmd *p = &prev->c[j];
            if (!fg_same_look(c, p, 0)) continue;
            float d = 0.0f;
            for (int v = 0; v < 3; v++)
                d += fabsf(c->pcx[v] - p->pcx[v]) + fabsf(c->pcy[v] - p->pcy[v]) + 160.0f * fabsf(c->pq[v] - p->pq[v]);
            if (d < best) { best = d; bj = j; }
        }
        if (bj >= 0 && best < ext) { prev->c[bj].used = 1; c->used = 1; c->next = bj; c->how = FG_HOW_ANCHOR; }
    }
}

/* Solve the motion of every world primitive of `cur`. Returns how many
 * primitives (world and backdrop) have a partner. */
static int fg_solve(FgList *prev, FgList *cur) {
    memset(s_fg_kh, 0xff, sizeof s_fg_kh); memset(s_fg_lh, 0xff, sizeof s_fg_lh);
    memset(s_fgv_hash, 0xff, sizeof s_fgv_hash);
    s_fgv_n = 0; s_fgq_head = s_fgq_len = 0;
    for (int i = prev->n - 1; i >= 0; i--) {              /* chains end up in draw order */
        FgCmd *p = &prev->c[i];
        if (p->op < FG_FIRST_PRIM) continue;
        p->used = 0;
        p->next = s_fg_kh[p->key & (FG_HASH - 1)]; s_fg_kh[p->key & (FG_HASH - 1)] = i;
        s_fg_lnext[i] = s_fg_lh[p->loose & (FG_HASH - 1)]; s_fg_lh[p->loose & (FG_HASH - 1)] = i;
    }
    fg_index_prev(prev);
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        s_fg_inq[i] = 0; s_fgv_of[i][0] = s_fgv_of[i][1] = s_fgv_of[i][2] = -1;
        if (c->op < FG_FIRST_PRIM) continue;
        c->used = 0; c->moved = 0; c->how = FG_HOW_NONE; c->next = -1;
        if (!fg_is_world(c) || kFgPos[c->op].nv != 3) continue;
        for (int v = 0; v < 3; v++) {
            float x, y; fg_corner(c, v, &x, &y);
            const int32_t id = fg_vert_id(x, y);
            s_fgv_of[i][v] = id;
            if (id >= 0) { s_fgv_next[i * 3 + v] = s_fgv[id].first; s_fgv[id].first = i * 3 + v; }
        }
    }
    /* How often each look occurs in this frame (reverse check of an anchor). */
    static int32_t ckh[FG_HASH], cnext[FG_MAX_CMDS];
    memset(ckh, 0xff, sizeof ckh);
    for (int i = cur->n - 1; i >= 0; i--) {
        if (cur->c[i].op < FG_FIRST_PRIM) continue;
        cnext[i] = ckh[cur->c[i].key & (FG_HASH - 1)]; ckh[cur->c[i].key & (FG_HASH - 1)] = i;
    }
    static int32_t anch[FG_MAX_CMDS]; int an = 0;
    /* Unmoved to the sub-pixel: fixed to the screen (HUD), or nothing moves. */
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (!fg_is_world(c)) continue;
        int steps = 0;
        for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512; j = fg_chain_next(prev, j, 0), steps++) {
            FgCmd *p = &prev->c[j];
            if (!fg_same_look(c, p, 0)) continue;
            int same = 1;
            for (int v = 0; v < kFgPos[c->op].nv && same; v++) {
                float ax, ay, bx, by; fg_corner(c, v, &ax, &ay); fg_corner(p, v, &bx, &by);
                if (fabsf(ax - bx) > 0.01f || fabsf(ay - by) > 0.01f) same = 0;
            }
            if (!same) continue;
            p->used = 1; c->used = 1; c->next = j; c->moved = 1; c->how = FG_HOW_STILL;
            for (int v = 0; v < 3; v++) c->vdx[v] = c->vdy[v] = 0.0f;
            anch[an++] = i;
            break;
        }
    }
    /* Only one possible partner within reach, in both directions. */
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || c->next >= 0) continue;
        float cx, cy; fg_centre(c, &cx, &cy);
        int cand = -1, count = 0, steps = 0;
        for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512 && count < 2; j = fg_chain_next(prev, j, 0), steps++) {
            const FgCmd *p = &prev->c[j];
            if (!fg_same_look(c, p, 0)) continue;
            if (fabsf(cx - s_fg_pg[j].cx) > FG_REACH || fabsf(cy - s_fg_pg[j].cy) > FG_REACH) continue;
            cand = j; count++;
        }
        if (count != 1) continue;
        float px, py; fg_centre(&prev->c[cand], &px, &py);
        int back = 0; steps = 0;
        for (int32_t k = ckh[c->key & (FG_HASH - 1)]; k >= 0 && steps < 512 && back < 2; k = cnext[k], steps++) {
            const FgCmd *d = &cur->c[k];
            if (d->key != c->key || d->op != c->op || d->next >= 0) continue;
            float qx, qy; fg_centre(d, &qx, &qy);
            if (fabsf(qx - px) <= FG_REACH && fabsf(qy - py) <= FG_REACH) back++;
        }
        if (back != 1) continue;
        c->next = cand; c->used = 1; c->moved = 1; c->how = FG_HOW_ANCHOR;
        prev->c[cand].used = 1;
        for (int v = 0; v < kFgPos[c->op].nv; v++) {
            float ax, ay, bx, by; fg_corner(c, v, &ax, &ay); fg_corner(&prev->c[cand], v, &bx, &by);
            c->vdx[v] = ax - bx; c->vdy[v] = ay - by;
        }
        anch[an++] = i;
    }
    fg_accel_find(prev, cur, anch, an);
    for (int k = 0; k < an; k++) {
        FgCmd *c = &cur->c[anch[k]];
        if (c->how != FG_HOW_ANCHOR) continue;
        float dx, dy; fg_mean_motion(c, &dx, &dy);
        if (!fg_plausible(c, &prev->c[c->next], NULL, dx, dy)) fg_unpair(prev, c);
    }
    fg_validate_anchors(prev, cur, anch, an);
    const FgFlow flow = fg_flow_find(prev, cur, anch, an);
    s_fg_flow_last = flow;
    /* A pair has to look alike and lie within 2 px of where the camera motion
     * puts it, so for pairing a fit that explains a sixth of the picture will
     * do (chance alone explains a thirtieth). */
    const FgFlow *const cam = flow.inliers >= 24 && flow.inliers * 6 >= flow.n ? &flow : NULL;
    if (cam) fg_flow_accept(prev, cur, cam, 0);
    for (int k = 0; k < an; k++) if (cur->c[anch[k]].next >= 0) fg_queue_around(cur, anch[k]);
    fg_grow(prev, cur, 0, FG_HOW_MESH);
    for (int sure = 1; sure >= 0; sure--) {
        fg_sample_all(cur);
        fg_seed_components(prev, cur, cam, 0, FG_HOW_SEED, sure);
    }
    if (cam) fg_flow_accept(prev, cur, cam, 1);
    for (int i = 0; i < cur->n; i++)                      /* scrolled textures */
        if (fg_is_world(&cur->c[i]) && kFgPos[cur->c[i].op].nv == 3 && cur->c[i].next < 0 && fg_is_textured(cur->c[i].op)) {
            fg_try_mesh(prev, cur, i, 1, FG_HOW_LOOSE);
            fg_grow(prev, cur, 1, FG_HOW_LOOSE);
        }
    fg_sample_all(cur);
    fg_seed_components(prev, cur, cam, 1, FG_HOW_LOOSE, 0);
    fg_sample_all(cur);

    /* Rectangles and lines: nearest to where the surroundings say, or unmoved. */
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || kFgPos[c->op].nv == 3 || c->next >= 0) continue;
        float cx, cy, ex, ey; fg_centre(c, &cx, &cy);
        const FgField f = fg_field(cx, cy, (float)i, cur->n, 1);
        fg_field_at(&f, cx, cy, &ex, &ey);
        int best = -1, steps = 0; float bd = 32.0f;
        for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512; j = fg_chain_next(prev, j, 0), steps++) {
            const FgCmd *p = &prev->c[j];
            if (!fg_same_look(c, p, 0)) continue;
            float px, py; fg_centre(p, &px, &py);
            const float d0 = hypotf(cx - px, cy - py), d1 = hypotf(cx - px - ex, cy - py - ey);
            const float d = d0 < d1 ? d0 : d1;
            if (d < bd) { bd = d; best = j; }
        }
        if (best >= 0) fg_accept(prev, cur, i, best, FG_HOW_SEED);
        else if (f.ok) { for (int v = 0; v < 3; v++) { c->vdx[v] = ex; c->vdy[v] = ey; } c->moved = 1; c->how = FG_HOW_FIELD; }
    }
    /* Pieces without a partner follow their corners, then the field. */
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (!fg_is_world(c) || kFgPos[c->op].nv != 3 || c->next >= 0) continue;
        int have[3] = { 0, 0, 0 }, found = 0; float sx = 0, sy = 0;
        for (int v = 0; v < 3; v++) {
            const int m = fg_vert_main(s_fgv_of[i][v]);
            if (m < 0) continue;
            c->vdx[v] = s_fgv[s_fgv_of[i][v]].dx[m]; c->vdy[v] = s_fgv[s_fgv_of[i][v]].dy[m];
            have[v] = 1; found++; sx += c->vdx[v]; sy += c->vdy[v];
        }
        if (found < 3) {
            float cx, cy; fg_centre(c, &cx, &cy);
            const FgField f = fg_field(cx, cy, (float)i, cur->n, 0);
            for (int v = 0; v < 3; v++) {
                if (have[v]) continue;
                /* The camera fit holds for depths it was measured at, not far nearer. */
                if (flow.ok && c->iz[v] > 0.0f && c->iz[v] < 3.0f * (float)flow.rs && !found) { float x, y; fg_corner(c, v, &x, &y); fg_flow_at(&flow, x, y, c->iz[v], &c->vdx[v], &c->vdy[v]); }
                else if (found) { c->vdx[v] = sx / (float)found; c->vdy[v] = sy / (float)found; }
                else if (f.ok) { float x, y; fg_corner(c, v, &x, &y); fg_field_at(&f, x, y, &c->vdx[v], &c->vdy[v]); }
                else c->vdx[v] = c->vdy[v] = 0.0f;
            }
        }
        for (int v = 0; v < 3; v++) {             /* a guess never flings a piece across the screen */
            if (have[v]) continue;
            c->vdx[v] = fminf(fmaxf(c->vdx[v], -128.0f), 128.0f);
            c->vdy[v] = fminf(fmaxf(c->vdy[v], -128.0f), 128.0f);
        }
        c->moved = 1; c->how = (uint8_t)(found ? FG_HOW_CORNER : FG_HOW_FIELD);
        for (int v = 0; v < 3; v++)               /* later pieces on the same new corner must not tear from it */
            if (!have[v]) fg_vert_add(s_fgv_of[i][v], c->vdx[v], c->vdy[v]);
    }
    fg_proj_pairs(prev, cur);
    int matched = 0;
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (c->op >= FG_FIRST_PRIM && !fg_is_backdrop(c) && c->next >= 0) matched++;
    }
    return matched;
}

/* The 2D backdrop (sky and water tile grids) moves as one rigid layer whose
 * tiles repeat, so every same-look pair votes for a shift and the shift most
 * tiles agree on wins; among equals, the smaller one. A nearest-tile pairing
 * reads a fast scroll as a slow one the other way. Returns the tiles with a
 * partner under that shift. Shapes fixed to the screen do not vote. */
static int s_fg_bd_have;
static float s_fg_bd_dx, s_fg_bd_dy;
static int fg_bd_fixed(const FgCmd *c, const FgCmd *p) {
    if (fg_is_textured(c->op) || p->op != c->op) return 0;
    for (int v = 0; v < kFgPos[c->op].nv; v++) {
        float ax, ay, bx, by; fg_corner(c, v, &ax, &ay); fg_corner(p, v, &bx, &by);
        if (fabsf(ax - bx) > 0.01f || fabsf(ay - by) > 0.01f) return 0;
    }
    return 1;
}
static int fg_solve_backdrop(FgList *prev, FgList *cur) {     /* after fg_solve (uses its chains) */
    enum { R = 72, W = 2 * R + 1 };
    static uint16_t votes[W][W];
    memset(votes, 0, sizeof votes);
    s_fg_bd_have = 0; s_fg_bd_dx = s_fg_bd_dy = 0.0f;
    int any = 0;
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (c->op < FG_FIRST_PRIM || !fg_is_backdrop(c)) continue;
        float cx, cy; fg_centre(c, &cx, &cy);
        int steps = 0, fixed = 0;
        for (int pass = 0; pass < 2 && !fixed; pass++) {
            steps = 0;
            for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512; j = fg_chain_next(prev, j, 0), steps++) {
                const FgCmd *p = &prev->c[j];
                if (p->op != c->op || p->key != c->key) continue;
                if (!pass) { if (fg_bd_fixed(c, p)) { fixed = 1; break; } continue; }
                float px, py; fg_centre(p, &px, &py);
                const int dx = (int)floorf(cx - px + 0.5f), dy = (int)floorf(cy - py + 0.5f);
                if (dx < -R || dx > R || dy < -R || dy > R) continue;
                if (votes[dy + R][dx + R] < 65535) votes[dy + R][dx + R]++;
                any = 1;
            }
        }
    }
    if (!any) return 0;
    int bx = 0, by = 0, bv = -1, bm = 1 << 30;
    for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
        const int v = votes[y][x], m = (x - R) * (x - R) + (y - R) * (y - R);
        if (v > bv || (v == bv && m < bm)) { bv = v; bm = m; bx = x - R; by = y - R; }
    }
    /* Sub-pixel value: mean of the pairs that voted for the winner. */
    double sx = 0, sy = 0; int n = 0, tiles = 0;
    for (int i = 0; i < cur->n; i++) {
        FgCmd *c = &cur->c[i];
        if (c->op < FG_FIRST_PRIM || !fg_is_backdrop(c)) continue;
        float cx, cy; fg_centre(c, &cx, &cy);
        int steps = 0;
        c->next = -1; c->used = 0;
        for (int32_t j = fg_chain_first(c, 0); j >= 0 && steps < 512; j = fg_chain_next(prev, j, 0), steps++) {
            const FgCmd *p = &prev->c[j];
            if (p->op != c->op || p->key != c->key) continue;
            if (fg_bd_fixed(c, p)) { c->next = j; c->used = 1; tiles++; break; }
            float px, py; fg_centre(p, &px, &py);
            const float dx = cx - px, dy = cy - py;
            if (fabsf(dx - (float)bx) > 1.0f || fabsf(dy - (float)by) > 1.0f) continue;
            c->next = j; c->used = 1; tiles++;
            sx += dx; sy += dy; n++;
            break;
        }
    }
    if (n) { s_fg_bd_have = 1; s_fg_bd_dx = (float)(sx / n); s_fg_bd_dy = (float)(sy / n); }
    return tiles;
}

#endif /* PSX_VK_FRAMEGEN_MATCH_H */
