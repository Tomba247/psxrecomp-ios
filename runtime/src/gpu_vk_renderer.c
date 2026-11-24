/* gpu_vk_renderer.c — hardware Vulkan renderer backend.  See gpu_vk_renderer.h.
 *
 * GPU-authoritative VRAM (mirrors the GL backend's model): the canonical copy of
 * VRAM is a Vulkan image (s_vram_img, RGBA8, 1024*S x 512*S).  A CPU mirror is
 * kept only for readback (screenshots / debug server / 24-bit FMV present) and
 * is synced lazily.  Present blits the displayed region straight from the VRAM
 * image to the swapchain — no per-frame readback.
 *
 * Vulkan is loaded entirely at runtime (SDL_Vulkan_LoadLibrary +
 * vkGetInstanceProcAddr), so the exe never link-depends on vulkan-1 and falls
 * back to software when no ICD is present.
 *
 * BUILD STATUS: instance/device/swapchain + VRAM image + fills + CPU<->VRAM
 * transfers + present are implemented; geometry pipelines (triangles/rects/
 * lines), texturing/CLUT, mask-bit, semi-transparency, SSAA and the native-wide
 * compositor are added in later phases — those vtable entries currently abort
 * via psx_fatal_halt rather than silently no-op (no stubs).
 */

#include "gpu.h"
#include "gpu_render.h"
#include "gpu_vk_renderer.h"
#include "psx_ios_boot.h"
#include "video_telemetry.h"
#include "gpu_sw_renderer.h"
#include "host_osd.h"
#include "host_touch_ui.h"
#include "frame_pacing.h"
#include "gpu_hd_textures.h"
#include "psx_savestate_menu.h"
#include "psx_host_menu.h"
#include "psx_rewind.h"
#include "crash_trace.h"
#include "gpu_vk_upload.h"
#include "gpu_projective_clip.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#if !defined(PSX_HAVE_VULKAN)
/* Vulkan SDK headers/glslc were not found at configure time: compile an inert
 * backend so the runtime still builds and simply falls back to software. */
int  vk_renderer_init_context(struct SDL_Window *win) { (void)win; return 0; }
void vk_renderer_shutdown(void) {}
int  vk_renderer_present_vram(int a,int b,int c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
int  vk_renderer_present_wide(int a,int b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
void vk_renderer_present_cpu(const uint32_t*p,int w,int h,int l,int f){(void)p;(void)w;(void)h;(void)l;(void)f;}
void vk_renderer_present_blank(void){}
void vk_renderer_sync_cpu(void){}
void vk_renderer_restage_vram_after_savestate(void){}
void vk_renderer_set_present_mode(int m){(void)m;}
void vk_renderer_set_display_aspect(int n,int d){(void)n;(void)d;}
void vk_renderer_display_flip(void){}
int  vk_renderer_projective_supported(void){return 0;}
void vk_renderer_draw_projected_triangle(const PSXProjectedVertex *v,uint16_t tp,uint16_t cx,uint16_t cy,int raw,int semi,int persp){(void)v;(void)tp;(void)cx;(void)cy;(void)raw;(void)semi;(void)persp;}
void vk_renderer_set_triangle_depth_scale(float q){(void)q;}
int vk_renderer_hd_texture_cached(uint64_t k, uint64_t g, uint32_t *w, uint32_t *h){ (void)k; (void)g; (void)w; (void)h; return 0; }
int  vk_perf_json(char *out,int cap,int count){(void)count; return cap>2?snprintf(out,cap,"[]"):0;}
const GpuRenderBackend *vk_backend_get(void) { return 0; }

#else  /* PSX_HAVE_VULKAN */

#include "psx_sdl.h"
#if defined(PSX_SDL3)
#include <SDL3/SDL_vulkan.h>
#else
#include <SDL_vulkan.h>
#endif
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "vk_shaders_spv.h"   /* generated: spv_geo_vert/frag, spv_geo_tex_vert/frag,
                               *            spv_pack_comp, spv_blit_vert/frag */

#define VRAM_W 1024
#define VRAM_H 512

static int s_aspect_num = 4, s_aspect_den = 3;

void vk_renderer_set_display_aspect(int num, int den) {
    if (num <= 0 || den <= 0) { num = 4; den = 3; }
    s_aspect_num = num; s_aspect_den = den;
}

/* ---- dynamic loader ---------------------------------------------------- */
/* vkGetInstanceProcAddr comes from SDL; everything else is loaded through it
 * (global/instance scope) or vkGetDeviceProcAddr (device scope). */
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr;
static PFN_vkGetDeviceProcAddr   p_vkGetDeviceProcAddr;

#define VK_GLOBAL_FUNCS(X) \
    X(vkCreateInstance) \
    X(vkEnumerateInstanceExtensionProperties) \
    X(vkEnumerateInstanceLayerProperties)

#define VK_INSTANCE_FUNCS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkCreateDevice) \
    X(vkDestroySurfaceKHR)

#define VK_DEVICE_FUNCS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkDeviceWaitIdle) \
    X(vkQueueWaitIdle) \
    X(vkCreateSwapchainKHR) \
    X(vkDestroySwapchainKHR) \
    X(vkGetSwapchainImagesKHR) \
    X(vkAcquireNextImageKHR) \
    X(vkQueuePresentKHR) \
    X(vkCreateImageView) \
    X(vkDestroyImageView) \
    X(vkCreateImage) \
    X(vkDestroyImage) \
    X(vkGetImageMemoryRequirements) \
    X(vkBindImageMemory) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkMapMemory) \
    X(vkUnmapMemory) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkFreeCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkQueueSubmit) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdClearColorImage) \
    X(vkCmdClearDepthStencilImage) \
    X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) \
    X(vkCmdBlitImage) \
    X(vkCmdCopyImage) \
    X(vkCreateRenderPass) \
    X(vkDestroyRenderPass) \
    X(vkCreateFramebuffer) \
    X(vkDestroyFramebuffer) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateGraphicsPipelines) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkCreateSampler) \
    X(vkDestroySampler) \
    X(vkCmdBeginRenderPass) \
    X(vkCmdEndRenderPass) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindVertexBuffers) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdSetViewport) \
    X(vkCmdSetScissor) \
    X(vkCmdSetStencilReference) \
    X(vkCmdDispatch) \
    X(vkCmdDraw) \
    X(vkCmdClearAttachments)

#define DECL_FP(n) static PFN_##n p_##n;
VK_GLOBAL_FUNCS(DECL_FP)
VK_INSTANCE_FUNCS(DECL_FP)
VK_DEVICE_FUNCS(DECL_FP)
#undef DECL_FP

/* ---- state ------------------------------------------------------------- */
static SDL_Window      *s_win;
static VkInstance       s_instance;
static VkSurfaceKHR     s_surface;
static VkPhysicalDevice s_phys;
static VkDevice         s_dev;
static uint32_t         s_qfam = 0;       /* graphics+present queue family */
static VkQueue          s_queue;
static VkCommandPool    s_cmd_pool;       /* present command buffers */
static VkCommandPool    s_work_pool;      /* one-shot work CBs (reset at gpu_sync) */
static int              s_work_pending;   /* submitted work not yet waited */
static VkPhysicalDeviceMemoryProperties s_memprops;

/* Staging buffers whose GPU consumption is still in flight; freed at gpu_sync
 * after the queue goes idle (work is submitted without per-op waits now). */
#define PENDING_STAGING_MAX 1024
static VkBuffer       s_pending_buf[PENDING_STAGING_MAX];
static VkDeviceMemory s_pending_mem[PENDING_STAGING_MAX];
static int            s_pending_n;

#define STAGING_CACHE_MAX 16
typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
    void *map;
    int busy;
} StagingCacheEntry;
static StagingCacheEntry s_staging_cache[STAGING_CACHE_MAX];

static void staging_release(VkBuffer buf, VkDeviceMemory mem) {
    for (int i = 0; i < STAGING_CACHE_MAX; ++i) {
        if (s_staging_cache[i].buf == buf && s_staging_cache[i].mem == mem) {
            s_staging_cache[i].busy = 0;
            return;
        }
    }
    p_vkUnmapMemory(s_dev, mem);
    p_vkDestroyBuffer(s_dev, buf, NULL);
    p_vkFreeMemory(s_dev, mem, NULL);
}

static void staging_destroy(VkBuffer buf, VkDeviceMemory mem) {
    for (int i = 0; i < STAGING_CACHE_MAX; ++i) {
        StagingCacheEntry *entry = &s_staging_cache[i];
        if (entry->buf == buf && entry->mem == mem) {
            if (entry->map) p_vkUnmapMemory(s_dev, mem);
            p_vkDestroyBuffer(s_dev, buf, NULL);
            p_vkFreeMemory(s_dev, mem, NULL);
            memset(entry, 0, sizeof *entry);
            return;
        }
    }
    p_vkUnmapMemory(s_dev, mem);
    p_vkDestroyBuffer(s_dev, buf, NULL);
    p_vkFreeMemory(s_dev, mem, NULL);
}

static VkSwapchainKHR   s_swapchain;
static VkFormat         s_sc_format;
static VkExtent2D       s_sc_extent;
static uint32_t         s_sc_count;
static VkImage          s_sc_images[8];
static int              s_present_mode_req = 1;   /* 1 tear-free, 0 IMMEDIATE, -1 MAILBOX */

/* Per-frame sync (double-buffered command recording). */
#define VK_FRAMES 2
static VkCommandBuffer  s_cmd[VK_FRAMES];
static VkSemaphore      s_sem_acquire[VK_FRAMES];
static VkSemaphore      s_sem_render[VK_FRAMES];
static VkFence          s_fence[VK_FRAMES];
static uint32_t         s_frame_idx;

/* GPU-authoritative VRAM image + CPU mirror. */
static int        s_scale = 1;
static VkImage    s_vram_img;
static VkDeviceMemory s_vram_mem;
static VkImageLayout s_vram_layout = VK_IMAGE_LAYOUT_UNDEFINED;  /* tracked */
static uint16_t  *s_vram;                 /* gpu.c's CPU VRAM array (1024x512) */
static int        s_cpu_dirty;            /* CPU mirror stale vs GPU image */

static int        s_ready;                /* full pipeline ready (Phase 2+) */
static int        s_ctx_ok;              /* instance/device/swapchain up */

static int        s_texfilter;

/* ---- geometry / render targets ----------------------------------------- */
static VkImageView     s_vram_view;       /* RGBA8 hr render target view */
static VkRenderPass    s_rpass;           /* color (hr) + depth/stencil */
/* MSAA (settings panel, applies at start): the native-wide surfaces -- what
 * gameplay displays -- render into multisampled colour + stencil images and
 * resolve into the single-sample surface at the end of every wide pass.
 * Canonical VRAM stays single-sample (it is read back and sampled as data). */
static int             s_msaa = 1;        /* 1, 2 or 4 samples */
static VkRenderPass    s_rpass_ms;        /* MS colour + MS stencil + resolve */
static int             s_pipe_ms;         /* pipelines bind for the MS pass */
static int             s_make_samples = 1;
static VkFramebuffer   s_fbo;

/* Native R16_UINT raw VRAM mirror: the canonical 1555 bits, sampled in the
 * textured fragment shader (CLUT decode needs the exact index bits, which the
 * lossy RGBA8 image cannot provide) and read back to the CPU mirror. Kept
 * coherent with the hr render target by the PACK compute pass. */
static VkImage         s_raw_img;
static VkDeviceMemory  s_raw_mem;
static VkImageView     s_raw_view;
static VkImageLayout   s_raw_layout = VK_IMAGE_LAYOUT_UNDEFINED;

/* Depth/stencil attachment: stencil bit0 mirrors VRAM bit15 (mask). */
static VkImage         s_ds_img;
static VkDeviceMemory  s_ds_mem;
static VkImageView     s_ds_view;
static VkFormat        s_ds_format;
/* Both candidate formats (D24S8 / D32S8) are COMBINED depth+stencil: views,
 * barriers and image clears must name BOTH aspects unless
 * separateDepthStencilLayouts is enabled (it is not). Stencil-only leaves the
 * depth half in UNDEFINED; on AMD (no D24S8 -> D32S8) depth and stencil share
 * compression metadata. */
/* Only the stencil is used (PSX mask bit). A stencil-only format is one byte
 * per pixel instead of eight for depth32+stencil8, which matters at high
 * internal resolutions and with MSAA. */
static VkImageAspectFlags s_ds_aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
#define DS_ASPECT s_ds_aspect

/* Scratch hr image for VRAM->VRAM copies (resolves overlap). */
static VkImage         s_scratch_img;
static VkDeviceMemory  s_scratch_mem;
static VkImageView     s_scratch_view;
static VkImageLayout   s_scratch_layout = VK_IMAGE_LAYOUT_UNDEFINED;

/* Native-res RGBA8 upload staging image (CPU->VRAM blit source). */
static VkImage         s_up_img;
static VkDeviceMemory  s_up_mem;
static VkImageView     s_up_view;
static VkImageLayout   s_up_layout = VK_IMAGE_LAYOUT_UNDEFINED;

/* Samplers + descriptor sets. One nearest sampler serves every fetch. */
/* ---- native-wide compositor surfaces (mirrors the GL backend) -----------
 * Canonical VRAM (the hr image) stays 100% faithful; framebuffer-targeting
 * batches are ALSO drawn into a per-base_x wide surface, x-translated in the
 * vertex shader via the existing u_xoff/u_xhalf push constants. Each surface
 * carries its OWN depth-stencil image (PSX mask-bit mirror; Part-A lesson:
 * never render stencil-enabled passes into a stencil-less target) and its own
 * VkFramebuffer on the SHARED render pass, so every existing pipeline works
 * unchanged. The mirror is appended to the SAME one-shot command buffer as the
 * canonical pass — one extra render pass per flushed batch, no extra submits. */
#define VK_WIDE_MAX_SURF 6
static VkImage        s_wide_img[VK_WIDE_MAX_SURF];
static VkDeviceMemory s_wide_mem[VK_WIDE_MAX_SURF];
static VkImageView    s_wide_view[VK_WIDE_MAX_SURF];
static VkImage        s_wide_ds_img[VK_WIDE_MAX_SURF];
static VkDeviceMemory s_wide_ds_mem[VK_WIDE_MAX_SURF];
static VkImageView    s_wide_ds_view[VK_WIDE_MAX_SURF];
static VkFramebuffer  s_wide_fb[VK_WIDE_MAX_SURF];
static VkImageLayout  s_wide_layout[VK_WIDE_MAX_SURF];
static VkImage        s_wide_ms_img[VK_WIDE_MAX_SURF];    /* MSAA colour (s_msaa > 1) */
static VkDeviceMemory s_wide_ms_mem[VK_WIDE_MAX_SURF];
static VkImageView    s_wide_ms_view[VK_WIDE_MAX_SURF];
static int            s_wide_base[VK_WIDE_MAX_SURF] = { -1, -1, -1, -1, -1, -1 };
static int            s_wide_w = 0;        /* wide width (native px); 0 = disabled */
static int            s_wide_offset = 0;   /* centering OFFSET (native px) */
static int            s_wide_cur = -1;     /* active mirror surface index, -1 = none */
static int            s_wide_cur_base = 0;
/* Frame generation (see the fg_* block): while s_fg_replay is set the draw
 * paths render ONLY the wide mirror pass, into surface s_fg_surf. */
static int            s_fg_replay = 0;
static int            s_fg_surf = -1;      /* surface the replay draws into */
static int            s_fg_surfs[3] = { -1, -1, -1 };   /* generated pictures of this frame */
static int            s_fg_n = 0;          /* how many of them are valid (1 = 60 fps, 3 = 120) */
static int            s_fg_next = 0;       /* next one to show */
static float          s_fg_t = 0.5f;       /* position of the picture being built, prev -> cur */
static int            s_fg_120_off = 0;    /* 120 turned itself off (display cannot keep up) */
static int            s_fg_hold = 0;       /* a generated picture waits for the next present */
static uint32_t       s_fg_hold_stamp = 0;
static void wide_free_all(void);           /* defined with the wide helpers below */
static void hd_vk_shutdown(void);
static void hd_vk_reap(void);

static VkSampler       s_samp;
static VkDescriptorPool s_dpool;
static VkDescriptorSetLayout s_dsl_tex;   /* binding0: raw usampler2D (frag) */
static VkDescriptorSetLayout s_dsl_pack;  /* binding0: hr sampler, binding1: raw storage (compute) */
static VkDescriptorSetLayout s_dsl_blit;  /* binding0: src sampler (frag) */
static VkDescriptorSet s_ds_tex;
/* ---- HD texture replacements (texture packs) ------------------------------
 * A matched draw samples a replacement image instead of native texels. Each
 * cached replacement is an RGBA8 image with a mip chain and its own
 * descriptor set (set 1 of the textured pipeline); draws without one bind a
 * 1x1 dummy. The replacement is a batch key. */
#define HD_VK_CAP 4096
#define HD_VK_HINT 8192
typedef struct {
    uint64_t key, gen;
    VkImage img; VkDeviceMemory mem; VkImageView view; VkDescriptorSet ds;
    uint32_t w, h, last; size_t bytes;
} HdVkTex;
static HdVkTex          s_hdv[HD_VK_CAP];
static uint16_t         s_hd_hint[HD_VK_HINT];
/* Evicted images wait here until the queue is idle (pending draws may still
 * sample them); their slots stay reserved until then. */
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; int slot; } s_hd_retired[HD_VK_CAP];
static int              s_hd_retired_n;
static uint8_t          s_hd_slot_busy[HD_VK_CAP];
/* Draw memo: the answer of the texture-pack lookup for one draw footprint,
 * valid until a VRAM write touches the texels or the CLUT it read. Lookups
 * hash VRAM and search the pack; steady scenes repeat the same few hundred
 * footprints every frame. Negative answers expire (a decode may be pending). */
#define HD_MEMO_N 16384
typedef struct {
    uint64_t key;            /* 0 = empty */
    uint64_t tex_key, tex_gen;
    int16_t  slot;           /* -1 = no replacement */
    int16_t  x0, y0, x1, y1; /* VRAM words read for texels */
    int16_t  cx0, cx1, cy;   /* CLUT words (cx1 < cx0 = none) */
    uint8_t  mode;
    float    src[4];
    uint32_t stamp;          /* flip count when stored (negatives expire) */
} HdMemo;
static HdMemo           s_hd_memo[HD_MEMO_N];
static int              s_hd_memo_live;
static int              s_hd_bb[4] = { 32767, 32767, -1, -1 };   /* VRAM box all entries read from */
static unsigned         s_hd_stat_miss, s_hd_stat_upload, s_hd_stat_direct, s_hd_stat_composed, s_hd_stat_inval;
/* Animated textures (water, palette cycling): the game rewrites those VRAM
 * words every few frames, and a pack rarely holds every frame of the
 * animation, so the picture alternated between replacement and original.
 * VRAM is tracked in 64x64-word tiles; a tile that keeps being rewritten is
 * "animated" (see hd_flip_tick) and its texels are never replaced. */
static int s_hd_ready;
#define HD_TILE_W 16
#define HD_TILE_H 8
static uint32_t s_hd_wr_last[HD_TILE_H][HD_TILE_W];
static uint8_t  s_hd_wr_cnt[HD_TILE_H][HD_TILE_W], s_hd_anim[HD_TILE_H][HD_TILE_W];
static uint8_t  s_hd_run[HD_TILE_H][HD_TILE_W], s_hd_quiet[HD_TILE_H][HD_TILE_W];
static unsigned s_hd_flip;      /* flips seen by the tracker */
static int hd_rect_animated(int x0, int y0, int x1, int y1) {
    if (x1 < x0 || y1 < y0) return 0;
    for (int ty = (y0 & 511) >> 6; ty <= ((y1 & 511) >> 6); ty++)
        for (int tx = (x0 & 1023) >> 6; tx <= ((x1 & 1023) >> 6); tx++)
            if (s_hd_anim[ty][tx]) return 1;
    return 0;
}
static void hd_note_write(int x0, int y0, int x1, int y1) {
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 > VRAM_W - 1) x1 = VRAM_W - 1; if (y1 > VRAM_H - 1) y1 = VRAM_H - 1;
    for (int ty = y0 >> 6; ty <= (y1 >> 6); ty++)
        for (int tx = x0 >> 6; tx <= (x1 >> 6); tx++)
            if (s_hd_wr_last[ty][tx] != s_hd_flip + 1u) {
                s_hd_wr_last[ty][tx] = s_hd_flip + 1u;
                if (s_hd_wr_cnt[ty][tx] < 255) s_hd_wr_cnt[ty][tx]++;
            }
}
static void hd_memo_invalidate(int x0, int y0, int x1, int y1) {
    if (s_hd_ready) hd_note_write(x0, y0, x1, y1);
    if (!s_hd_memo_live) return;
    if (x1 < s_hd_bb[0] || x0 > s_hd_bb[2] || y1 < s_hd_bb[1] || y0 > s_hd_bb[3]) return;   /* framebuffer writes */
    s_hd_stat_inval++;
    int live = 0;
    for (int i = 0; i < HD_MEMO_N; i++) {
        HdMemo *m = &s_hd_memo[i];
        if (!m->key) continue;
        const int tex = !(x1 < m->x0 || x0 > m->x1 || y1 < m->y0 || y0 > m->y1);
        const int clut = m->cx1 >= m->cx0 && !(y1 < m->cy || y0 > m->cy || x1 < m->cx0 || x0 > m->cx1);
        if (tex || clut) m->key = 0; else live++;
    }
    s_hd_memo_live = live;
    if (!live) { s_hd_bb[0] = s_hd_bb[1] = 32767; s_hd_bb[2] = s_hd_bb[3] = -1; }
}   /* (key,gen) -> slot+1, direct mapped */
static VkDescriptorSetLayout s_dsl_hd;
static VkDescriptorPool s_hd_pool;
static VkSampler        s_hd_samp;
static VkImage          s_hd_none_img; static VkDeviceMemory s_hd_none_mem; static VkImageView s_hd_none_view;
static VkDescriptorSet  s_hd_none_ds;
static size_t           s_hd_bytes, s_hd_budget;
static uint32_t         s_hd_clock;
static int              s_hd_ready;
static VkDescriptorSet  s_tb_hd_ds;      /* replacement of the open textured batch */
static VkDescriptorSet s_ds_pack;
/* Blit src image is rebound per blit; a ring of sets avoids updating a set still
 * in flight (work is no longer waited per-op). Cycled within a frame; the index
 * resets at gpu_sync (which waits the queue idle). */
#define BLIT_DESC_RING 64
static VkDescriptorSet s_ds_blit_ring[BLIT_DESC_RING];
static int s_ds_blit_idx;

/* ---- always-on per-frame perf ring (transition-stall diagnosis) ----------
 * Per-op counters accumulate into s_perf_cur; each present() snapshots them into
 * the ring keyed by present index, then resets. Query via the debug-server
 * "vk_perf" command (vk_perf_json). Op COUNTS alone reveal the dominant
 * per-frame cost (e.g. make_staging vkAllocateMemory churn during transitions:
 * a frame that jumps from ~10 to ~2000 allocs is the smoking gun). Always on. */
typedef struct {
    uint32_t present_idx;
    uint64_t present_qpc;
    uint32_t wait_us, acquire_us, present_us;
    uint32_t allocs, alloc_kb, oneshots, submits, syncs, pack_flushes,
             blits, upload_blocks, copy_rects, geo_flushes, tex_flushes,
             wide_passes, wide_clears;
} VkPerf;
#define VK_PERF_RING 256
static VkPerf s_perf_ring[VK_PERF_RING];
static uint32_t s_perf_head;          /* number of frames recorded (monotonic) */
static VkPerf s_perf_cur;             /* current (in-progress) frame */
static uint32_t s_present_idx;

static uint32_t perf_elapsed_us(uint64_t start) {
    uint64_t frequency = SDL_GetPerformanceFrequency();
    uint64_t ticks = SDL_GetPerformanceCounter() - start;
    return frequency ? (uint32_t)(ticks * 1000000u / frequency) : 0;
}

static unsigned s_flip_count;   /* GP1(05h) display flips = guest frames */
/* The stencil mirrors the PSX mask bit and is only ever READ by draws with
 * mask-check on. An opaque textured batch needs a second full shading pass
 * just to set that bit for its STP texels; while the game has not drawn with
 * mask-check for a while that pass is skipped (about half the fragment work
 * of textured geometry). It resumes with the first mask-check draw and stays
 * on for 600 frames after the last one. */
static unsigned s_mask_check_flip;     /* flip of the last mask-check draw */
static int      s_mask_check_ever;
static int      s_stencil_fix_force = -1;
static int stencil_fix_needed(void) {
    if (s_stencil_fix_force < 0) { const char *e = getenv("PSX_VK_STENCIL_FIX"); s_stencil_fix_force = e ? (*e == '1') : 2; }
    if (s_stencil_fix_force != 2) return s_stencil_fix_force;
    return s_mask_check_ever && (s_flip_count - s_mask_check_flip) < 600u;
}
static unsigned s_stat_fix_on, s_stat_fix_off;
/* Every 32 frames: re-classify the VRAM tiles; a change drops the memo. */
static void hd_flip_tick(void) {
    if (!s_hd_ready) return;
    if ((++s_hd_flip & 31u) != 0) return;
    int changed = 0;
    for (int ty = 0; ty < HD_TILE_H; ty++)
        for (int tx = 0; tx < HD_TILE_W; tx++) {
            /* A texture that is loaded is written once; one that is written
             * again and again is an animation, however slow (a waterfall
             * steps every few frames, a palette cycle once a second). Two
             * writes in a window, or a write in each of three windows in a
             * row, make the tile animated; it stays so until ten windows
             * (about ten seconds) pass without any write. */
            const uint8_t c = s_hd_wr_cnt[ty][tx];
            uint8_t a = s_hd_anim[ty][tx];
            if (c) { if (s_hd_run[ty][tx] < 255) s_hd_run[ty][tx]++; s_hd_quiet[ty][tx] = 0; }
            else   { s_hd_run[ty][tx] = 0; if (s_hd_quiet[ty][tx] < 255) s_hd_quiet[ty][tx]++; }
            if (c >= 2 || s_hd_run[ty][tx] >= 3) a = 1;
            else if (s_hd_quiet[ty][tx] >= 10) a = 0;
            if (a != s_hd_anim[ty][tx]) { s_hd_anim[ty][tx] = a; changed = 1; }
            s_hd_wr_cnt[ty][tx] = 0;
        }
    if (changed) {
        for (int i = 0; i < HD_MEMO_N; i++) s_hd_memo[i].key = 0;
        s_hd_memo_live = 0;
        s_hd_bb[0] = s_hd_bb[1] = 32767; s_hd_bb[2] = s_hd_bb[3] = -1;
    }
}
static uint64_t s_t_flush, s_t_close;   /* ticks in batch flushes / pass close */
static void perf_snapshot_present(void) {
    s_perf_cur.present_idx = s_present_idx++;
    s_perf_cur.present_qpc = SDL_GetPerformanceCounter();
    {   /* One log line every 600 presents: is the host keeping up? */
        static uint64_t t0, wait_us, acq_us, pres_us, flushes, passes;
        static unsigned n, flips0;
        flushes += s_perf_cur.tex_flushes + s_perf_cur.geo_flushes; passes += s_perf_cur.wide_passes;
        if (!t0) { t0 = s_perf_cur.present_qpc; flips0 = s_flip_count; }
        wait_us += s_perf_cur.wait_us; acq_us += s_perf_cur.acquire_us; pres_us += s_perf_cur.present_us;
        if (++n >= 600) {
            const double sec = (double)(s_perf_cur.present_qpc - t0) / (double)SDL_GetPerformanceFrequency();
            if (sec > 0.0)
                fprintf(stdout, "psxrecomp: vulkan: %.1f pictures/s, %.1f game frames/s; per picture: GPU wait %.2f ms, "
                                "acquire %.2f ms, present %.2f ms; scale %dx\n",
                        n / sec, (s_flip_count - flips0) / sec, wait_us / 1000.0 / n, acq_us / 1000.0 / n,
                        pres_us / 1000.0 / n, s_scale);
            if (sec > 0.0) {
                const double f = 1000.0 / (double)SDL_GetPerformanceFrequency() / n;
                fprintf(stdout, "psxrecomp: vulkan: per picture: %.0f batches in %.1f wide passes, "
                                "batch CPU %.2f ms, pass replay CPU %.2f ms\n",
                        (double)flushes / n, (double)passes / n, s_t_flush * f, s_t_close * f);
            }
            if (s_fg_n >= 2 && !s_fg_120_off && (acq_us + pres_us) / n > 4500) {
                s_fg_120_off = 1;   /* presents are blocking: the display is not taking 120 */
                fprintf(stdout, "psxrecomp: vulkan: 90/120 fps turned off, the display is not accepting pictures that fast\n");
            }
            fprintf(stdout, "psxrecomp: vulkan: mask-bit pass %s (%u batches with, %u without)\n",
                    stencil_fix_needed() ? "on" : "skipped", s_stat_fix_on, s_stat_fix_off);
            s_stat_fix_on = s_stat_fix_off = 0;
            if (s_hd_ready && gpu_hd_textures_active())
                fprintf(stdout, "psxrecomp: vulkan: HD textures per second: %.1f lookups (%.1f direct, %.1f assembled), "
                                "%.1f uploads, %.1f invalidating writes; cache %u MB\n",
                        s_hd_stat_miss / sec, s_hd_stat_direct / sec, s_hd_stat_composed / sec,
                        s_hd_stat_upload / sec, s_hd_stat_inval / sec, (unsigned)(s_hd_bytes >> 20));
            s_hd_stat_miss = s_hd_stat_direct = s_hd_stat_composed = s_hd_stat_upload = s_hd_stat_inval = 0;
            fflush(stdout);
            s_t_flush = s_t_close = 0; flushes = passes = 0;
            t0 = s_perf_cur.present_qpc; flips0 = s_flip_count; n = 0; wait_us = acq_us = pres_us = 0;
        }
    }
    s_perf_ring[s_perf_head % VK_PERF_RING] = s_perf_cur;
    s_perf_head++;
    memset(&s_perf_cur, 0, sizeof s_perf_cur);
}

/* Pipeline layouts (one per program family). */
static VkPipelineLayout s_pl_geo;
static VkPipelineLayout s_pl_tex;
static VkPipelineLayout s_pl_blit;
static VkPipelineLayout s_pl_pack;
static VkPipeline       s_pipe_pack;      /* compute */

/* Lazy graphics-pipeline cache keyed by (program, topology, blend, stencil,
 * cmask). program: 0 GEO, 1 TEX, 2 BLIT. blend: 0 off, 1..4 = PS1 modes 0..3
 * (+1). stencil: 0 SET (always+replace, dyn ref), 1 CHECK keep, 2 CHECK
 * invert. cmask: 0 writes colour, 1 disables ALL colour writes (stencil-only
 * fixup pass of an opaque textured batch — mirrors GL's glColorMask off). */
#define PIPE_PROGS 3
#define PIPE_TOPOS 2   /* 0 TRIANGLE_LIST, 1 LINE_LIST */
#define PIPE_BLENDS 5
#define PIPE_STENCILS 3
#define PIPE_CMASKS 2
#define PIPE_CACHE_N (PIPE_PROGS * PIPE_TOPOS * PIPE_BLENDS * PIPE_STENCILS * PIPE_CMASKS)
static VkPipeline s_pipe_cache[PIPE_CACHE_N * 2];   /* [single-sample | MSAA] */
static VkShaderModule s_mod_geo_v, s_mod_geo_f, s_mod_tex_v, s_mod_tex_f, s_mod_blit_v, s_mod_blit_f;

/* Untextured batch (flat/gouraud triangles, lines-as-quads, flat rects). */
typedef struct { float x, y, r, g, b, a; } Vert;   /* GEO vertex, stride 24 */
#define VK_VBUF_VERTS 65536
static VkBuffer       s_vbuf;
static VkDeviceMemory s_vbuf_mem;
static Vert          *s_vmap;             /* persistently mapped host-visible */
static uint32_t       s_vcount;           /* verts accumulated this batch */
/* Draws are SUBMITTED without waiting (deferred-submit model), so a flushed
 * batch's vertices may still be read by the GPU when the next batch begins.
 * Rewriting the buffer from offset 0 per batch raced that read and shredded
 * whole batches on screen (terrain/mesh chunks missing or drawn with another
 * batch's vertices). Batches therefore SUB-ALLOCATE: each flush draws
 * [base, base+count) and bumps base; gpu_sync (queue idle) recycles to 0. */
static uint32_t       s_vbase;            /* first vertex of the open batch */
static int            s_geo_semi = -2;    /* open untextured batch keys */
static int            s_geo_mask = 0, s_geo_check = 0;
static int            s_geo_mirror_suppress = 0;   /* open batch: skip the wide mirror
                                                    * (full-screen overlay rect draws its
                                                    * own full-width wide pass instead) */

/* Sub-pixel vertex override + perspective UV weights for the next triangle
 * ([video] geometry_correction / perspective_texturing; see gpu_render.h).
 * gpu.c sets these immediately before the matching gr_draw_*_triangle and they
 * are consumed by it. All-zero == the faithful integer/affine path. */
static int   s_pc_valid = 0;           /* sub-pixel positions present */
static float s_pc_x[3], s_pc_y[3];     /* native VRAM px, fractional  */
static int   s_pq_valid = 0;           /* perspective weights present */
static float s_pq[3];

/* Textured batch: 19 floats/vert (pos,uv,col,tpage,clut,depth,raw,limits,q).
 * q is the perspective weight ([video] perspective_texturing); 0 = affine,
 * which the vertex shader turns into w == 1.0 — the faithful default. */
#define TEXV 24     /* +4 HD source rect, +1 HD mode */
typedef struct { float v[TEXV]; } TexVert;
#define VK_TBUF_VERTS 24576               /* multiple of 3 */
static VkBuffer       s_tbuf;
static VkDeviceMemory s_tbuf_mem;
static TexVert       *s_tmap;
static uint32_t       s_tcount;
static uint32_t       s_tbase;            /* first vertex of the open batch */
static int            s_tb_semi = -2, s_tb_mask = 0, s_tb_check = 0, s_tb_filter = 0;
static int            s_tb_twin[4] = {0,0,0,0};

static void flush_geometry(void);         /* commit pending untextured batch */
static void flush_tex_batch(void);        /* commit pending textured batch */
static void flush_cpu_upload(void);       /* pending CPU writes -> GPU images */
static void pack_flush(void);             /* hr -> raw mirror (dirty rect) */
static void flush_pack_if_sampling(int tpx, int tpy, int depth, int clx, int cly);
static void vram_upload_block(int x, int y, int w, int h, const uint16_t *data);
static void gpu_copy_rect(int sx, int sy, int dx, int dy, int w, int h);
static void ensure_cpu(void);             /* GPU -> CPU mirror readback */
static int  make_staging(VkDeviceSize bytes, VkBuffer *buf, VkDeviceMemory *mem, void **map);
static int  hd_vk_init(void);
static void free_staging(VkBuffer buf, VkDeviceMemory mem);

/* draw area / offset / per-prim render state (set via the vtable setters) */
static int s_da_x1, s_da_y1, s_da_x2 = VRAM_W - 1, s_da_y2 = VRAM_H - 1;
static int s_off_x, s_off_y;
static int s_semi_en = 0, s_semi_mode = 0;
static int s_mask_set = 0, s_mask_check = 0;
static int s_tw_mask_x = 0, s_tw_mask_y = 0, s_tw_off_x = 0, s_tw_off_y = 0;
static int s_mod_r = 128, s_mod_g = 128, s_mod_b = 128, s_mod_raw = 0;

/* Coherency: regions rendered to the hr image but not yet packed to the raw
 * mirror, and whether the hr image diverges from the CPU mirror. */
typedef struct { int set, x0, y0, x1, y1; } DirtyRect;
static DirtyRect s_pack_dirty;
static int       s_gpu_dirty;
/* CPU-side VRAM writes (GP0 A0 transfers, DMA, fills, single pokes) land in
 * the CPU array immediately and accumulate here; ONE batched upload flushes
 * them before the next GPU op that could observe them (mirrors the GL
 * backend). Eager per-op uploads are the transition-stall:
 * each PIXEL cost 2 vkAllocateMemory + 2 vkQueueSubmit + a 2-draw blit pass,
 * so a single GP0 A0 texture upload wedged the main thread for minutes inside
 * the driver (freeze dump: 8/8 samples in D3DKMTCreateAllocation churn).
 *
 * EXACT rect list, NOT a single union (ported from the GL fix for the MMX6
 * black-frame flicker, ISSUES #7): a union of DISJOINT uploads covers the
 * framebuffers in between, and flushing paints that whole box from the CPU
 * VRAM array — STALE under VK once the hr image is authoritative (ensure_cpu
 * syncs on demand only) — stomping freshly rendered frames. Only pixels the
 * CPU actually wrote may ever be uploaded. See up_add()/up_add_transfer(). */
/* 64 (vs GL's 16): VK pays TWO queue submits per flush (copies + blit pass),
 * so overflow-triggered flushes must stay rare. MDEC/FMV macroblock streams
 * coalesce into one rect per block row via the row-band merge, so a whole
 * streamed frame fits comfortably. */
#define UP_RECTS_MAX 64
static DirtyRect s_up_rects[UP_RECTS_MAX];
static int       s_up_nrects = 0;

static void rect_clear(DirtyRect *r) { r->set = 0; }
static void rect_add(DirtyRect *r, int x0, int y0, int x1, int y1) {
    if (x0 > x1 || y0 > y1) return;
    if (!r->set) { r->x0=x0; r->y0=y0; r->x1=x1; r->y1=y1; r->set=1; return; }
    if (x0 < r->x0) r->x0 = x0; if (y0 < r->y0) r->y0 = y0;
    if (x1 > r->x1) r->x1 = x1; if (y1 > r->y1) r->y1 = y1;
}
static int rect_intersects(const DirtyRect *r, int x0, int y0, int x1, int y1) {
    if (!r->set) return 0;
    return !(x1 < r->x0 || x0 > r->x1 || y1 < r->y0 || y0 > r->y1);
}

/* Add an uploaded rect (see s_up_rects). Merges ONLY when the merge adds no
 * uncovered pixels: containment either way, or same row-band / column-band
 * extension. On overflow: flush the pending set (order-preserving) when the
 * device is up; pre-ready the CPU array is fully authoritative, so a union is
 * harmless there. Mirrors the GL backend's up_add (20k-randomized-rect
 * host-test-proven merge rule). */
static void up_add(int x0, int y0, int x1, int y1) {
    hd_memo_invalidate(x0, y0, x1, y1);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 > VRAM_W - 1) x1 = VRAM_W - 1;
    if (y1 > VRAM_H - 1) y1 = VRAM_H - 1;
    if (x0 > x1 || y0 > y1) return;
    for (int i = 0; i < s_up_nrects; i++) {
        DirtyRect *r = &s_up_rects[i];
        if (x0 >= r->x0 && x1 <= r->x1 && y0 >= r->y0 && y1 <= r->y1)
            return;                                   /* contained */
        if (x0 <= r->x0 && x1 >= r->x1 && y0 <= r->y0 && y1 >= r->y1) {
            r->x0 = x0; r->y0 = y0; r->x1 = x1; r->y1 = y1;  /* contains */
            return;
        }
        if (y0 == r->y0 && y1 == r->y1 &&
            x0 <= r->x1 + 1 && x1 >= r->x0 - 1) {     /* same row-band extension */
            if (x0 < r->x0) r->x0 = x0;
            if (x1 > r->x1) r->x1 = x1;
            return;
        }
        if (x0 == r->x0 && x1 == r->x1 &&
            y0 <= r->y1 + 1 && y1 >= r->y0 - 1) {     /* same column-band extension */
            if (y0 < r->y0) r->y0 = y0;
            if (y1 > r->y1) r->y1 = y1;
            return;
        }
    }
    if (s_up_nrects >= UP_RECTS_MAX) {
        if (s_ready) {
            flush_cpu_upload();       /* order-preserving: land the old ones */
        } else {
            DirtyRect *r = &s_up_rects[0];
            for (int i = 1; i < s_up_nrects; i++) {
                if (s_up_rects[i].x0 < r->x0) r->x0 = s_up_rects[i].x0;
                if (s_up_rects[i].y0 < r->y0) r->y0 = s_up_rects[i].y0;
                if (s_up_rects[i].x1 > r->x1) r->x1 = s_up_rects[i].x1;
                if (s_up_rects[i].y1 > r->y1) r->y1 = s_up_rects[i].y1;
            }
            if (x0 < r->x0) r->x0 = x0;
            if (y0 < r->y0) r->y0 = y0;
            if (x1 > r->x1) r->x1 = x1;
            if (y1 > r->y1) r->y1 = y1;
            s_up_nrects = 1;
            return;
        }
    }
    DirtyRect *r = &s_up_rects[s_up_nrects++];
    r->x0 = x0; r->y0 = y0; r->x1 = x1; r->y1 = y1; r->set = 1;
}

/* GP0(A0) transfer: per-pixel wrap means a wrapping transfer touches up to
 * FOUR exact rects — never all of VRAM (the "wrapped: take all" union was the
 * same stale-stomp class). Mirrors the GL backend's up_add_transfer. */
static void up_add_transfer(int x, int y, int w, int h) {
    x &= VRAM_W - 1; y &= VRAM_H - 1;
    if (w > VRAM_W) w = VRAM_W;
    if (h > VRAM_H) h = VRAM_H;
    if (w <= 0 || h <= 0) return;
    int w1 = w, w2 = 0, h1 = h, h2 = 0;
    if (x + w > VRAM_W) { w1 = VRAM_W - x; w2 = w - w1; }
    if (y + h > VRAM_H) { h1 = VRAM_H - y; h2 = h - h1; }
    up_add(x, y, x + w1 - 1, y + h1 - 1);
    if (w2)       up_add(0, y, w2 - 1, y + h1 - 1);
    if (h2)       up_add(x, 0, x + w1 - 1, h2 - 1);
    if (w2 && h2) up_add(0, 0, w2 - 1, h2 - 1);
}

/* ---- helpers ----------------------------------------------------------- */
static void vk_log(const char *msg) { fprintf(stdout, "psxrecomp: vulkan: %s\n", msg); }

static int vk_die(const char *why) { fprintf(stdout, "psxrecomp: vulkan: %s\n", why); return 0; }

static uint32_t find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < s_memprops.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (s_memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

static void vk_gpu_sync_internal(void);   /* drain queue + reclaim work pool/staging */

/* Coalesce work operations into one command buffer per sync interval. This
 * preserves recording order while removing a queue submission per small GPU
 * operation. */
static VkCommandBuffer s_work_cb;

/* Pass coalescing: consecutive batch flushes share ONE canonical render pass
 * (left open in the work CB) and queue their native-wide mirror draws, which
 * replay in one wide pass when the canonical pass closes. Every other GPU
 * operation goes through begin_oneshot(), which closes the pass first, so
 * recording order is unchanged. A render pass per flush cost a Metal encoder
 * plus a full attachment load/store each on tile-based GPUs (iOS). */
static void pass_close(void);

/* The shared work CB without closing the coalesced pass (batch flushes). */
static VkCommandBuffer work_cb(void) {
    s_perf_cur.oneshots++;
    if (s_work_cb) return s_work_cb;
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = s_work_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    if (p_vkAllocateCommandBuffers(s_dev, &ai, &cb) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    p_vkBeginCommandBuffer(cb, &bi);
    s_work_cb = cb;
    return cb;
}

static VkCommandBuffer begin_oneshot(void) {
    pass_close();
    return work_cb();
}

/* Finish a logical operation but keep the shared command buffer open. */
static void end_oneshot(VkCommandBuffer cb) {
    (void)cb;
    s_work_pending = 1;
}

static void flush_work(void) {
    if (!s_work_cb) return;
    pass_close();
    p_vkEndCommandBuffer(s_work_cb);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s_work_cb;
    p_vkQueueSubmit(s_queue, 1, &si, VK_NULL_HANDLE);
    s_perf_cur.submits++;
    s_work_cb = VK_NULL_HANDLE;
}

/* Defer a staging buffer's destruction until the queue is next idle (its
 * copy may still be in flight). */
static void defer_staging(VkBuffer buf, VkDeviceMemory mem) {
    if (s_pending_n >= PENDING_STAGING_MAX)
        vk_gpu_sync_internal();   /* ring full mid-frame: drain so we never leak */
    s_pending_buf[s_pending_n] = buf; s_pending_mem[s_pending_n] = mem; s_pending_n++;
}

/* Drain all submitted work, then reclaim the work-CB pool and deferred staging.
 * The single sync point for the deferred-submit model. */
static void vk_gpu_sync_internal(void) {
    s_perf_cur.syncs++;
    flush_work();
    if (s_work_pending) {
        uint64_t wait_start = SDL_GetPerformanceCounter();
        p_vkQueueWaitIdle(s_queue);
        s_perf_cur.wait_us += perf_elapsed_us(wait_start);
        /* RELEASE_RESOURCES so the per-op command buffers allocated since the
         * last sync are actually freed (a plain reset only recycles them, and
         * begin_oneshot always allocates fresh -> unbounded growth). */
        p_vkResetCommandPool(s_dev, s_work_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        s_work_pending = 0;
    }
    for (int i = 0; i < s_pending_n; i++) {
        staging_release(s_pending_buf[i], s_pending_mem[i]);
    }
    s_pending_n = 0;
    if (s_hd_retired_n) hd_vk_reap();
    s_ds_blit_idx = 0;
    /* Queue idle: every submitted draw has consumed its vertex-buffer region,
     * so recycle the sub-allocation cursors. An OPEN (unflushed) batch's
     * vertices move to offset 0 so its eventual draw stays contiguous. */
    if (s_vmap) {
        if (s_vcount && s_vbase)
            memmove(s_vmap, s_vmap + s_vbase, (size_t)s_vcount * sizeof(Vert));
        s_vbase = 0;
    }
    if (s_tmap) {
        if (s_tcount && s_tbase)
            memmove(s_tmap, s_tmap + s_tbase, (size_t)s_tcount * sizeof(TexVert));
        s_tbase = 0;
    }
}
#define gpu_sync() vk_gpu_sync_internal()

/* Image layout transition (full subresource, color aspect). */
static void img_barrier(VkCommandBuffer cb, VkImage img,
                        VkImageLayout from, VkImageLayout to,
                        VkAccessFlags src_access, VkAccessFlags dst_access,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = src_access; b.dstAccessMask = dst_access;
    p_vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

/* Clear a swapchain image (TRANSFER_DST_OPTIMAL) to opaque black, then order
 * that clear before the next transfer write to the same image. Validation
 * reported SYNC-HAZARD-WRITE-AFTER-WRITE for clear -> blit without a barrier;
 * on RDNA3 the clear overlapped the blit and tiles of the frame (24-bit FMV
 * via vk_renderer_present_cpu) came out black. */
static void sc_clear_black(VkCommandBuffer cb, VkImage sc) {
    VkClearColorValue black = {{0, 0, 0, 1}};
    VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    p_vkCmdClearColorImage(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rng);
    img_barrier(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

/* Map a layout to its (access mask, pipeline stage) for barriers. */
static void layout_access(VkImageLayout l, VkAccessFlags *acc, VkPipelineStageFlags *stage) {
    switch (l) {
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            *acc = VK_ACCESS_TRANSFER_READ_BIT;  *stage = VK_PIPELINE_STAGE_TRANSFER_BIT; break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            *acc = VK_ACCESS_TRANSFER_WRITE_BIT; *stage = VK_PIPELINE_STAGE_TRANSFER_BIT; break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            *acc = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            *stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            /* sampled in both the fragment (textured draw) and compute (pack)
             * stages, so cover all commands rather than guess a single stage. */
            *acc = VK_ACCESS_SHADER_READ_BIT; *stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; break;
        case VK_IMAGE_LAYOUT_GENERAL:
            *acc = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            *stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; break;
        default:
            *acc = 0; *stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; break;
    }
}

/* Transition an arbitrary color-aspect image to `want` on cb, tracking layout
 * through *cur. */
static void img_to(VkCommandBuffer cb, VkImage img, VkImageLayout *cur, VkImageLayout want) {
    if (*cur == want) return;
    VkAccessFlags sa, da; VkPipelineStageFlags ss, ds;
    layout_access(*cur, &sa, &ss);
    layout_access(want, &da, &ds);
    img_barrier(cb, img, *cur, want, sa, da, ss, ds);
    *cur = want;
}

/* Transition the VRAM image to `want` on cb, tracking its current layout. */
static void vram_to(VkCommandBuffer cb, VkImageLayout want) {
    img_to(cb, s_vram_img, &s_vram_layout, want);
}

/* ---- loader bring-up --------------------------------------------------- */
static int load_global_funcs(void) {
#define LOAD(n) p_##n = (PFN_##n)p_vkGetInstanceProcAddr(NULL, #n); if (!p_##n) return vk_die("missing " #n);
    VK_GLOBAL_FUNCS(LOAD)
#undef LOAD
    return 1;
}
static int load_instance_funcs(void) {
#define LOAD(n) p_##n = (PFN_##n)p_vkGetInstanceProcAddr(s_instance, #n); if (!p_##n) return vk_die("missing " #n);
    VK_INSTANCE_FUNCS(LOAD)
#undef LOAD
    p_vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)
        p_vkGetInstanceProcAddr(s_instance, "vkGetDeviceProcAddr");
    return p_vkGetDeviceProcAddr ? 1 : vk_die("missing vkGetDeviceProcAddr");
}
static int load_device_funcs(void) {
#define LOAD(n) p_##n = (PFN_##n)p_vkGetDeviceProcAddr(s_dev, #n); if (!p_##n) return vk_die("missing " #n);
    VK_DEVICE_FUNCS(LOAD)
#undef LOAD
    return 1;
}

/* ---- instance / device / swapchain ------------------------------------- */
static int create_instance(void) {
    unsigned ext_count = 0;
#if defined(PSX_SDL3)
    const char *const *sdl_exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);
    if (!sdl_exts)
        return vk_die("SDL_Vulkan_GetInstanceExtensions failed");
    const char *exts[16];
    if (ext_count > 16) ext_count = 16;
    for (unsigned i = 0; i < ext_count; ++i) exts[i] = sdl_exts[i];
#else
    if (!SDL_Vulkan_GetInstanceExtensions(s_win, &ext_count, NULL))
        return vk_die("SDL_Vulkan_GetInstanceExtensions(count) failed");
    const char *exts[16];
    if (ext_count > 16) ext_count = 16;
    if (!SDL_Vulkan_GetInstanceExtensions(s_win, &ext_count, exts))
        return vk_die("SDL_Vulkan_GetInstanceExtensions(list) failed");
#endif

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "psxrecomp";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = ext_count;
    ci.ppEnabledExtensionNames = exts;
    if (p_vkCreateInstance(&ci, NULL, &s_instance) != VK_SUCCESS)
        return vk_die("vkCreateInstance failed");
    return 1;
}

static int pick_device(void) {
    uint32_t n = 0;
    p_vkEnumeratePhysicalDevices(s_instance, &n, NULL);
    if (!n) return vk_die("no Vulkan physical devices");
    if (n > 8) n = 8;
    VkPhysicalDevice devs[8];
    p_vkEnumeratePhysicalDevices(s_instance, &n, devs);

    /* Prefer a discrete GPU with a graphics+present queue family. */
    VkPhysicalDevice best = VK_NULL_HANDLE; uint32_t best_fam = 0; int best_score = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(devs[i], &pr);
        uint32_t qn = 0; p_vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
        if (qn > 16) qn = 16;
        VkQueueFamilyProperties qf[16];
        p_vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf);
        for (uint32_t f = 0; f < qn; f++) {
            VkBool32 present = VK_FALSE;
            p_vkGetPhysicalDeviceSurfaceSupportKHR(devs[i], f, s_surface, &present);
            if ((qf[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                int score = (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2 : 1;
                if (score > best_score) { best_score = score; best = devs[i]; best_fam = f; }
                break;
            }
        }
    }
    if (best == VK_NULL_HANDLE) return vk_die("no graphics+present queue family");
    s_phys = best; s_qfam = best_fam;
    p_vkGetPhysicalDeviceMemoryProperties(s_phys, &s_memprops);
    VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(s_phys, &pr);
    fprintf(stdout, "psxrecomp: vulkan: device = %s\n", pr.deviceName);
    return 1;
}

static int create_device(void) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    q.queueFamilyIndex = s_qfam; q.queueCount = 1; q.pQueuePriorities = &prio;
    const char *dev_exts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo ci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    ci.queueCreateInfoCount = 1; ci.pQueueCreateInfos = &q;
    ci.enabledExtensionCount = 1; ci.ppEnabledExtensionNames = dev_exts;
    if (p_vkCreateDevice(s_phys, &ci, NULL, &s_dev) != VK_SUCCESS)
        return vk_die("vkCreateDevice failed");
    if (!load_device_funcs()) return 0;
    p_vkGetDeviceQueue(s_dev, s_qfam, 0, &s_queue);
    return 1;
}

static VkPresentModeKHR choose_present_mode(void) {
    uint32_t n = 0;
    p_vkGetPhysicalDeviceSurfacePresentModesKHR(s_phys, s_surface, &n, NULL);
    if (n > 8) n = 8;
    VkPresentModeKHR modes[8];
    p_vkGetPhysicalDeviceSurfacePresentModesKHR(s_phys, s_surface, &n, modes);
    /* The frontend already owns frame cadence. Prefer MAILBOX for tear-free
     * delivery so FIFO does not introduce a second blocking pacer. */
    VkPresentModeKHR want = (s_present_mode_req == 0)
                          ? VK_PRESENT_MODE_IMMEDIATE_KHR
                          : VK_PRESENT_MODE_MAILBOX_KHR;
    for (uint32_t i = 0; i < n; i++) if (modes[i] == want) return want;
    return VK_PRESENT_MODE_FIFO_KHR;  /* always supported */
}

static int create_swapchain(void) {
    VkSurfaceCapabilitiesKHR caps;
    p_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_phys, s_surface, &caps);

    uint32_t fn = 0;
    p_vkGetPhysicalDeviceSurfaceFormatsKHR(s_phys, s_surface, &fn, NULL);
    if (fn > 16) fn = 16;
    VkSurfaceFormatKHR fmts[16];
    p_vkGetPhysicalDeviceSurfaceFormatsKHR(s_phys, s_surface, &fn, fmts);
    VkSurfaceFormatKHR chosen = fmts[0];
    for (uint32_t i = 0; i < fn; i++) {
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM &&
            fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = fmts[i]; break; }
    }
    s_sc_format = chosen.format;

    VkExtent2D ext = caps.currentExtent;
    if (ext.width == 0xFFFFFFFFu) {
        int w = 0, h = 0;
#if defined(PSX_SDL3)
        SDL_GetWindowSizeInPixels(s_win, &w, &h);
#else
        SDL_Vulkan_GetDrawableSize(s_win, &w, &h);
#endif
        ext.width = (uint32_t)w; ext.height = (uint32_t)h;
    }
    s_sc_extent = ext;

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount && img_count > caps.maxImageCount) img_count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    ci.surface = s_surface;
    ci.minImageCount = img_count;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = choose_present_mode();
    ci.clipped = VK_TRUE;
    if (p_vkCreateSwapchainKHR(s_dev, &ci, NULL, &s_swapchain) != VK_SUCCESS)
        return vk_die("vkCreateSwapchainKHR failed");

    s_sc_count = 0;
    p_vkGetSwapchainImagesKHR(s_dev, s_swapchain, &s_sc_count, NULL);
    if (s_sc_count > 8) s_sc_count = 8;
    p_vkGetSwapchainImagesKHR(s_dev, s_swapchain, &s_sc_count, s_sc_images);
    return 1;
}

static void pp_release_swapchain(void);
static void pp_destroy_all(void);

static void destroy_swapchain(void) {
    pp_release_swapchain();   /* present-pass views/framebuffers/pipeline */
    if (s_swapchain) { p_vkDestroySwapchainKHR(s_dev, s_swapchain, NULL); s_swapchain = VK_NULL_HANDLE; }
}

static int create_sync(void) {
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = s_cmd_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = VK_FRAMES;
    if (p_vkAllocateCommandBuffers(s_dev, &ai, s_cmd) != VK_SUCCESS)
        return vk_die("vkAllocateCommandBuffers failed");
    VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < VK_FRAMES; i++) {
        if (p_vkCreateSemaphore(s_dev, &si, NULL, &s_sem_acquire[i]) != VK_SUCCESS ||
            p_vkCreateSemaphore(s_dev, &si, NULL, &s_sem_render[i]) != VK_SUCCESS ||
            p_vkCreateFence(s_dev, &fi, NULL, &s_fence[i]) != VK_SUCCESS)
            return vk_die("sync object creation failed");
    }
    return 1;
}

/* The VRAM image: RGBA8, transfer src+dst (fills via clear, copies, present
 * blit) + color attachment (geometry, Phase 2). Created in UNDEFINED, cleared
 * to black, left in TRANSFER_SRC_OPTIMAL between frames. */
static int create_vram_image(void) {
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent.width = VRAM_W * s_scale;
    ci.extent.height = VRAM_H * s_scale;
    ci.extent.depth = 1;
    ci.mipLevels = 1; ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (p_vkCreateImage(s_dev, &ci, NULL, &s_vram_img) != VK_SUCCESS)
        return vk_die("vram image create failed");

    VkMemoryRequirements req; p_vkGetImageMemoryRequirements(s_dev, s_vram_img, &req);
    VkMemoryAllocateInfo mi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mi.allocationSize = req.size;
    mi.memoryTypeIndex = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mi.memoryTypeIndex == UINT32_MAX) return vk_die("no device-local memory");
    if (p_vkAllocateMemory(s_dev, &mi, NULL, &s_vram_mem) != VK_SUCCESS)
        return vk_die("vram memory alloc failed");
    p_vkBindImageMemory(s_dev, s_vram_img, s_vram_mem, 0);

    /* Clear to black and park in TRANSFER_SRC_OPTIMAL. */
    VkCommandBuffer cb = begin_oneshot();
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue black = {{0,0,0,0}};
    VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    p_vkCmdClearColorImage(cb, s_vram_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rng);
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    end_oneshot(cb);
    return 1;
}

/* ---- geometry pipeline (Phase 2) --------------------------------------- */
static VkShaderModule make_module(const uint32_t *code, uint32_t size) {
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = size; ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    p_vkCreateShaderModule(s_dev, &ci, NULL, &m);
    return m;
}

/* ---- push-constant blocks (must match the shaders' layouts) ------------- */
typedef struct { float shift, xoff, xhalf; } GeoPush;          /* 12 bytes */
typedef struct {
    float shift, xoff, xhalf;        /* 0,4,8 */
    int   semipass, maskset, filter; /* 12,16,20 */
    int   _pad[2];                   /* 24,28 -> ivec4 16-byte align */
    int   twin[4];                   /* 32..47 */
} TexPush;                           /* 48 bytes */
typedef struct {
    float shift;        /* 0 */
    int   stp_pass;     /* 4 */
    int   maskset;      /* 8 */
    int   src_div;      /* 12 */
    int   src_off[2];   /* 16,20 */
    int   _pad[2];      /* 24,28 -> ivec4 16-byte align */
    int   rect[4];      /* 32..47: x0,y0,x1,y1 native px */
} BlitPush;             /* 48 bytes */
typedef struct { int scale, off_x, off_y; } PackPush;          /* 12 bytes */

/* Create a device-local image + view (color aspect by default). */
static int make_image(VkFormat fmt, int w, int h, VkImageUsageFlags usage,
                      VkImageAspectFlags aspect,
                      VkImage *img, VkDeviceMemory *mem, VkImageView *view) {
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt;
    ci.extent.width = w; ci.extent.height = h; ci.extent.depth = 1;
    ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = (VkSampleCountFlagBits)s_make_samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (p_vkCreateImage(s_dev, &ci, NULL, img) != VK_SUCCESS) return vk_die("image create failed");
    VkMemoryRequirements req; p_vkGetImageMemoryRequirements(s_dev, *img, &req);
    VkMemoryAllocateInfo mi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mi.allocationSize = req.size;
    mi.memoryTypeIndex = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mi.memoryTypeIndex == UINT32_MAX || p_vkAllocateMemory(s_dev, &mi, NULL, mem) != VK_SUCCESS)
        return vk_die("image memory alloc failed");
    p_vkBindImageMemory(s_dev, *img, *mem, 0);
    if (view) {
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = *img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange.aspectMask = aspect;
        vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
        if (p_vkCreateImageView(s_dev, &vi, NULL, view) != VK_SUCCESS)
            return vk_die("image view failed");
    }
    return 1;
}

static VkFormat choose_ds_format(void) {
    {
        const char *e = getenv("PSX_VK_STENCIL_ONLY");
        VkFormatProperties fp; p_vkGetPhysicalDeviceFormatProperties(s_phys, VK_FORMAT_S8_UINT, &fp);
        if ((!e || *e != '0') && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            s_ds_aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
            return VK_FORMAT_S8_UINT;
        }
    }
    VkFormat cand[2] = { VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT };
    for (int i = 0; i < 2; i++) {
        VkFormatProperties fp; p_vkGetPhysicalDeviceFormatProperties(s_phys, cand[i], &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return cand[i];
    }
    return VK_FORMAT_D24_UNORM_S8_UINT;
}

static int create_buffer_mapped(VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem, void **map) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size; bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (p_vkCreateBuffer(s_dev, &bci, NULL, buf) != VK_SUCCESS) return vk_die("vertex buffer failed");
    VkMemoryRequirements req; p_vkGetBufferMemoryRequirements(s_dev, *buf, &req);
    VkMemoryAllocateInfo mi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mi.allocationSize = req.size;
    mi.memoryTypeIndex = find_mem_type(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mi.memoryTypeIndex == UINT32_MAX || p_vkAllocateMemory(s_dev, &mi, NULL, mem) != VK_SUCCESS)
        return vk_die("vertex buffer memory failed");
    p_vkBindBufferMemory(s_dev, *buf, *mem, 0);
    p_vkMapMemory(s_dev, *mem, 0, size, 0, map);
    return 1;
}

/* Build all render targets, samplers, descriptors, pipeline layouts and the
 * pack compute pipeline. Graphics pipelines are built lazily by get_pipeline. */
static int create_render_targets(void) {
    int S = s_scale;

    /* RGBA8 hr render-target view */
    VkImageViewCreateInfo vv = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vv.image = s_vram_img; vv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vv.format = VK_FORMAT_R8G8B8A8_UNORM;
    vv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vv.subresourceRange.levelCount = 1; vv.subresourceRange.layerCount = 1;
    if (p_vkCreateImageView(s_dev, &vv, NULL, &s_vram_view) != VK_SUCCESS)
        return vk_die("vram image view failed");

    /* Native raw mirror (sampled + storage), depth/stencil, scratch, upload. */
    if (!make_image(VK_FORMAT_R16_UINT, VRAM_W, VRAM_H,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &s_raw_img, &s_raw_mem, &s_raw_view)) return 0;
    s_ds_format = choose_ds_format();
    if (!make_image(s_ds_format, VRAM_W * S, VRAM_H * S,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, DS_ASPECT,
                    &s_ds_img, &s_ds_mem, &s_ds_view)) return 0;
    if (!make_image(VK_FORMAT_R8G8B8A8_UNORM, VRAM_W * S, VRAM_H * S,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &s_scratch_img, &s_scratch_mem, &s_scratch_view)) return 0;
    if (!make_image(VK_FORMAT_R8G8B8A8_UNORM, VRAM_W, VRAM_H,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &s_up_img, &s_up_mem, &s_up_view)) return 0;

    /* Clear raw mirror to 0 (park sampled) and stencil to 0 (park attachment). */
    {
        VkCommandBuffer cb = begin_oneshot();
        img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue zero = {{0,0,0,0}};
        VkImageSubresourceRange crng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        p_vkCmdClearColorImage(cb, s_raw_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &crng);
        img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkImageMemoryBarrier db = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; db.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        db.image = s_ds_img; db.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        db.subresourceRange.aspectMask = DS_ASPECT;
        db.subresourceRange.levelCount = 1; db.subresourceRange.layerCount = 1;
        p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, NULL, 0, NULL, 1, &db);
        VkClearDepthStencilValue dsv = { 0.0f, 0 };
        VkImageSubresourceRange srng = { DS_ASPECT, 0, 1, 0, 1 };
        p_vkCmdClearDepthStencilImage(cb, s_ds_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dsv, 1, &srng);
        db.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        db.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, NULL, 0, NULL, 1, &db);
        end_oneshot(cb);
    }

    /* Render pass: color (hr, load/store) + depth-stencil (stencil load/store). */
    VkAttachmentDescription att[2] = {0};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att[1].format = s_ds_format; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD; att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference dref = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &cref;
    sub.pDepthStencilAttachment = &dref;
    VkRenderPassCreateInfo rpi = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpi.attachmentCount = 2; rpi.pAttachments = att;
    rpi.subpassCount = 1; rpi.pSubpasses = &sub;
    if (p_vkCreateRenderPass(s_dev, &rpi, NULL, &s_rpass) != VK_SUCCESS)
        return vk_die("render pass failed");
    {
        int want = touch_ui_msaa();
        VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(s_phys, &pr);
        const VkSampleCountFlags okc = pr.limits.framebufferColorSampleCounts &
                                       pr.limits.framebufferStencilSampleCounts;
        while (want > 1 && !(okc & (VkSampleCountFlags)want)) want >>= 1;
        s_msaa = want < 1 ? 1 : want;
    }
    if (s_msaa > 1) {
        VkAttachmentDescription ma[3];
        ma[0] = att[0]; ma[0].samples = (VkSampleCountFlagBits)s_msaa;
        ma[1] = att[1]; ma[1].samples = (VkSampleCountFlagBits)s_msaa;
        ma[2] = att[0]; ma[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        VkAttachmentReference rref = { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkSubpassDescription msub = sub;
        msub.pResolveAttachments = &rref;
        VkRenderPassCreateInfo mrpi = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        mrpi.attachmentCount = 3; mrpi.pAttachments = ma;
        mrpi.subpassCount = 1; mrpi.pSubpasses = &msub;
        if (p_vkCreateRenderPass(s_dev, &mrpi, NULL, &s_rpass_ms) != VK_SUCCESS) s_msaa = 1;
    }
    fprintf(stdout, "psxrecomp: vulkan: anti-aliasing %s (%d sample%s)\n",
            s_msaa > 1 ? "MSAA on the widescreen picture" : "off", s_msaa, s_msaa > 1 ? "s" : "");

    VkImageView fbviews[2] = { s_vram_view, s_ds_view };
    VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fi.renderPass = s_rpass; fi.attachmentCount = 2; fi.pAttachments = fbviews;
    fi.width = VRAM_W * S; fi.height = VRAM_H * S; fi.layers = 1;
    if (p_vkCreateFramebuffer(s_dev, &fi, NULL, &s_fbo) != VK_SUCCESS)
        return vk_die("framebuffer failed");

    /* Nearest sampler (texelFetch ignores filtering/addressing, but a combined
     * image sampler still needs one). */
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (p_vkCreateSampler(s_dev, &sci, NULL, &s_samp) != VK_SUCCESS) return vk_die("sampler failed");

    /* Descriptor set layouts. */
    {
        VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                           VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
        VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        ci.bindingCount = 1; ci.pBindings = &b;
        if (p_vkCreateDescriptorSetLayout(s_dev, &ci, NULL, &s_dsl_tex) != VK_SUCCESS) return vk_die("dsl tex");
        if (p_vkCreateDescriptorSetLayout(s_dev, &ci, NULL, &s_dsl_blit) != VK_SUCCESS) return vk_die("dsl blit");
        VkDescriptorSetLayoutBinding pb[2] = {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
        ci.bindingCount = 2; ci.pBindings = pb;
        if (p_vkCreateDescriptorSetLayout(s_dev, &ci, NULL, &s_dsl_pack) != VK_SUCCESS) return vk_die("dsl pack");
    }

    /* Descriptor pool + sets: ds_tex (1) + ds_pack (1) + blit ring (BLIT_DESC_RING). */
    {
        VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, BLIT_DESC_RING + 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 } };
        VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pci.maxSets = BLIT_DESC_RING + 2; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        if (p_vkCreateDescriptorPool(s_dev, &pci, NULL, &s_dpool) != VK_SUCCESS) return vk_die("desc pool");
        VkDescriptorSetLayout layouts[2] = { s_dsl_tex, s_dsl_pack };
        VkDescriptorSet *sets[2] = { &s_ds_tex, &s_ds_pack };
        for (int i = 0; i < 2; i++) {
            VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool = s_dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &layouts[i];
            if (p_vkAllocateDescriptorSets(s_dev, &ai, sets[i]) != VK_SUCCESS) return vk_die("desc set alloc");
        }
        for (int i = 0; i < BLIT_DESC_RING; i++) {
            VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            ai.descriptorPool = s_dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &s_dsl_blit;
            if (p_vkAllocateDescriptorSets(s_dev, &ai, &s_ds_blit_ring[i]) != VK_SUCCESS) return vk_die("blit set alloc");
        }
        /* ds_tex: raw mirror sampled. ds_pack: hr sampled + raw storage. */
        VkDescriptorImageInfo raw_smp = { s_samp, s_raw_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo hr_smp  = { s_samp, s_vram_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo raw_st  = { VK_NULL_HANDLE, s_raw_view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[3] = {0};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = s_ds_tex; w[0].dstBinding = 0;
        w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &raw_smp;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = s_ds_pack; w[1].dstBinding = 0;
        w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &hr_smp;
        w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = s_ds_pack; w[2].dstBinding = 1;
        w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2].pImageInfo = &raw_st;
        p_vkUpdateDescriptorSets(s_dev, 3, w, 0, NULL);
    }

    /* Pipeline layouts. */
    {
        VkPushConstantRange pr;
        VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; pr.offset = 0; pr.size = sizeof(GeoPush);
        li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
        if (p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_pl_geo) != VK_SUCCESS) return vk_die("pl geo");
        pr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; pr.size = sizeof(TexPush);
        if (!hd_vk_init()) return vk_die("hd texture setup");
        VkDescriptorSetLayout texl[2] = { s_dsl_tex, s_dsl_hd };
        li.setLayoutCount = 2; li.pSetLayouts = texl;
        if (p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_pl_tex) != VK_SUCCESS) return vk_die("pl tex");
        li.setLayoutCount = 1;
        pr.size = sizeof(BlitPush); li.pSetLayouts = &s_dsl_blit;
        if (p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_pl_blit) != VK_SUCCESS) return vk_die("pl blit");
        pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pr.size = sizeof(PackPush); li.pSetLayouts = &s_dsl_pack;
        if (p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_pl_pack) != VK_SUCCESS) return vk_die("pl pack");
    }

    /* Shader modules (kept for lazy graphics-pipeline creation). */
    s_mod_geo_v  = make_module(spv_geo_vert, spv_geo_vert_size);
    s_mod_geo_f  = make_module(spv_geo_frag, spv_geo_frag_size);
    s_mod_tex_v  = make_module(spv_geo_tex_vert, spv_geo_tex_vert_size);
    s_mod_tex_f  = make_module(spv_geo_tex_frag, spv_geo_tex_frag_size);
    s_mod_blit_v = make_module(spv_blit_vert, spv_blit_vert_size);
    s_mod_blit_f = make_module(spv_blit_frag, spv_blit_frag_size);
    if (!s_mod_geo_v || !s_mod_geo_f || !s_mod_tex_v || !s_mod_tex_f || !s_mod_blit_v || !s_mod_blit_f)
        return vk_die("shader module failed");

    /* Pack compute pipeline. */
    {
        VkShaderModule cm = make_module(spv_pack_comp, spv_pack_comp_size);
        if (!cm) return vk_die("pack module failed");
        VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = cm; ci.stage.pName = "main";
        ci.layout = s_pl_pack;
        VkResult r = p_vkCreateComputePipelines(s_dev, VK_NULL_HANDLE, 1, &ci, NULL, &s_pipe_pack);
        p_vkDestroyShaderModule(s_dev, cm, NULL);
        if (r != VK_SUCCESS) return vk_die("pack pipeline failed");
    }

    /* Vertex buffers (untextured + textured). */
    if (!create_buffer_mapped((VkDeviceSize)VK_VBUF_VERTS * sizeof(Vert),
                              &s_vbuf, &s_vbuf_mem, (void**)&s_vmap)) return 0;
    if (!create_buffer_mapped((VkDeviceSize)VK_TBUF_VERTS * sizeof(TexVert),
                              &s_tbuf, &s_tbuf_mem, (void**)&s_tmap)) return 0;
    gpu_sync();   /* complete the init clears (VRAM/raw/stencil) before any frame */
    s_ready = 1;
    /* CPU VRAM writes made before the context existed (facade routed to us
     * since gr_init, s_ready was 0) accumulated only in the CPU array — seed
     * a full-frame upload so the GPU images start coherent with it. */
    up_add(0, 0, VRAM_W - 1, VRAM_H - 1);
    flush_cpu_upload();
    return 1;
}

/* Lazily build + cache a graphics pipeline for (program, topology, blend,
 * stencil). prog 0 GEO / 1 TEX / 2 BLIT; topo 0 tri / 1 line; blend 0 off else
 * PS1 mode+1; stencil 0 SET / 1 CHECK-keep / 2 CHECK-invert. */
static VkPipeline get_pipeline(int prog, int topo, int blend, int stencil, int cmask) {
    int key = (((prog * PIPE_TOPOS + topo) * PIPE_BLENDS + blend) * PIPE_STENCILS + stencil)
              * PIPE_CMASKS + cmask;
    const int msp = s_pipe_ms && s_msaa > 1;
    if (msp) key += PIPE_CACHE_N;
    if (s_pipe_cache[key]) return s_pipe_cache[key];

    VkShaderModule vs, fs; VkPipelineLayout layout;
    VkVertexInputBindingDescription bind = {0};
    VkVertexInputAttributeDescription attrs[11]; uint32_t nattr;
    if (prog == 0) {            /* GEO */
        vs = s_mod_geo_v; fs = s_mod_geo_f; layout = s_pl_geo;
        bind = (VkVertexInputBindingDescription){ 0, sizeof(Vert), VK_VERTEX_INPUT_RATE_VERTEX };
        attrs[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
        attrs[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8 };
        nattr = 2;
    } else if (prog == 1) {     /* TEX */
        vs = s_mod_tex_v; fs = s_mod_tex_f; layout = s_pl_tex;
        bind = (VkVertexInputBindingDescription){ 0, sizeof(TexVert), VK_VERTEX_INPUT_RATE_VERTEX };
        attrs[0] = (VkVertexInputAttributeDescription){ 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
        attrs[1] = (VkVertexInputAttributeDescription){ 1, 0, VK_FORMAT_R32G32_SFLOAT, 8 };
        attrs[2] = (VkVertexInputAttributeDescription){ 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
        attrs[3] = (VkVertexInputAttributeDescription){ 3, 0, VK_FORMAT_R32G32_SFLOAT, 32 };
        attrs[4] = (VkVertexInputAttributeDescription){ 4, 0, VK_FORMAT_R32G32_SFLOAT, 40 };
        attrs[5] = (VkVertexInputAttributeDescription){ 5, 0, VK_FORMAT_R32_SFLOAT, 48 };
        attrs[6] = (VkVertexInputAttributeDescription){ 6, 0, VK_FORMAT_R32_SFLOAT, 52 };
        attrs[7] = (VkVertexInputAttributeDescription){ 7, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 56 };
        attrs[8] = (VkVertexInputAttributeDescription){ 8, 0, VK_FORMAT_R32_SFLOAT, 72 };  /* a_q */
        attrs[9] = (VkVertexInputAttributeDescription){ 9, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 76 };  /* a_hd_source */
        attrs[10] = (VkVertexInputAttributeDescription){ 10, 0, VK_FORMAT_R32_SFLOAT, 92 };         /* a_hd_mode */
        nattr = 11;
    } else {                    /* BLIT (vertex-less: rect from push constant) */
        vs = s_mod_blit_v; fs = s_mod_blit_f; layout = s_pl_blit;
        nattr = 0;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vin = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vin.vertexBindingDescriptionCount = nattr ? 1 : 0; vin.pVertexBindingDescriptions = nattr ? &bind : NULL;
    vin.vertexAttributeDescriptionCount = nattr; vin.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = topo == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = msp ? (VkSampleCountFlagBits)s_msaa : VK_SAMPLE_COUNT_1_BIT;

    /* Blend (PS1 modes). Alpha always replaced so the colour-alpha mask mirror
     * is exact (src ONE / dst ZERO). */
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = cmask ? 0 : 0xF;
    float bc[4] = {1,1,1,1};
    if (blend == 0) {
        cba.blendEnable = VK_FALSE;
    } else {
        cba.blendEnable = VK_TRUE;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        switch (blend - 1) {
        case 0: /* B/2 + F/2 */
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
            cba.colorBlendOp = VK_BLEND_OP_ADD; bc[0]=bc[1]=bc[2]=bc[3]=0.5f; break;
        case 1: /* B + F */
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; cba.colorBlendOp = VK_BLEND_OP_ADD; break;
        case 2: /* B - F */
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT; break;
        case 3: /* B + F/4 */
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
            cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            cba.colorBlendOp = VK_BLEND_OP_ADD; bc[0]=bc[1]=bc[2]=bc[3]=0.25f; break;
        }
    }
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    cb.blendConstants[0]=bc[0]; cb.blendConstants[1]=bc[1]; cb.blendConstants[2]=bc[2]; cb.blendConstants[3]=bc[3];

    /* Stencil = bit15 mirror (bit0). SET: always+replace, dyn ref. CHECK: test
     * stored==0; write 1 via INVERT (stencil 2) or leave (stencil 1). */
    VkStencilOpState st = {0};
    st.compareMask = 0x01; st.writeMask = 0x01; st.reference = 0;
    if (stencil == 0)      { st.compareOp = VK_COMPARE_OP_ALWAYS; st.failOp = VK_STENCIL_OP_KEEP; st.passOp = VK_STENCIL_OP_REPLACE; st.depthFailOp = VK_STENCIL_OP_KEEP; }
    else if (stencil == 1) { st.compareOp = VK_COMPARE_OP_EQUAL;  st.failOp = VK_STENCIL_OP_KEEP; st.passOp = VK_STENCIL_OP_KEEP;    st.depthFailOp = VK_STENCIL_OP_KEEP; }
    else                   { st.compareOp = VK_COMPARE_OP_EQUAL;  st.failOp = VK_STENCIL_OP_KEEP; st.passOp = VK_STENCIL_OP_INVERT;  st.depthFailOp = VK_STENCIL_OP_KEEP; }
    VkPipelineDepthStencilStateCreateInfo dss = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    dss.depthTestEnable = VK_FALSE; dss.depthWriteEnable = VK_FALSE;
    dss.stencilTestEnable = VK_TRUE; dss.front = st; dss.back = st;

    VkDynamicState dyn[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 3; dy.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo pi = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vin; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss; pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy; pi.layout = layout; pi.renderPass = msp ? s_rpass_ms : s_rpass;
    VkPipeline p = VK_NULL_HANDLE;
    if (p_vkCreateGraphicsPipelines(s_dev, VK_NULL_HANDLE, 1, &pi, NULL, &p) != VK_SUCCESS) {
        vk_die("graphics pipeline create failed"); return VK_NULL_HANDLE;
    }
    s_pipe_cache[key] = p;
    return p;
}

/* PSX 1555 -> RGBA8 (bit15 mask -> alpha). */
static inline void rgb555_to_rgba8(uint16_t p, uint8_t out[4]) {
    out[0] = (uint8_t)(((p)        & 0x1F) << 3);
    out[1] = (uint8_t)(((p >> 5)   & 0x1F) << 3);
    out[2] = (uint8_t)(((p >> 10)  & 0x1F) << 3);
    out[3] = (uint8_t)((p & 0x8000) ? 0xFF : 0x00);
}
static inline uint16_t rgba8_to_rgb555(const uint8_t in[4]) {
    return (uint16_t)(((in[0] >> 3) & 0x1F) |
                      (((in[1] >> 3) & 0x1F) << 5) |
                      (((in[2] >> 3) & 0x1F) << 10) |
                      (in[3] >= 0x80 ? 0x8000 : 0));
}

int vk_renderer_init_context(SDL_Window *win) {
    s_win = win;
#if defined(PSX_SDL3)
    if (!SDL_Vulkan_LoadLibrary(NULL))
#else
    if (SDL_Vulkan_LoadLibrary(NULL) != 0)
#endif
        return vk_die("SDL_Vulkan_LoadLibrary failed (no Vulkan ICD?)");
    p_vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
    if (!p_vkGetInstanceProcAddr) return vk_die("no vkGetInstanceProcAddr from SDL");

    if (!load_global_funcs()) return 0;
    if (!create_instance()) return 0;
    if (!load_instance_funcs()) return 0;
#if defined(PSX_SDL3)
    if (!SDL_Vulkan_CreateSurface(s_win, s_instance, NULL, &s_surface))
#else
    if (!SDL_Vulkan_CreateSurface(s_win, s_instance, &s_surface))
#endif
        return vk_die("SDL_Vulkan_CreateSurface failed");
    if (!pick_device()) return 0;
    if (!create_device()) return 0;

    VkCommandPoolCreateInfo pi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = s_qfam;
    if (p_vkCreateCommandPool(s_dev, &pi, NULL, &s_cmd_pool) != VK_SUCCESS)
        return vk_die("vkCreateCommandPool failed");
    /* Work pool for one-shot ops, bulk-reset at gpu_sync (TRANSIENT: short-lived
     * CBs). */
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (p_vkCreateCommandPool(s_dev, &pi, NULL, &s_work_pool) != VK_SUCCESS)
        return vk_die("vkCreateCommandPool(work) failed");

    if (!create_swapchain()) return 0;
    if (!create_sync()) return 0;
    if (!create_vram_image()) return 0;
    if (!create_render_targets()) return 0;

    s_ctx_ok = 1;
    vk_log("instance/device/swapchain/VRAM + raw mirror/pipelines ready");
    return 1;
}

static void cpres_cache_free(void);   /* fwd: CPU-present resource cache */
static void osd_staging_free(void);
void vk_renderer_shutdown(void) {
    if (!s_dev) return;
    p_vkDeviceWaitIdle(s_dev);
    pp_destroy_all();
    vk_gpu_sync_internal();   /* reclaim deferred staging before tearing down */
    cpres_cache_free();       /* FMV CPU-present cached image + staging */
    osd_staging_free();       /* host toast OSD staging */
    for (int i = 0; i < STAGING_CACHE_MAX; ++i) {
        if (s_staging_cache[i].buf)
            staging_destroy(s_staging_cache[i].buf, s_staging_cache[i].mem);
    }
    wide_free_all();          /* native-wide surfaces (color + DS + framebuffers) */
    for (int i = 0; i < PIPE_CACHE_N * 2; i++)
        if (s_pipe_cache[i]) p_vkDestroyPipeline(s_dev, s_pipe_cache[i], NULL);
    if (s_pipe_pack)   p_vkDestroyPipeline(s_dev, s_pipe_pack, NULL);
    if (s_mod_geo_v)   p_vkDestroyShaderModule(s_dev, s_mod_geo_v, NULL);
    if (s_mod_geo_f)   p_vkDestroyShaderModule(s_dev, s_mod_geo_f, NULL);
    if (s_mod_tex_v)   p_vkDestroyShaderModule(s_dev, s_mod_tex_v, NULL);
    if (s_mod_tex_f)   p_vkDestroyShaderModule(s_dev, s_mod_tex_f, NULL);
    if (s_mod_blit_v)  p_vkDestroyShaderModule(s_dev, s_mod_blit_v, NULL);
    if (s_mod_blit_f)  p_vkDestroyShaderModule(s_dev, s_mod_blit_f, NULL);
    if (s_pl_geo)  p_vkDestroyPipelineLayout(s_dev, s_pl_geo, NULL);
    if (s_pl_tex)  p_vkDestroyPipelineLayout(s_dev, s_pl_tex, NULL);
    if (s_pl_blit) p_vkDestroyPipelineLayout(s_dev, s_pl_blit, NULL);
    if (s_pl_pack) p_vkDestroyPipelineLayout(s_dev, s_pl_pack, NULL);
    hd_vk_shutdown();
    if (s_dpool)    p_vkDestroyDescriptorPool(s_dev, s_dpool, NULL);
    if (s_dsl_tex)  p_vkDestroyDescriptorSetLayout(s_dev, s_dsl_tex, NULL);
    if (s_dsl_pack) p_vkDestroyDescriptorSetLayout(s_dev, s_dsl_pack, NULL);
    if (s_dsl_blit) p_vkDestroyDescriptorSetLayout(s_dev, s_dsl_blit, NULL);
    if (s_samp)     p_vkDestroySampler(s_dev, s_samp, NULL);
    if (s_fbo)         p_vkDestroyFramebuffer(s_dev, s_fbo, NULL);
    if (s_rpass)       p_vkDestroyRenderPass(s_dev, s_rpass, NULL);
    if (s_rpass_ms)    { p_vkDestroyRenderPass(s_dev, s_rpass_ms, NULL); s_rpass_ms = VK_NULL_HANDLE; }
    if (s_vram_view)   p_vkDestroyImageView(s_dev, s_vram_view, NULL);
    if (s_raw_view)    p_vkDestroyImageView(s_dev, s_raw_view, NULL);
    if (s_ds_view)     p_vkDestroyImageView(s_dev, s_ds_view, NULL);
    if (s_scratch_view)p_vkDestroyImageView(s_dev, s_scratch_view, NULL);
    if (s_up_view)     p_vkDestroyImageView(s_dev, s_up_view, NULL);
    if (s_vbuf)        p_vkDestroyBuffer(s_dev, s_vbuf, NULL);
    if (s_vbuf_mem)    p_vkFreeMemory(s_dev, s_vbuf_mem, NULL);
    if (s_tbuf)        p_vkDestroyBuffer(s_dev, s_tbuf, NULL);
    if (s_tbuf_mem)    p_vkFreeMemory(s_dev, s_tbuf_mem, NULL);
    if (s_vram_img) { p_vkDestroyImage(s_dev, s_vram_img, NULL); s_vram_img = VK_NULL_HANDLE; }
    if (s_vram_mem) { p_vkFreeMemory(s_dev, s_vram_mem, NULL); s_vram_mem = VK_NULL_HANDLE; }
    if (s_raw_img)     p_vkDestroyImage(s_dev, s_raw_img, NULL);
    if (s_raw_mem)     p_vkFreeMemory(s_dev, s_raw_mem, NULL);
    if (s_ds_img)      p_vkDestroyImage(s_dev, s_ds_img, NULL);
    if (s_ds_mem)      p_vkFreeMemory(s_dev, s_ds_mem, NULL);
    if (s_scratch_img) p_vkDestroyImage(s_dev, s_scratch_img, NULL);
    if (s_scratch_mem) p_vkFreeMemory(s_dev, s_scratch_mem, NULL);
    if (s_up_img)      p_vkDestroyImage(s_dev, s_up_img, NULL);
    if (s_up_mem)      p_vkFreeMemory(s_dev, s_up_mem, NULL);
    for (int i = 0; i < VK_FRAMES; i++) {
        if (s_sem_acquire[i]) p_vkDestroySemaphore(s_dev, s_sem_acquire[i], NULL);
        if (s_sem_render[i])  p_vkDestroySemaphore(s_dev, s_sem_render[i], NULL);
        if (s_fence[i])       p_vkDestroyFence(s_dev, s_fence[i], NULL);
    }
    destroy_swapchain();
    if (s_work_pool) p_vkDestroyCommandPool(s_dev, s_work_pool, NULL);
    if (s_cmd_pool) p_vkDestroyCommandPool(s_dev, s_cmd_pool, NULL);
    p_vkDestroyDevice(s_dev, NULL); s_dev = VK_NULL_HANDLE;
    if (s_surface) p_vkDestroySurfaceKHR(s_instance, s_surface, NULL);
    if (s_instance) p_vkDestroyInstance(s_instance, NULL);
    s_ctx_ok = 0;
}

void vk_renderer_set_present_mode(int mode) { s_present_mode_req = mode; }

/* ---- present ----------------------------------------------------------- */
/* Acquire a swapchain image, run `record` (which leaves it in
 * PRESENT_SRC_KHR-ready state), submit, present. record may be NULL for a
 * blank (the image is cleared to black). Returns 1 on success. */
static int acquire_present(VkImage *out_sc, VkCommandBuffer *out_cb,
                           uint32_t *out_idx, uint32_t *out_frame) {
#ifdef PSX_IOS
    psx_ios_wait_foreground();
#endif
    uint32_t fr = s_frame_idx % VK_FRAMES;
    p_vkWaitForFences(s_dev, 1, &s_fence[fr], VK_TRUE, UINT64_MAX);
    uint32_t img_idx = 0;
    uint64_t acquire_start = SDL_GetPerformanceCounter();
    VkResult r = p_vkAcquireNextImageKHR(s_dev, s_swapchain, UINT64_MAX,
                                         s_sem_acquire[fr], VK_NULL_HANDLE, &img_idx);
    s_perf_cur.acquire_us += perf_elapsed_us(acquire_start);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        p_vkDeviceWaitIdle(s_dev);
        destroy_swapchain();
        if (!create_swapchain()) return 0;
        return 0;  /* skip this frame; next one uses the new swapchain */
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return 0;
    p_vkResetFences(s_dev, 1, &s_fence[fr]);
    p_vkResetCommandBuffer(s_cmd[fr], 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    p_vkBeginCommandBuffer(s_cmd[fr], &bi);
    *out_sc = s_sc_images[img_idx];
    *out_cb = s_cmd[fr];
    *out_idx = img_idx;
    *out_frame = fr;
    return 1;
}

/* Copy host toast into the swapchain (must still be TRANSFER_DST). Opaque
 * overwrite — host_osd bakes an opaque panel so no blend pass is needed. */
static VkBuffer       s_osd_buf  = VK_NULL_HANDLE;
static VkDeviceMemory s_osd_mem  = VK_NULL_HANDLE;
static void          *s_osd_map  = NULL;
static VkDeviceSize   s_osd_cap  = 0;

static void osd_staging_free(void) {
    if (s_osd_buf) {
        staging_destroy(s_osd_buf, s_osd_mem);
        s_osd_buf = VK_NULL_HANDLE;
        s_osd_mem = VK_NULL_HANDLE;
    }
    s_osd_map = NULL;
    s_osd_cap = 0;
}

static void vk_osd_copy_rect(VkCommandBuffer cb, VkImage sc,
                             const uint32_t *px, int w, int h,
                             int x, int y, VkDeviceSize buf_off) {
    if (!px || w <= 0 || h <= 0) return;
    int dw = w, dh = h;
    if (x + dw > (int)s_sc_extent.width) dw = (int)s_sc_extent.width - x;
    if (y + dh > (int)s_sc_extent.height) dh = (int)s_sc_extent.height - y;
    if (dw < 1 || dh < 1) return;
    memcpy((uint8_t *)s_osd_map + (size_t)buf_off, px,
           (size_t)w * (size_t)h * 4u);
    VkBufferImageCopy bc = {0};
    bc.bufferOffset = buf_off;
    bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bc.imageSubresource.layerCount = 1;
    bc.imageOffset.x = x;
    bc.imageOffset.y = y;
    bc.imageExtent.width = (uint32_t)dw;
    bc.imageExtent.height = (uint32_t)dh;
    bc.imageExtent.depth = 1;
    bc.bufferRowLength = (uint32_t)w;
    bc.bufferImageHeight = (uint32_t)h;
    p_vkCmdCopyBufferToImage(cb, s_osd_buf, sc,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
}

static void vk_osd_blit(VkCommandBuffer cb, VkImage sc) {
    const uint32_t *text_px = NULL, *vol_px = NULL, *rw_px = NULL, *ssm_px = NULL;
    int tw = 0, th = 0, vw = 0, vh = 0, rw = 0, rh = 0, sw = 0, sh = 0;
    const int have_text = host_osd_image(&text_px, &tw, &th) && text_px;
    const int have_vol = host_osd_volume_image(&vol_px, &vw, &vh) && vol_px;
    const int have_rw = psx_rewind_overlay_image(&rw_px, &rw, &rh) && rw_px;
    const int have_ssm =
        psx_host_menu_overlay_image(&ssm_px, &sw, &sh) && ssm_px;
    if (!have_text && !have_vol && !have_rw && !have_ssm) {
        host_osd_present_done();
        return;
    }
    /* Only BGRA swapchains match ARGB8888 LE packing used by host_osd. */
    if (s_sc_format != VK_FORMAT_B8G8R8A8_UNORM &&
        s_sc_format != VK_FORMAT_B8G8R8A8_SRGB) {
        host_osd_present_done();
        return;
    }
    VkDeviceSize text_bytes =
        have_text ? (VkDeviceSize)tw * (VkDeviceSize)th * 4u : 0;
    VkDeviceSize vol_bytes =
        have_vol ? (VkDeviceSize)vw * (VkDeviceSize)vh * 4u : 0;
    VkDeviceSize rw_bytes =
        have_rw ? (VkDeviceSize)rw * (VkDeviceSize)rh * 4u : 0;
    VkDeviceSize ssm_bytes =
        have_ssm ? (VkDeviceSize)sw * (VkDeviceSize)sh * 4u : 0;
    VkDeviceSize bytes = text_bytes + vol_bytes + rw_bytes + ssm_bytes;
    if (bytes > s_osd_cap) {
        p_vkQueueWaitIdle(s_queue);
        osd_staging_free();
        if (!make_staging(bytes, &s_osd_buf, &s_osd_mem, &s_osd_map)) {
            host_osd_present_done();
            return;
        }
        s_osd_cap = bytes;
    }
    const int margin = 8;
    VkDeviceSize off = 0;
    if (have_text) {
        vk_osd_copy_rect(cb, sc, text_px, tw, th, margin, margin, off);
        off += text_bytes;
    }
    if (have_vol) {
        int x = ((int)s_sc_extent.width > vw + margin)
                    ? ((int)s_sc_extent.width - vw - margin)
                    : margin;
        int y = ((int)s_sc_extent.height > vh)
                    ? (((int)s_sc_extent.height - vh) / 2)
                    : margin;
        vk_osd_copy_rect(cb, sc, vol_px, vw, vh, x, y, off);
        off += vol_bytes;
    }
    if (have_rw) {
        float slide = psx_rewind_slide();
        int x = ((int)s_sc_extent.width > rw)
                    ? (((int)s_sc_extent.width - rw) / 2)
                    : 0;
        int y = (int)s_sc_extent.height - (int)((float)rh * slide + 0.5f);
        vk_osd_copy_rect(cb, sc, rw_px, rw, rh, x, y, off);
        off += rw_bytes;
    }
    if (have_ssm) {
        int x = ((int)s_sc_extent.width > sw)
                    ? (((int)s_sc_extent.width - sw) / 2)
                    : 0;
        int y = ((int)s_sc_extent.height > sh)
                    ? (((int)s_sc_extent.height - sh) / 2)
                    : 0;
        vk_osd_copy_rect(cb, sc, ssm_px, sw, sh, x, y, off);
    }
    host_osd_present_done();
}

static void submit_present(VkCommandBuffer cb, uint32_t img_idx, uint32_t fr);
static void ov_draw(VkCommandBuffer cb, uint32_t idx);
static void ov_release_swapchain(void);
static void ov_destroy_all(void);

static void finish_present(VkCommandBuffer cb, VkImage sc,
                           uint32_t img_idx, uint32_t fr) {
    vk_osd_blit(cb, sc);
    ov_draw(cb, img_idx);
    touch_ui_presented();
    img_barrier(cb, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    submit_present(cb, img_idx, fr);
}

static void submit_present(VkCommandBuffer cb, uint32_t img_idx, uint32_t fr) {
    p_vkEndCommandBuffer(cb);
    /* Present command buffers write the acquired swapchain image with
     * transfer barriers, clears, and blits. Waiting only at color attachment
     * output does not order those operations after presentation-engine
     * release on stricter drivers. */
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT |
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = 1; si.pWaitSemaphores = &s_sem_acquire[fr];
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &s_sem_render[fr];
    p_vkQueueSubmit(s_queue, 1, &si, s_fence[fr]);

    VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &s_sem_render[fr];
    pi.swapchainCount = 1; pi.pSwapchains = &s_swapchain;
    pi.pImageIndices = &img_idx;
    uint64_t present_start = SDL_GetPerformanceCounter();
    psx_video_telemetry_compose(PSX_VIDEO_VULKAN, s_scale, 0, 0,
                               (int)s_sc_extent.width, (int)s_sc_extent.height, 0, 0);
    const VkResult present_result = p_vkQueuePresentKHR(s_queue, &pi);
    psx_video_telemetry_present(present_result == VK_SUCCESS || present_result == VK_SUBOPTIMAL_KHR);
    s_perf_cur.present_us += perf_elapsed_us(present_start);
    s_frame_idx++;
}

/* Compute a letterbox rect for the swapchain given a source aspect. */
static void letterbox(int sw, int sh, int aw, int ah, VkOffset3D off[2]) {
    if (touch_ui_fit_screen()) {   /* settings: stretch over the whole screen */
        off[0].x = 0;  off[0].y = 0;  off[0].z = 0;
        off[1].x = sw; off[1].y = sh; off[1].z = 1;
        return;
    }
    /* center-fit aw:ah inside sw x sh */
    int tw = sw, th = sw * ah / aw;
    if (th > sh) { th = sh; tw = sh * aw / ah; }
    int x = (sw - tw) / 2, y = (sh - th) / 2;
    off[0].x = x;       off[0].y = y;       off[0].z = 0;
    off[1].x = x + tw;  off[1].y = y + th;  off[1].z = 1;
}

/* ---- opaque swapchain present pass --------------------------------------- */
/* The plain present path blitted the VRAM image into the swapchain with
 * vkCmdBlitImage, which copies alpha too. VRAM alpha is the PSX mask/STP bit
 * (0 on most pixels), and AMD's compositor honours it despite
 * VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR: parts of the window turned transparent
 * (black/white blocks of whatever is behind it). This pass draws the displayed
 * rect with a fullscreen triangle whose fragment shader writes alpha = 1.
 * Everything is created lazily on first use; any failure sets s_pp_failed and
 * the callers fall back to the original blit. The render pass ends in
 * TRANSFER_DST_OPTIMAL so finish_present() (OSD blit + present barrier) runs
 * unchanged. Format-dependent objects are released with the swapchain. */
static VkRenderPass          s_pp_rp;
static VkImageView           s_pp_views[8];
static VkFramebuffer         s_pp_fbs[8];
static VkPipeline            s_pp_pipe;
static VkDescriptorSetLayout s_pp_dsl;
static VkPipelineLayout      s_pp_pl;
static VkDescriptorPool      s_pp_pool;
static VkDescriptorSet       s_pp_ds[VK_FRAMES];
static VkShaderModule        s_pp_vs, s_pp_fs;
static VkSampler             s_pp_near, s_pp_lin;
static int                   s_pp_failed;

typedef struct { float src_rect[4]; } PresentPush;

/* Swapchain-lifetime objects (the render pass depends on s_sc_format). */
static void pp_release_swapchain(void) {
    for (int i = 0; i < 8; i++) {
        if (s_pp_fbs[i])   { p_vkDestroyFramebuffer(s_dev, s_pp_fbs[i], NULL); s_pp_fbs[i] = VK_NULL_HANDLE; }
        if (s_pp_views[i]) { p_vkDestroyImageView(s_dev, s_pp_views[i], NULL); s_pp_views[i] = VK_NULL_HANDLE; }
    }
    if (s_pp_pipe) { p_vkDestroyPipeline(s_dev, s_pp_pipe, NULL); s_pp_pipe = VK_NULL_HANDLE; }
    if (s_pp_rp)   { p_vkDestroyRenderPass(s_dev, s_pp_rp, NULL); s_pp_rp = VK_NULL_HANDLE; }
    ov_release_swapchain();
}

static void pp_destroy_all(void) {
    pp_release_swapchain();
    ov_destroy_all();
    if (s_pp_pool) { p_vkDestroyDescriptorPool(s_dev, s_pp_pool, NULL); s_pp_pool = VK_NULL_HANDLE; }
    if (s_pp_pl)   { p_vkDestroyPipelineLayout(s_dev, s_pp_pl, NULL); s_pp_pl = VK_NULL_HANDLE; }
    if (s_pp_dsl)  { p_vkDestroyDescriptorSetLayout(s_dev, s_pp_dsl, NULL); s_pp_dsl = VK_NULL_HANDLE; }
    if (s_pp_vs)   { p_vkDestroyShaderModule(s_dev, s_pp_vs, NULL); s_pp_vs = VK_NULL_HANDLE; }
    if (s_pp_fs)   { p_vkDestroyShaderModule(s_dev, s_pp_fs, NULL); s_pp_fs = VK_NULL_HANDLE; }
    if (s_pp_near) { p_vkDestroySampler(s_dev, s_pp_near, NULL); s_pp_near = VK_NULL_HANDLE; }
    if (s_pp_lin)  { p_vkDestroySampler(s_dev, s_pp_lin, NULL); s_pp_lin = VK_NULL_HANDLE; }
    s_pp_failed = 0;
}

/* Device-lifetime objects: shaders, samplers, set layout, pool, sets, layout. */
static int pp_init_device_objects(void) {
    if (s_pp_pl) return 1;
    s_pp_vs = make_module(spv_present_vert, spv_present_vert_size);
    s_pp_fs = make_module(spv_present_frag, spv_present_frag_size);
    if (!s_pp_vs || !s_pp_fs) return 0;

    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (p_vkCreateSampler(s_dev, &sci, NULL, &s_pp_near) != VK_SUCCESS) return 0;
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    if (p_vkCreateSampler(s_dev, &sci, NULL, &s_pp_lin) != VK_SUCCESS) return 0;

    VkDescriptorSetLayoutBinding b = {0};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dci.bindingCount = 1; dci.pBindings = &b;
    if (p_vkCreateDescriptorSetLayout(s_dev, &dci, NULL, &s_pp_dsl) != VK_SUCCESS) return 0;

    /* One set per in-flight frame: acquire_present waits the frame fence, so
     * the set is idle when it is rewritten for the next present. */
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_FRAMES };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = VK_FRAMES; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (p_vkCreateDescriptorPool(s_dev, &pci, NULL, &s_pp_pool) != VK_SUCCESS) return 0;
    for (int i = 0; i < VK_FRAMES; i++) {
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = s_pp_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &s_pp_dsl;
        if (p_vkAllocateDescriptorSets(s_dev, &ai, &s_pp_ds[i]) != VK_SUCCESS) return 0;
    }

    VkPushConstantRange pr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PresentPush) };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1; li.pSetLayouts = &s_pp_dsl;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    return p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_pp_pl) == VK_SUCCESS;
}

static int pp_init_swapchain_objects(void) {
    if (s_pp_pipe) return 1;
    VkAttachmentDescription att = {0};
    att.format = s_sc_format; att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;          /* black letterbox bars */
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;   /* finish_present */
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &cref;
    VkSubpassDependency dep[2] = {0};
    dep[0].srcSubpass = VK_SUBPASS_EXTERNAL; dep[0].dstSubpass = 0;
    dep[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    /* The OSD blit in finish_present writes the image with transfer ops. */
    dep[1].srcSubpass = 0; dep[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dep[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkRenderPassCreateInfo rpi = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpi.attachmentCount = 1; rpi.pAttachments = &att;
    rpi.subpassCount = 1; rpi.pSubpasses = &sub;
    rpi.dependencyCount = 2; rpi.pDependencies = dep;
    if (p_vkCreateRenderPass(s_dev, &rpi, NULL, &s_pp_rp) != VK_SUCCESS) return 0;

    VkPipelineShaderStageCreateInfo st[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = s_pp_vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = s_pp_fs; st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba = {0};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1; cb.pAttachments = &ba;
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    ci.stageCount = 2; ci.pStages = st;
    ci.pVertexInputState = &vin; ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp; ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms; ci.pColorBlendState = &cb; ci.pDynamicState = &dy;
    ci.layout = s_pp_pl; ci.renderPass = s_pp_rp;
    return p_vkCreateGraphicsPipelines(s_dev, VK_NULL_HANDLE, 1, &ci, NULL, &s_pp_pipe) == VK_SUCCESS;
}

/* Framebuffer for swapchain image idx, created on first use. */
static VkFramebuffer pp_framebuffer(uint32_t idx) {
    if (idx >= 8 || idx >= s_sc_count) return VK_NULL_HANDLE;
    if (s_pp_fbs[idx]) return s_pp_fbs[idx];
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = s_sc_images[idx]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = s_sc_format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    if (p_vkCreateImageView(s_dev, &vi, NULL, &s_pp_views[idx]) != VK_SUCCESS) {
        s_pp_views[idx] = VK_NULL_HANDLE; return VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fi.renderPass = s_pp_rp; fi.attachmentCount = 1; fi.pAttachments = &s_pp_views[idx];
    fi.width = s_sc_extent.width; fi.height = s_sc_extent.height; fi.layers = 1;
    if (p_vkCreateFramebuffer(s_dev, &fi, NULL, &s_pp_fbs[idx]) != VK_SUCCESS)
        s_pp_fbs[idx] = VK_NULL_HANDLE;
    return s_pp_fbs[idx];
}

/* Record the opaque present of src_view (already SHADER_READ_ONLY_OPTIMAL,
 * normalized src rect u0..v1) into the letterbox rect dst of the acquired
 * swapchain image, leaving it in TRANSFER_DST_OPTIMAL. Returns 0 (and records
 * nothing) when the pass is unavailable: the caller then uses the blit. */
static int pp_draw(VkCommandBuffer cb, uint32_t idx, uint32_t fr, VkImageView src_view,
                   int linear, const VkOffset3D dst[2],
                   float u0, float v0, float u1, float v1) {
    if (s_pp_failed) return 0;
    if (!pp_init_device_objects() || !pp_init_swapchain_objects()) {
        s_pp_failed = 1;
        fprintf(stdout, "psxrecomp: vulkan: present pass unavailable, using blit\n");
        return 0;
    }
    VkFramebuffer fb = pp_framebuffer(idx);
    if (!fb) return 0;

    VkDescriptorImageInfo ii = { linear ? s_pp_lin : s_pp_near, src_view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = s_pp_ds[fr]; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    p_vkUpdateDescriptorSets(s_dev, 1, &w, 0, NULL);

    VkClearValue clear = {0};
    clear.color.float32[3] = 1.0f;
    VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    bi.renderPass = s_pp_rp; bi.framebuffer = fb;
    bi.renderArea.extent = s_sc_extent;
    bi.clearValueCount = 1; bi.pClearValues = &clear;
    p_vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport view = { (float)dst[0].x, (float)dst[0].y,
                        (float)(dst[1].x - dst[0].x), (float)(dst[1].y - dst[0].y), 0.0f, 1.0f };
    VkRect2D sci = { { dst[0].x, dst[0].y },
                     { (uint32_t)(dst[1].x - dst[0].x), (uint32_t)(dst[1].y - dst[0].y) } };
    p_vkCmdSetViewport(cb, 0, 1, &view);
    p_vkCmdSetScissor(cb, 0, 1, &sci);
    p_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pp_pipe);
    p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pp_pl, 0, 1, &s_pp_ds[fr], 0, NULL);
    PresentPush pc = { { u0, v0, u1, v1 } };
    p_vkCmdPushConstants(cb, s_pp_pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
    p_vkCmdDraw(cb, 3, 1, 0, 0);
    p_vkCmdEndRenderPass(cb);
    return 1;
}

/* ---- touch UI overlay pass ------------------------------------------------ */
/* host_touch_ui draws the on-screen pad, settings button and settings panel
 * into one premultiplied ARGB canvas. It is uploaded when it changes and
 * alpha-blended over the finished swapchain image, which stays in
 * TRANSFER_DST_OPTIMAL for finish_present's barrier. Created lazily; any
 * failure turns the overlay off for the session. */
static VkRenderPass          s_ov_rp;
static VkImageView           s_ov_views[8];
static VkFramebuffer         s_ov_fbs[8];
static VkPipeline            s_ov_pipe;
static VkDescriptorSetLayout s_ov_dsl;
static VkPipelineLayout      s_ov_pl;
static VkDescriptorPool      s_ov_pool;
static VkDescriptorSet       s_ov_ds;
static VkShaderModule        s_ov_vs, s_ov_fs;
static VkSampler             s_ov_samp;
static VkImage               s_ov_img;
static VkDeviceMemory        s_ov_mem;
static VkImageView           s_ov_view;
static VkImageLayout         s_ov_layout;
static int                   s_ov_w, s_ov_h, s_ov_failed;
static unsigned              s_ov_version;
static VkBuffer              s_ov_buf;
static VkDeviceMemory        s_ov_bmem;
static void                 *s_ov_map;

static void ov_release_swapchain(void) {
    for (int i = 0; i < 8; i++) {
        if (s_ov_fbs[i])   { p_vkDestroyFramebuffer(s_dev, s_ov_fbs[i], NULL); s_ov_fbs[i] = VK_NULL_HANDLE; }
        if (s_ov_views[i]) { p_vkDestroyImageView(s_dev, s_ov_views[i], NULL); s_ov_views[i] = VK_NULL_HANDLE; }
    }
    if (s_ov_pipe) { p_vkDestroyPipeline(s_dev, s_ov_pipe, NULL); s_ov_pipe = VK_NULL_HANDLE; }
    if (s_ov_rp)   { p_vkDestroyRenderPass(s_dev, s_ov_rp, NULL); s_ov_rp = VK_NULL_HANDLE; }
}

static void ov_release_image(void) {
    if (s_ov_view) { p_vkDestroyImageView(s_dev, s_ov_view, NULL); s_ov_view = VK_NULL_HANDLE; }
    if (s_ov_img)  { p_vkDestroyImage(s_dev, s_ov_img, NULL); s_ov_img = VK_NULL_HANDLE; }
    if (s_ov_mem)  { p_vkFreeMemory(s_dev, s_ov_mem, NULL); s_ov_mem = VK_NULL_HANDLE; }
    if (s_ov_buf)  { staging_destroy(s_ov_buf, s_ov_bmem); s_ov_buf = VK_NULL_HANDLE; s_ov_bmem = VK_NULL_HANDLE; }
    s_ov_map = NULL; s_ov_w = s_ov_h = 0; s_ov_version = 0;
    s_ov_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

static void ov_destroy_all(void) {
    ov_release_swapchain();
    ov_release_image();
    if (s_ov_pool) { p_vkDestroyDescriptorPool(s_dev, s_ov_pool, NULL); s_ov_pool = VK_NULL_HANDLE; }
    if (s_ov_pl)   { p_vkDestroyPipelineLayout(s_dev, s_ov_pl, NULL); s_ov_pl = VK_NULL_HANDLE; }
    if (s_ov_dsl)  { p_vkDestroyDescriptorSetLayout(s_dev, s_ov_dsl, NULL); s_ov_dsl = VK_NULL_HANDLE; }
    if (s_ov_vs)   { p_vkDestroyShaderModule(s_dev, s_ov_vs, NULL); s_ov_vs = VK_NULL_HANDLE; }
    if (s_ov_fs)   { p_vkDestroyShaderModule(s_dev, s_ov_fs, NULL); s_ov_fs = VK_NULL_HANDLE; }
    if (s_ov_samp) { p_vkDestroySampler(s_dev, s_ov_samp, NULL); s_ov_samp = VK_NULL_HANDLE; }
    s_ov_ds = VK_NULL_HANDLE;
    s_ov_failed = 0;
}

static int ov_init_device_objects(void) {
    if (s_ov_pl) return 1;
    s_ov_vs = make_module(spv_present_vert, spv_present_vert_size);
    s_ov_fs = make_module(spv_overlay_frag, spv_overlay_frag_size);
    if (!s_ov_vs || !s_ov_fs) return 0;
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (p_vkCreateSampler(s_dev, &sci, NULL, &s_ov_samp) != VK_SUCCESS) return 0;
    VkDescriptorSetLayoutBinding b = {0};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dci.bindingCount = 1; dci.pBindings = &b;
    if (p_vkCreateDescriptorSetLayout(s_dev, &dci, NULL, &s_ov_dsl) != VK_SUCCESS) return 0;
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (p_vkCreateDescriptorPool(s_dev, &pci, NULL, &s_ov_pool) != VK_SUCCESS) return 0;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = s_ov_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &s_ov_dsl;
    if (p_vkAllocateDescriptorSets(s_dev, &ai, &s_ov_ds) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1; li.pSetLayouts = &s_ov_dsl;
    return p_vkCreatePipelineLayout(s_dev, &li, NULL, &s_ov_pl) == VK_SUCCESS;
}

static int ov_init_swapchain_objects(void) {
    if (s_ov_pipe) return 1;
    VkAttachmentDescription att = {0};
    att.format = s_sc_format; att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;           /* keep the game picture */
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkAttachmentReference cref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1; sub.pColorAttachments = &cref;
    VkSubpassDependency dep[2] = {0};
    dep[0].srcSubpass = VK_SUBPASS_EXTERNAL; dep[0].dstSubpass = 0;
    dep[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[1].srcSubpass = 0; dep[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dep[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dep[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    VkRenderPassCreateInfo rpi = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpi.attachmentCount = 1; rpi.pAttachments = &att;
    rpi.subpassCount = 1; rpi.pSubpasses = &sub;
    rpi.dependencyCount = 2; rpi.pDependencies = dep;
    if (p_vkCreateRenderPass(s_dev, &rpi, NULL, &s_ov_rp) != VK_SUCCESS) return 0;

    VkPipelineShaderStageCreateInfo st[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = s_ov_vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = s_ov_fs; st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba = {0};
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.alphaBlendOp = VK_BLEND_OP_ADD;
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cbs.attachmentCount = 1; cbs.pAttachments = &ba;
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo ci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    ci.stageCount = 2; ci.pStages = st;
    ci.pVertexInputState = &vin; ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp; ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms; ci.pColorBlendState = &cbs; ci.pDynamicState = &dy;
    ci.layout = s_ov_pl; ci.renderPass = s_ov_rp;
    return p_vkCreateGraphicsPipelines(s_dev, VK_NULL_HANDLE, 1, &ci, NULL, &s_ov_pipe) == VK_SUCCESS;
}

static VkFramebuffer ov_framebuffer(uint32_t idx) {
    if (idx >= 8 || idx >= s_sc_count) return VK_NULL_HANDLE;
    if (s_ov_fbs[idx]) return s_ov_fbs[idx];
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = s_sc_images[idx]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = s_sc_format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    if (p_vkCreateImageView(s_dev, &vi, NULL, &s_ov_views[idx]) != VK_SUCCESS) {
        s_ov_views[idx] = VK_NULL_HANDLE; return VK_NULL_HANDLE;
    }
    VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fi.renderPass = s_ov_rp; fi.attachmentCount = 1; fi.pAttachments = &s_ov_views[idx];
    fi.width = s_sc_extent.width; fi.height = s_sc_extent.height; fi.layers = 1;
    if (p_vkCreateFramebuffer(s_dev, &fi, NULL, &s_ov_fbs[idx]) != VK_SUCCESS)
        s_ov_fbs[idx] = VK_NULL_HANDLE;
    return s_ov_fbs[idx];
}

/* Blend the touch UI over swapchain image idx (in TRANSFER_DST_OPTIMAL). */
static void ov_draw(VkCommandBuffer cb, uint32_t idx) {
    const uint32_t *px; int w, h; unsigned ver;
    if (s_ov_failed) return;
    if (!touch_ui_canvas((int)s_sc_extent.width, (int)s_sc_extent.height, &px, &w, &h, &ver)) return;
    if (!ov_init_device_objects() || !ov_init_swapchain_objects()) {
        s_ov_failed = 1;
        fprintf(stdout, "psxrecomp: vulkan: touch overlay pass unavailable\n");
        return;
    }
    if (w != s_ov_w || h != s_ov_h || !s_ov_img) {
        p_vkDeviceWaitIdle(s_dev);
        ov_release_image();
        if (!make_image(VK_FORMAT_B8G8R8A8_UNORM, w, h,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, &s_ov_img, &s_ov_mem, &s_ov_view) ||
            !make_staging((VkDeviceSize)w * (VkDeviceSize)h * 4u, &s_ov_buf, &s_ov_bmem, &s_ov_map)) {
            s_ov_failed = 1;
            fprintf(stdout, "psxrecomp: vulkan: touch overlay image unavailable\n");
            return;
        }
        s_ov_w = w; s_ov_h = h;
        VkDescriptorImageInfo ii = { s_ov_samp, s_ov_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        wr.dstSet = s_ov_ds; wr.dstBinding = 0; wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr.pImageInfo = &ii;
        p_vkUpdateDescriptorSets(s_dev, 1, &wr, 0, NULL);
    }
    if (ver != s_ov_version || s_ov_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        memcpy(s_ov_map, px, (size_t)w * (size_t)h * 4u);
        img_barrier(cb, s_ov_img, s_ov_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy bc = {0};
        bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bc.imageSubresource.layerCount = 1;
        bc.imageExtent.width = (uint32_t)w; bc.imageExtent.height = (uint32_t)h; bc.imageExtent.depth = 1;
        p_vkCmdCopyBufferToImage(cb, s_ov_buf, s_ov_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
        img_barrier(cb, s_ov_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        s_ov_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        s_ov_version = ver;
    }
    VkFramebuffer fb = ov_framebuffer(idx);
    if (!fb) return;
    VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    bi.renderPass = s_ov_rp; bi.framebuffer = fb;
    bi.renderArea.extent = s_sc_extent;
    p_vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport view = { 0.0f, 0.0f, (float)s_sc_extent.width, (float)s_sc_extent.height, 0.0f, 1.0f };
    VkRect2D sci = { { 0, 0 }, s_sc_extent };
    p_vkCmdSetViewport(cb, 0, 1, &view);
    p_vkCmdSetScissor(cb, 0, 1, &sci);
    p_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_ov_pipe);
    p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_ov_pl, 0, 1, &s_ov_ds, 0, NULL);
    p_vkCmdDraw(cb, 3, 1, 0, 0);
    p_vkCmdEndRenderPass(cb);
}

int vk_renderer_present_vram(int disp_x, int disp_y, int w, int h,
                             int linear, int force_4_3) {
    if (!s_ctx_ok) return 0;
    flush_cpu_upload();   /* displayed VRAM may include pending CPU writes */
    flush_tex_batch(); flush_geometry(); gpu_sync();   /* drain all draws; VRAM in TRANSFER_SRC */
    VkImage sc; VkCommandBuffer cb; uint32_t idx, fr;
    if (!acquire_present(&sc, &cb, &idx, &fr)) return 1; /* frame skipped/recreated */

    int S = s_scale;
    VkOffset3D dst[2];
    letterbox((int)s_sc_extent.width, (int)s_sc_extent.height,
              force_4_3 ? 4 : s_aspect_num, force_4_3 ? 3 : s_aspect_den, dst);

    /* Opaque shader present first (see pp_draw); blit only as the fallback. */
    vram_to(cb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    int drawn = pp_draw(cb, idx, fr, s_vram_view, linear, dst,
                        disp_x / (float)VRAM_W, disp_y / (float)VRAM_H,
                        (disp_x + w) / (float)VRAM_W, (disp_y + h) / (float)VRAM_H);
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);   /* re-park */
    if (drawn) {
        finish_present(cb, sc, idx, fr);
        perf_snapshot_present();
        return 1;
    }

    img_barrier(cb, sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    /* clear the swapchain (letterbox bars) to black */
    sc_clear_black(cb, sc);

    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0].x = disp_x * S;        blit.srcOffsets[0].y = disp_y * S;
    blit.srcOffsets[1].x = (disp_x + w) * S;  blit.srcOffsets[1].y = (disp_y + h) * S;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = dst[0]; blit.dstOffsets[1] = dst[1];

    /* VRAM image is parked in TRANSFER_SRC_OPTIMAL. */
    p_vkCmdBlitImage(cb, s_vram_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     1, &blit, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);

    finish_present(cb, sc, idx, fr);
    perf_snapshot_present();
    return 1;
}

void vk_renderer_present_blank(void) {
    if (!s_ctx_ok) return;
    flush_tex_batch(); flush_geometry(); gpu_sync();
    VkImage sc; VkCommandBuffer cb; uint32_t idx, fr;
    if (!acquire_present(&sc, &cb, &idx, &fr)) return;
    img_barrier(cb, sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    sc_clear_black(cb, sc);
    finish_present(cb, sc, idx, fr);
    perf_snapshot_present();
}

/* Present an ARGB8888 image (24-bit FMV / display-disabled clear) letterboxed.
 * The pixels are 0xAARRGGBB; little-endian byte order [B,G,R,A] matches a
 * B8G8R8A8_UNORM staging image, blitted straight to the swapchain. */
/* CPU-present resource cache (FMV path): the depth24 compose presents a new
 * frame 15-24x/s for minutes; creating + destroying a VkImage and a staging
 * buffer per frame is the same driver-allocation churn class the batched
 * upload fix killed. Cache both keyed by (w,h) — zero allocations at steady
 * state; the staging stays persistently mapped. Freed on resize + shutdown.
 * The per-frame vkQueueWaitIdle below also makes reuse race-free (the copy
 * from the staging has fully completed before the next frame rewrites it). */
static VkImage        s_cpres_img  = VK_NULL_HANDLE;
static VkDeviceMemory s_cpres_imem = VK_NULL_HANDLE;
static VkImageLayout  s_cpres_layout = VK_IMAGE_LAYOUT_UNDEFINED;
static VkBuffer       s_cpres_buf  = VK_NULL_HANDLE;
static VkDeviceMemory s_cpres_mem  = VK_NULL_HANDLE;
static void          *s_cpres_map  = NULL;
static int            s_cpres_w = 0, s_cpres_h = 0;

static void cpres_cache_free(void) {
    if (s_cpres_img)  { p_vkDestroyImage(s_dev, s_cpres_img, NULL);  s_cpres_img = VK_NULL_HANDLE; }
    if (s_cpres_imem) { p_vkFreeMemory(s_dev, s_cpres_imem, NULL);   s_cpres_imem = VK_NULL_HANDLE; }
    if (s_cpres_buf)  {
        staging_destroy(s_cpres_buf, s_cpres_mem);
        s_cpres_buf = VK_NULL_HANDLE;
        s_cpres_mem = VK_NULL_HANDLE;
    }
    s_cpres_map = NULL; s_cpres_w = s_cpres_h = 0;
    s_cpres_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void vk_renderer_present_cpu(const uint32_t *pixels, int src_w, int src_h,
                             int linear, int force_4_3) {
    (void)force_4_3;
    if (!s_ctx_ok || !pixels || src_w <= 0 || src_h <= 0) { vk_renderer_present_blank(); return; }
    flush_tex_batch(); flush_geometry(); gpu_sync();

    if (s_cpres_w != src_w || s_cpres_h != src_h) {
        p_vkQueueWaitIdle(s_queue);      /* old resources may be in flight */
        cpres_cache_free();
        if (!make_staging((VkDeviceSize)src_w * src_h * 4,
                          &s_cpres_buf, &s_cpres_mem, &s_cpres_map)) {
            vk_renderer_present_blank(); return;
        }
        if (!make_image(VK_FORMAT_B8G8R8A8_UNORM, src_w, src_h,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, &s_cpres_img, &s_cpres_imem, NULL)) {
            cpres_cache_free(); vk_renderer_present_blank(); return;
        }
        s_cpres_w = src_w; s_cpres_h = src_h;
        s_cpres_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    memcpy(s_cpres_map, pixels, (size_t)src_w * src_h * 4);
    VkBuffer buf = s_cpres_buf;
    VkImage img = s_cpres_img;

    VkImage sc; VkCommandBuffer cb; uint32_t idx, fr;
    if (!acquire_present(&sc, &cb, &idx, &fr)) return;
    img_to(cb, img, &s_cpres_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy bc = {0};
    bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bc.imageSubresource.layerCount = 1;
    bc.imageExtent.width = src_w; bc.imageExtent.height = src_h; bc.imageExtent.depth = 1;
    p_vkCmdCopyBufferToImage(cb, buf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
    img_to(cb, img, &s_cpres_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    img_barrier(cb, sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    sc_clear_black(cb, sc);

    VkOffset3D dst[2];
    letterbox((int)s_sc_extent.width, (int)s_sc_extent.height, force_4_3 ? 4 : s_aspect_num, force_4_3 ? 3 : s_aspect_den, dst);
    /* Short GP1(07h) bands: letterbox inside the 4:3 rect (see GL present). */
    if (src_h > 0 && src_h < 240) {
        int box_h = dst[1].y - dst[0].y;
        int content_h = (box_h * src_h) / 240;
        if (content_h < 1) content_h = 1;
        int y0 = dst[0].y + (box_h - content_h) / 2;
        dst[0].y = y0;
        dst[1].y = y0 + content_h;
    }
    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = src_w; blit.srcOffsets[1].y = src_h; blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = dst[0]; blit.dstOffsets[1] = dst[1];
    p_vkCmdBlitImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     1, &blit, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    finish_present(cb, sc, idx, fr);

    /* Cached resources are rewritten next frame: wait for this frame's copy +
     * blit to complete first (FMV cadence is 15-24 fps; one idle-wait per
     * movie frame is cheap, and it is what makes the persistent staging safe). */
    p_vkQueueWaitIdle(s_queue);
    perf_snapshot_present();
}

/* GPU-direct native-wide present: blit the displayed buffer's wide surface
 * band straight to the swapchain (no readback), mirroring
 * gl_renderer_present_wide_fbo. Letterbox aspect generalises past 16:9:
 * the 4:3 native frame widened by EXTRA presents at (4*wide_w : 3*native_w).
 * Returns 0 when no wide surface exists for disp_x (caller falls back). */
static int present_wide_surf(int i, int disp_y, int disp_h, int linear);
/* Scheduled pictures of the current game frame (90/120 fps). */
static struct { int surf; uint64_t due; } s_fgq[4];
static int s_fgq_n, s_fgq_i, s_fgq_y, s_fgq_h, s_fgq_linear;
static uint64_t fgq_due(void) { return s_fgq_i < s_fgq_n ? s_fgq[s_fgq_i].due : 0; }
static void fgq_run(void) {
    while (s_fgq_i < s_fgq_n && SDL_GetPerformanceCounter() >= s_fgq[s_fgq_i].due) {
        const int surf = s_fgq[s_fgq_i++].surf;
        if (s_ready && surf >= 0 && s_wide_img[surf]) present_wide_surf(surf, s_fgq_y, s_fgq_h, s_fgq_linear);
    }
}
int vk_renderer_present_wide(int disp_x, int disp_y, int disp_h, int linear) {
    if (!s_ctx_ok || !s_ready || s_wide_w <= 0) return 0;
    int i = -1;
    for (int k = 0; k < VK_WIDE_MAX_SURF; k++)
        if (s_wide_img[k] && s_wide_base[k] == disp_x) { i = k; break; }
    if (i < 0) return 0;
    if (s_fgq_i < s_fgq_n) {
        /* Pictures of the current frame are still scheduled (90/120): the
         * pacer pump shows them; this vblank adds nothing. */
        if (!s_fg_hold) { fgq_run(); return 1; }
        s_fgq_n = s_fgq_i = 0;          /* a new frame arrived early: drop the rest */
    }
    if (s_fg_hold && s_present_idx - s_fg_hold_stamp <= 5) {
        /* Generated in-between pictures go out before the real frame. With one
         * (60 fps) the real frame follows at the next vblank. With two or
         * three (90/120) the rest are scheduled evenly across the game frame
         * and shown from the pacer's wait, one at a time. */
        int g[3], ok = 1;
        for (int k = 0; k < s_fg_n; k++) { g[k] = s_fg_surfs[k]; if (g[k] < 0 || !s_wide_img[g[k]]) ok = 0; }
        s_fg_hold = 0;
        if (ok && s_fg_n >= 2) {
            const uint64_t now = SDL_GetPerformanceCounter();
            const double frame = (double)SDL_GetPerformanceFrequency() * (2.0 / 59.94);
            s_fgq_n = s_fgq_i = 0;
            for (int k = 1; k <= s_fg_n; k++) {
                s_fgq[s_fgq_n].surf = k < s_fg_n ? g[k] : i;
                s_fgq[s_fgq_n].due = now + (uint64_t)(frame * (double)k / (double)(s_fg_n + 1));
                s_fgq_n++;
            }
            s_fgq_y = disp_y; s_fgq_h = disp_h; s_fgq_linear = linear;
            return present_wide_surf(g[0], disp_y, disp_h, linear);
        }
        if (ok && s_fg_n == 1) i = g[0];
    } else {
        s_fg_hold = 0;
        /* Diagnostic: PSX_VK_FG_DEBUG=1 shows only generated pictures. */
        static int dbg = -1;
        if (dbg < 0) { const char *e = getenv("PSX_VK_FG_DEBUG"); dbg = e && *e == '1'; }
        if (dbg && s_fg_n >= 1 && s_fg_surfs[0] >= 0 && s_wide_img[s_fg_surfs[0]]) i = s_fg_surfs[0];
    }
    return present_wide_surf(i, disp_y, disp_h, linear);
}

/* Diagnostic: PSX_VK_SEQ_DUMP=<dir>. Creating <dir>/go dumps the next 48
 * presented pictures, in presentation order, as raw RGBA files. */
static int make_staging(VkDeviceSize bytes, VkBuffer *buf, VkDeviceMemory *mem, void **map);
static void seq_dump(int i, int disp_y, int disp_h) {
    static const char *dir; static int init, left, serial, step = 1, count = 48;
    if (!init) {
        init = 1; dir = getenv("PSX_VK_SEQ_DUMP");
        const char *st = getenv("PSX_VK_SEQ_STEP");
        if (st && atoi(st) > 0) step = atoi(st);
        st = getenv("PSX_VK_SEQ_COUNT");
        if (st && atoi(st) > 0) count = atoi(st);
    }
    if (!dir || !*dir) return;
    char path[512];
    if (!left) {
        if ((s_present_idx & 15u) != 0) return;
        snprintf(path, sizeof path, "%s/go", dir);
        if (remove(path) != 0) return;
        left = count; serial = 0;
    }
    if (step > 1 && (s_present_idx % (unsigned)step) != 0) return;
    const int S = s_scale, W = s_wide_w * S;
    int y0 = disp_y * S, h = disp_h * S;
    if (y0 < 0) y0 = 0;
    if (y0 + h > VRAM_H * S) h = VRAM_H * S - y0;
    if (h <= 0) return;
    VkBuffer buf; VkDeviceMemory mem; void *map;
    if (!make_staging((VkDeviceSize)W * h * 4, &buf, &mem, &map)) return;
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy rc = {0};
    rc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rc.imageSubresource.layerCount = 1;
    rc.imageOffset.y = y0;
    rc.imageExtent.width = (uint32_t)W; rc.imageExtent.height = (uint32_t)h; rc.imageExtent.depth = 1;
    p_vkCmdCopyImageToBuffer(cb, s_wide_img[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &rc);
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    end_oneshot(cb);
    gpu_sync();
    snprintf(path, sizeof path, "%s/seq_%03d.raw", dir, serial++);
    FILE *f = fopen(path, "wb");
    if (f) {
        const int hdr[4] = { W, h, i, disp_y };
        fwrite(hdr, sizeof hdr, 1, f); fwrite(map, 4, (size_t)W * h, f); fclose(f);
    }
    free_staging(buf, mem);
    left--;
}

static int present_wide_surf(int i, int disp_y, int disp_h, int linear) {
    flush_cpu_upload();
    flush_tex_batch(); flush_geometry(); gpu_sync();
    seq_dump(i, disp_y, disp_h);
    VkImage sc; VkCommandBuffer cb; uint32_t idx, fr;
    if (!acquire_present(&sc, &cb, &idx, &fr)) return 1;  /* frame skipped/recreated */

    int S = s_scale;
    int native_w = s_wide_w - 2 * s_wide_offset;
    if (native_w <= 0) native_w = s_wide_w;
    VkOffset3D dst[2];
    letterbox((int)s_sc_extent.width, (int)s_sc_extent.height,
              4 * s_wide_w, 3 * native_w, dst);

    /* Opaque shader present first (see pp_draw); blit only as the fallback. */
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    int drawn = pp_draw(cb, idx, fr, s_wide_view[i], linear, dst,
                        0.0f, disp_y / (float)VRAM_H,
                        1.0f, (disp_y + disp_h) / (float)VRAM_H);
    if (drawn) {
        img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        finish_present(cb, sc, idx, fr);
        perf_snapshot_present();
        return 1;
    }

    img_barrier(cb, sc, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    sc_clear_black(cb, sc);

    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    int sy0 = disp_y * S, sy1 = (disp_y + disp_h) * S;
    if (sy0 < 0) sy0 = 0;
    if (sy1 > VRAM_H * S) sy1 = VRAM_H * S;
    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0].x = 0;              blit.srcOffsets[0].y = sy0;
    blit.srcOffsets[1].x = s_wide_w * S;   blit.srcOffsets[1].y = sy1;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = dst[0]; blit.dstOffsets[1] = dst[1];
    p_vkCmdBlitImage(cb, s_wide_img[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     1, &blit, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    finish_present(cb, sc, idx, fr);
    perf_snapshot_present();
    return 1;
}

void vk_renderer_sync_cpu(void) {
    if (!s_ctx_ok) return;
    ensure_cpu();   /* drain batches, pack hr->raw, copy raw->CPU mirror */
}

/* Dump the last `count` per-frame perf records as a JSON array (oldest..newest).
 * Always-on; the debug-server "vk_perf" command reads this to find which op
 * explodes during a transition stall (e.g. allocs jumping 10 -> thousands). */
int vk_perf_json(char *out, int cap, int count) {
    uint32_t total = s_perf_head;
    uint32_t n = ((uint32_t)count < total) ? (uint32_t)count : total;
    if (n > VK_PERF_RING) n = VK_PERF_RING;
    int o = snprintf(out, cap, "[");
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k = total - n + i;          /* frame index, oldest..newest */
        VkPerf *p = &s_perf_ring[k % VK_PERF_RING];
        o += snprintf(out + o, cap - o,
            "%s{\"f\":%u,\"qpc\":%llu,\"wait_us\":%u,\"acquire_us\":%u,\"present_us\":%u,"
            "\"alloc\":%u,\"alloc_kb\":%u,\"oneshot\":%u,\"submit\":%u,"
            "\"sync\":%u,\"pack\":%u,\"blit\":%u,\"upload\":%u,\"copy\":%u,"
            "\"geo\":%u,\"tex\":%u,\"wide\":%u,\"wclr\":%u}",
            i ? "," : "",
            p->present_idx, (unsigned long long)p->present_qpc,
            p->wait_us, p->acquire_us, p->present_us,
            p->allocs, p->alloc_kb, p->oneshots, p->submits,
            p->syncs, p->pack_flushes, p->blits, p->upload_blocks, p->copy_rects,
            p->geo_flushes, p->tex_flushes, p->wide_passes, p->wide_clears);
        if (o >= cap - 256) break;
    }
    o += snprintf(out + o, cap - o, "]");
    return o;
}

/* ---- render-pass + coherency helpers ----------------------------------- */
/* Half an HR pixel in native units, minus 1/64 to keep exact-edge samples off
 * texel centres (see geo.vert / the GL backend's u_shift). */
static float px_shift(void) { return 0.5f / (float)s_scale - 1.0f / 64.0f; }

static void set_vp_full(VkCommandBuffer cb) {
    VkViewport vp = { 0.f, 0.f, (float)(VRAM_W * s_scale), (float)(VRAM_H * s_scale), 0.f, 1.f };
    p_vkCmdSetViewport(cb, 0, 1, &vp);
}
static void set_scissor_px(VkCommandBuffer cb, int x, int y, int w, int h) {
    if (x < 0) { w += x; x = 0; } if (y < 0) { h += y; y = 0; }
    if (x + w > VRAM_W) w = VRAM_W - x; if (y + h > VRAM_H) h = VRAM_H - y;
    if (w < 0) w = 0; if (h < 0) h = 0;
    VkRect2D sc = { { x * s_scale, y * s_scale }, { (uint32_t)(w * s_scale), (uint32_t)(h * s_scale) } };
    p_vkCmdSetScissor(cb, 0, 1, &sc);
}

/* A same-layout transition is elided by img_to(), while the render pass has
 * no external dependency. Explicitly order a preceding pass's attachment
 * writes before a following LOAD/blend pass on the same color image. */
static void color_self_barrier(VkCommandBuffer cb, VkImage img) {
    VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);
}

static void begin_geo_pass(VkCommandBuffer cb) {
    /* Explicit stencil ordering across one-shot submits: this pass both TESTS
     * and WRITES the stencil (PSX mask bits) that a PRIOR submit's pass wrote.
     * Cross-submit ordering previously rode on the COLOR image barriers'
     * execution dependencies alone — no memory dependency covered the DS
     * attachment, which is spec-fragile (worked on this NVIDIA driver, could
     * produce mask-bit flicker elsewhere). A self-barrier on the stencil
     * aspect (late-tests write -> early-tests read|write) makes prior stencil
     * writes visible before this pass's tests, per submission order. */
    VkImageMemoryBarrier db = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    db.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = s_ds_img;
    db.subresourceRange.aspectMask = DS_ASPECT;
    db.subresourceRange.levelCount = 1;
    db.subresourceRange.layerCount = 1;
    db.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, NULL, 0, NULL, 1, &db);
    color_self_barrier(cb, s_vram_img);
    VkRenderPassBeginInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    s_pipe_ms = 0;
    rp.renderPass = s_rpass; rp.framebuffer = s_fbo;
    rp.renderArea.extent.width = VRAM_W * s_scale;
    rp.renderArea.extent.height = VRAM_H * s_scale;
    p_vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
}
/* Bind a graphics pipeline implementing GL's mask_stencil: check==0 -> SET
 * (always+replace, ref=write_val); check!=0 -> test stored==0 and write 1 via
 * INVERT (write_val!=0) or leave (write_val==0), ref=0. */
static void bind_masked(VkCommandBuffer cb, int prog, int topo, int blend, int check, int write_val) {
    int stencil = !check ? 0 : (write_val ? 2 : 1);
    VkPipeline p = get_pipeline(prog, topo, blend, stencil, 0);
    p_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    p_vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, check ? 0u : (uint32_t)write_val);
}
/* Like bind_masked but with ALL colour writes disabled (stencil-only pass). */
static void bind_masked_stencil_only(VkCommandBuffer cb, int prog, int topo, int check, int write_val) {
    int stencil = !check ? 0 : (write_val ? 2 : 1);
    VkPipeline p = get_pipeline(prog, topo, 0, stencil, 1);
    p_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    p_vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, check ? 0u : (uint32_t)write_val);
}

/* Host-visible staging buffer (TRANSFER_SRC). */
static int make_staging(VkDeviceSize bytes, VkBuffer *buf, VkDeviceMemory *mem, void **map) {
    int best = -1;
    for (int i = 0; i < STAGING_CACHE_MAX; ++i) {
        StagingCacheEntry *entry = &s_staging_cache[i];
        if (!entry->busy && entry->buf && entry->size >= bytes &&
            (best < 0 || entry->size < s_staging_cache[best].size))
            best = i;
    }
    if (best >= 0) {
        StagingCacheEntry *entry = &s_staging_cache[best];
        entry->busy = 1;
        *buf = entry->buf;
        *mem = entry->mem;
        *map = entry->map;
        return 1;
    }

    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    /* Staging buffers serve uploads (copy source) AND readbacks via
     * vkCmdCopyImageToBuffer (copy destination). */
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (p_vkCreateBuffer(s_dev, &bci, NULL, buf) != VK_SUCCESS) return 0;
    VkMemoryRequirements req; p_vkGetBufferMemoryRequirements(s_dev, *buf, &req);
    VkMemoryAllocateInfo mi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mi.allocationSize = req.size;
    mi.memoryTypeIndex = find_mem_type(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mi.memoryTypeIndex == UINT32_MAX || p_vkAllocateMemory(s_dev, &mi, NULL, mem) != VK_SUCCESS) {
        p_vkDestroyBuffer(s_dev, *buf, NULL); return 0;
    }
    p_vkBindBufferMemory(s_dev, *buf, *mem, 0);
    if (p_vkMapMemory(s_dev, *mem, 0, bytes, 0, map) != VK_SUCCESS) {
        p_vkDestroyBuffer(s_dev, *buf, NULL);
        p_vkFreeMemory(s_dev, *mem, NULL);
        return 0;
    }
    for (int i = 0; i < STAGING_CACHE_MAX; ++i) {
        if (!s_staging_cache[i].buf) {
            s_staging_cache[i].buf = *buf;
            s_staging_cache[i].mem = *mem;
            s_staging_cache[i].size = bytes;
            s_staging_cache[i].map = *map;
            s_staging_cache[i].busy = 1;
            break;
        }
    }
    s_perf_cur.allocs++;
    s_perf_cur.alloc_kb += (uint32_t)(bytes / 1024);
    return 1;
}
static void free_staging(VkBuffer buf, VkDeviceMemory mem) {
    staging_release(buf, mem);
}

/* hr -> raw mirror: pack the dirty rect via the compute pass (top-left sample of
 * each SxS block). Caller has already drained the batches that dirtied it. */
static void pack_flush(void) {
    if (!s_ready || !s_pack_dirty.set) return;
    s_perf_cur.pack_flushes++;
    int x = s_pack_dirty.x0, y = s_pack_dirty.y0;
    int w = s_pack_dirty.x1 - s_pack_dirty.x0 + 1;
    int h = s_pack_dirty.y1 - s_pack_dirty.y0 + 1;
    rect_clear(&s_pack_dirty);
    VkCommandBuffer cb = begin_oneshot();
    vram_to(cb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);     /* hr sampled */
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_GENERAL); /* raw storage */
    p_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipe_pack);
    p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s_pl_pack, 0, 1, &s_ds_pack, 0, NULL);
    PackPush pp = { s_scale, x, y };
    p_vkCmdPushConstants(cb, s_pl_pack, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pp, &pp);
    p_vkCmdDispatch(cb, (uint32_t)((w + 7) / 8), (uint32_t)((h + 7) / 8), 1);
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    end_oneshot(cb);
}

/* Ensure the raw mirror is current for a textured draw sampling the given page
 * / CLUT (mirrors the GL backend). */
static void flush_pack_if_sampling(int tpx, int tpy, int depth, int clx, int cly) {
    if (!s_pack_dirty.set) return;
    int page_w = depth == 0 ? 64 : depth == 1 ? 128 : 256;
    if (rect_intersects(&s_pack_dirty, tpx, tpy, tpx + page_w - 1, tpy + 255)) {
        flush_tex_batch(); pack_flush(); return;
    }
    if (depth <= 1) {
        int n = depth == 0 ? 16 : 256;
        if (rect_intersects(&s_pack_dirty, clx, cly, clx + n - 1, cly)) {
            flush_tex_batch(); pack_flush();
        }
    }
}

/* Blit a native [x,y,w,h] rect from src_img into the hr image (two STP passes,
 * bit15 -> stencil mirror). plain=1 (CPU upload: SET stencil, src already final)
 * or plain=0 (VRAM copy: honour mask set/check). src_div: 1 hr-res src, S native.
 * Marks the dest pack-dirty when mark_pack (copies re-derive raw from hr). */
static void blit_region(int x, int y, int w, int h, VkImageView src_view,
                        VkImage src_img, VkImageLayout *src_layout,
                        int src_div, int sox, int soy, int plain, int mark_pack) {
    if (w <= 0 || h <= 0) return;
    s_perf_cur.blits++;
    if (s_ds_blit_idx >= BLIT_DESC_RING) gpu_sync();   /* ring exhausted: drain, reset idx */
    VkDescriptorSet ds = s_ds_blit_ring[s_ds_blit_idx++];
    VkDescriptorImageInfo ii = { s_samp, src_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wr.dstSet = ds; wr.dstBinding = 0; wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr.pImageInfo = &ii;
    p_vkUpdateDescriptorSets(s_dev, 1, &wr, 0, NULL);

    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, src_img, src_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vram_to(cb, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_geo_pass(cb);
    set_vp_full(cb); set_scissor_px(cb, x, y, w, h);
    p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl_blit, 0, 1, &ds, 0, NULL);
    BlitPush bp = {0};
    bp.shift = px_shift(); bp.maskset = plain ? 0 : s_mask_set; bp.src_div = src_div;
    bp.src_off[0] = sox; bp.src_off[1] = soy;
    bp.rect[0] = x; bp.rect[1] = y; bp.rect[2] = x + w; bp.rect[3] = y + h;
    int chk = plain ? 0 : s_mask_check;
    /* pass 1: bit15=0 texels, write stencil 0 (or honour mask set) */
    bind_masked(cb, 2, 0, 0, chk, plain ? 0 : s_mask_set);
    bp.stp_pass = 1;
    p_vkCmdPushConstants(cb, s_pl_blit, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof bp, &bp);
    p_vkCmdDraw(cb, 6, 1, 0, 0);
    /* pass 2: bit15=1 texels, write stencil 1 */
    bind_masked(cb, 2, 0, 0, chk, 1);
    bp.stp_pass = 2;
    p_vkCmdPushConstants(cb, s_pl_blit, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof bp, &bp);
    p_vkCmdDraw(cb, 6, 1, 0, 0);
    p_vkCmdEndRenderPass(cb);
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    end_oneshot(cb);
    if (mark_pack) rect_add(&s_pack_dirty, x, y, x + w - 1, y + h - 1);
    s_gpu_dirty = 1;
}

/* ---- VRAM transfers (CPU <-> GPU image) -------------------------------- */
/* Upload a w x h block of 1555 pixels at (x,y). The raw mirror takes the exact
 * 1555 bits directly (CLUT indices must survive), and the hr render target is
 * updated via a blit so the stencil bit15 mirror stays exact. */
static void vram_upload_block(int x, int y, int w, int h, const uint16_t *data) {
    if (w <= 0 || h <= 0 || !s_ready) return;
    s_perf_cur.upload_blocks++;
    flush_tex_batch(); flush_geometry();

    /* 1) raw mirror: native 1555, exact. */
    VkBuffer rbuf; VkDeviceMemory rmem; void *rmap;
    if (make_staging((VkDeviceSize)w * h * 2, &rbuf, &rmem, &rmap)) {
        memcpy(rmap, data, (size_t)w * h * 2);
        VkCommandBuffer cb = begin_oneshot();
        img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy rc = {0};
        rc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; rc.imageSubresource.layerCount = 1;
        rc.imageOffset.x = x; rc.imageOffset.y = y;
        rc.imageExtent.width = w; rc.imageExtent.height = h; rc.imageExtent.depth = 1;
        p_vkCmdCopyBufferToImage(cb, rbuf, s_raw_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rc);
        img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        end_oneshot(cb);
        defer_staging(rbuf, rmem);   /* copy in flight; freed at next gpu_sync */
    }

    /* 2) native RGBA8 staging -> upload image. */
    VkBuffer ubuf; VkDeviceMemory umem; void *umap;
    if (make_staging((VkDeviceSize)w * h * 4, &ubuf, &umem, &umap)) {
        uint8_t *m = (uint8_t*)umap;
        for (int i = 0; i < w * h; i++) rgb555_to_rgba8(data[i], m + (size_t)i * 4);
        VkCommandBuffer cb = begin_oneshot();
        img_to(cb, s_up_img, &s_up_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy uc = {0};
        uc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; uc.imageSubresource.layerCount = 1;
        uc.imageOffset.x = x; uc.imageOffset.y = y;
        uc.imageExtent.width = w; uc.imageExtent.height = h; uc.imageExtent.depth = 1;
        p_vkCmdCopyBufferToImage(cb, ubuf, s_up_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &uc);
        end_oneshot(cb);
        defer_staging(ubuf, umem);   /* copy in flight; freed at next gpu_sync */
        /* 3) blit upload image -> hr (native src: src_div=S, off 0). */
        blit_region(x, y, w, h, s_up_view, s_up_img, &s_up_layout, s_scale, 0, 0, 1, 0);
    }
}

/* CPU -> GPU: flush accumulated CPU-side VRAM writes as ONE upload (see
 * s_up_rects). vram_upload_block drains the draw batches first, so PS1
 * submission order is preserved: every draw appended before these writes has
 * landed by the time the upload writes VRAM, and every GPU-op entry point
 * calls this before touching VRAM so the writes land before later ops. */
static void flush_cpu_upload(void) {
    if (s_up_nrects == 0 || !s_ready || !s_vram) return;
    /* Snapshot + clear first (re-entrancy safe: up_add overflow calls back). */
    DirtyRect rects[UP_RECTS_MAX];
    int nrects = s_up_nrects;
    memcpy(rects, s_up_rects, (size_t)nrects * sizeof(DirtyRect));
    s_up_nrects = 0;

    flush_tex_batch(); flush_geometry();   /* PS1 order: queued draws land first */
    s_perf_cur.upload_blocks++;

    /* COALESCED upload: exactly TWO queue submits per flush regardless of rect
     * count (one for the buffer->image copies, one render pass for the hr
     * blits). The naive per-rect vram_upload_block/blit_region loop paid
     * 2 staging allocs + 3 submits PER RECT — with the exact-rect list feeding
     * MDEC macroblock streams that re-created the boot-wedge submit churn the
     * batched-upload fix existed to kill (measured 0.7 fps at the Tomba2
     * Whoopee logo). Rect data is packed back-to-back in ONE 1555 staging +
     * ONE RGBA8 staging, addressed via VkBufferImageCopy bufferOffset. */
    size_t total = 0;
    for (int i = 0; i < nrects; i++)
        total += vk_upload_aligned_texels(
            (size_t)(rects[i].x1 - rects[i].x0 + 1) *
            (size_t)(rects[i].y1 - rects[i].y0 + 1));
    if (total == 0) return;

    VkBuffer rbuf; VkDeviceMemory rmem; void *rmap;
    if (!make_staging((VkDeviceSize)total * 2, &rbuf, &rmem, &rmap)) return;
    VkBuffer ubuf; VkDeviceMemory umem; void *umap;
    if (!make_staging((VkDeviceSize)total * 4, &ubuf, &umem, &umap)) {
        defer_staging(rbuf, rmem);
        return;
    }

    VkBufferImageCopy rcopies[UP_RECTS_MAX], ucopies[UP_RECTS_MAX];
    memset(rcopies, 0, sizeof rcopies); memset(ucopies, 0, sizeof ucopies);
    size_t texoff = 0;
    uint16_t *rdst = (uint16_t *)rmap;
    uint8_t  *udst = (uint8_t *)umap;
    for (int i = 0; i < nrects; i++) {
        int x = rects[i].x0, y = rects[i].y0;
        int w = rects[i].x1 - rects[i].x0 + 1;
        int h = rects[i].y1 - rects[i].y0 + 1;
        for (int row = 0; row < h; row++) {
            const uint16_t *src = s_vram + (size_t)(y + row) * VRAM_W + x;
            memcpy(rdst + texoff + (size_t)row * w, src, (size_t)w * 2);
            uint8_t *m = udst + (texoff + (size_t)row * w) * 4;
            for (int col = 0; col < w; col++)
                rgb555_to_rgba8(src[col], m + (size_t)col * 4);
        }
        rcopies[i].bufferOffset = (VkDeviceSize)texoff * 2;
        rcopies[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rcopies[i].imageSubresource.layerCount = 1;
        rcopies[i].imageOffset.x = x; rcopies[i].imageOffset.y = y;
        rcopies[i].imageExtent.width = w; rcopies[i].imageExtent.height = h;
        rcopies[i].imageExtent.depth = 1;
        ucopies[i] = rcopies[i];
        ucopies[i].bufferOffset = (VkDeviceSize)texoff * 4;
        /* Padding prevents an odd-sized R16 rect from leaving all following
         * VkBufferImageCopy offsets at two modulo four bytes. */
        texoff += vk_upload_aligned_texels((size_t)w * h);
    }

    /* Submit 1: both image copy sets. */
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    p_vkCmdCopyBufferToImage(cb, rbuf, s_raw_img,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             (uint32_t)nrects, rcopies);
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    img_to(cb, s_up_img, &s_up_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    p_vkCmdCopyBufferToImage(cb, ubuf, s_up_img,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             (uint32_t)nrects, ucopies);
    end_oneshot(cb);
    defer_staging(rbuf, rmem);
    defer_staging(ubuf, umem);

    /* Submit 2: one render pass, N scissored 2-draw blits upload-image -> hr
     * (plain blit semantics, matching vram_upload_block's blit_region call:
     * mask writes off, no pack mark — uploaded content IS the CPU data). */
    if (s_ds_blit_idx >= BLIT_DESC_RING) gpu_sync();
    VkDescriptorSet ds = s_ds_blit_ring[s_ds_blit_idx++];
    VkDescriptorImageInfo ii = { s_samp, s_up_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet wr = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    wr.dstSet = ds; wr.dstBinding = 0; wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr.pImageInfo = &ii;
    p_vkUpdateDescriptorSets(s_dev, 1, &wr, 0, NULL);
    cb = begin_oneshot();
    img_to(cb, s_up_img, &s_up_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vram_to(cb, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_geo_pass(cb);
    set_vp_full(cb);
    p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl_blit, 0, 1, &ds, 0, NULL);
    for (int i = 0; i < nrects; i++) {
        int x = rects[i].x0, y = rects[i].y0;
        int w = rects[i].x1 - rects[i].x0 + 1;
        int h = rects[i].y1 - rects[i].y0 + 1;
        set_scissor_px(cb, x, y, w, h);
        s_perf_cur.blits++;
        BlitPush bp = {0};
        bp.shift = px_shift(); bp.maskset = 0; bp.src_div = s_scale;
        bp.src_off[0] = 0; bp.src_off[1] = 0;
        bp.rect[0] = x; bp.rect[1] = y; bp.rect[2] = x + w; bp.rect[3] = y + h;
        bind_masked(cb, 2, 0, 0, 0, 0);
        bp.stp_pass = 1;
        p_vkCmdPushConstants(cb, s_pl_blit, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof bp, &bp);
        p_vkCmdDraw(cb, 6, 1, 0, 0);
        bind_masked(cb, 2, 0, 0, 0, 1);
        bp.stp_pass = 2;
        p_vkCmdPushConstants(cb, s_pl_blit, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof bp, &bp);
        p_vkCmdDraw(cb, 6, 1, 0, 0);
    }
    p_vkCmdEndRenderPass(cb);
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    end_oneshot(cb);
    s_gpu_dirty = 1;
}

/* GPU -> CPU mirror: drain batches, pack, copy the whole raw mirror down. */
static void ensure_cpu(void) {
    extern int psx_netplay_active(void);
    if (!s_ready || !s_gpu_dirty || !s_vram) return;
    if (psx_netplay_active()) {
        s_gpu_dirty = 0;
        return;
    }
    flush_cpu_upload();   /* readback overwrites s_vram: pending writes land first */
    flush_tex_batch(); flush_geometry();
    /* Readback must reflect ALL current hr content, not just the incremental
     * s_pack_dirty rect. pack_flush() packs only that rect and early-returns if
     * it is unset — so draws that updated hr (and present) but never marked the
     * pack rect (e.g. the BIOS shell after the logo) left the raw mirror stale,
     * and screenshots/guest VRAM reads returned an old frame (the logo) while
     * the live window showed the shell. Force a full-frame pack here; this path
     * is readback-only (NOT the per-frame present path), so the cost is paid
     * only on screenshots / guest VRAM reads, never every frame. */
    rect_add(&s_pack_dirty, 0, 0, VRAM_W - 1, VRAM_H - 1);
    pack_flush();
    VkBuffer buf; VkDeviceMemory mem; void *map;
    if (!make_staging((VkDeviceSize)VRAM_W * VRAM_H * 2, &buf, &mem, &map)) return;
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy rc = {0};
    rc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; rc.imageSubresource.layerCount = 1;
    rc.imageExtent.width = VRAM_W; rc.imageExtent.height = VRAM_H; rc.imageExtent.depth = 1;
    p_vkCmdCopyImageToBuffer(cb, s_raw_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &rc);
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    end_oneshot(cb);
    gpu_sync();   /* the readback result must be complete before we read it */
    memcpy(s_vram, map, (size_t)VRAM_W * VRAM_H * 2);
    free_staging(buf, mem);
    s_gpu_dirty = 0;
}

/* ---- backend vtable ---------------------------------------------------- */
/* Also initialize the software renderer on the SAME VRAM array so a failed
 * Vulkan context creation can fall back to software cleanly (mirrors how the GL
 * backend's init calls sw_renderer_init). When Vulkan is active the SW mirror
 * is just the CPU readback target. */
static void vkb_init(uint16_t *vram) { s_vram = vram; s_cpu_dirty = 0; sw_renderer_init(vram); }
static void vkb_set_scale(int scale) {
    /* The settings panel's Resolution row wins over the session's scale (it is
     * read once per run, so the value never changes under live images). */
    /* MSAA multiplies the widescreen surfaces' memory by the sample count. */
    const int cap = touch_ui_msaa() >= 4 ? 4 : touch_ui_msaa() == 2 ? 5 : 8;
    if (scale > cap) scale = cap;
    const int user = touch_ui_resolution() > cap ? cap : touch_ui_resolution();
    if (user >= 1 && user <= 8) {
        if (s_scale != user) fprintf(stdout, "psxrecomp: vulkan: internal resolution %dx from the settings panel\n", user);
        s_scale = user;
        return;
    }
    if (scale >= 1 && scale <= 4) s_scale = scale;
}
static int  vkb_scale(void) { return s_scale; }
static void vkb_set_texture_filter(int b) { s_texfilter = b ? 1 : 0; }
static int  vkb_texture_filter(void) { return s_texfilter; }

static void vkb_set_draw_area(int x1, int y1, int x2, int y2) {
    /* Scissor is per-batch (one render pass uses one scissor), so drain both
     * independent geometry batches before the clip rect changes. Otherwise
     * earlier textured GP0 primitives render with the next draw area's scissor. */
    if (x1 != s_da_x1 || y1 != s_da_y1 || x2 != s_da_x2 || y2 != s_da_y2) {
        flush_tex_batch();
        flush_geometry();
    }
    s_da_x1 = x1; s_da_y1 = y1; s_da_x2 = x2; s_da_y2 = y2;
}
static void vkb_get_draw_area(int *x1, int *y1, int *x2, int *y2) {
    if (x1) *x1 = s_da_x1; if (y1) *y1 = s_da_y1; if (x2) *x2 = s_da_x2; if (y2) *y2 = s_da_y2;
}
static void vkb_set_draw_offset(int x, int y) { s_off_x = x; s_off_y = y; }

/* GP0 02h fill: set a VRAM rect to a solid 1555 colour. The write lands in
 * the CPU array immediately (like every CPU-side VRAM write) and rides the
 * batched s_up_rects upload — the old per-fill staging upload was part of
 * the per-op alloc/submit churn. */
static void vkb_fill_rect(int x, int y, int w, int h, uint16_t color) {
    sw_fill_rect(x, y, w, h, color);
    if (!s_ctx_ok) return;
    if (x < 0) { w += x; x = 0; } if (y < 0) { h += y; y = 0; }
    if (x + w > VRAM_W) w = VRAM_W - x; if (y + h > VRAM_H) h = VRAM_H - y;
    if (w <= 0 || h <= 0) return;
    up_add(x, y, x + w - 1, y + h - 1);
}

/* Same policy as GL: skip framebuffer-sized depth24 (RGB888) transfers only;
 * still stage smaller texture A0s. On leave clear skipped FB union — never
 * restage packed RGB888 as 1555. */
static int s_depth24_skip_up = 0;
static DirtyRect s_d24_skip_fb;

static int depth24_is_fb_transfer(int x, int y, int w, int h) {
    if (!gpu_display_is_depth24() || w <= 0 || h <= 0) return 0;
    GpuDisplayInfo di;
    gpu_get_display_info(&di);
    int fb_w = (int)((di.width * 3u + 1u) / 2u);
    int fb_h = (int)di.height;
    if (fb_w < 8) fb_w = 8;
    if (fb_h < 1) fb_h = 1;
    /* See GL: 256×256 pages must not match the half-FB area heuristic. */
    if (w <= 256 && h <= 256) return 0;
    {
        int dx = (int)(di.display_x & 1023u);
        int dy = (int)(di.display_y & 511u);
        int x0 = x & (VRAM_W - 1), y0 = y & (VRAM_H - 1);
        if (x0 + w <= dx || x0 >= dx + fb_w || y0 + h <= dy || y0 >= dy + fb_h)
            return 0;
    }
    if (h >= fb_h - 8 && h <= fb_h + 16 && w >= (fb_w * 3) / 4) return 1;
    if ((int64_t)w * (int64_t)h >= ((int64_t)fb_w * fb_h) / 2) return 1;
    return 0;
}

static void depth24_mark_scanout_band(void) {
    GpuDisplayInfo di;
    int fb_w, fb_h, x0, y0, x1, y1;
    if (!gpu_display_is_depth24()) return;
    gpu_get_display_info(&di);
    fb_w = (int)((di.width * 3u + 1u) / 2u);
    fb_h = (int)di.height;
    if (fb_w < 8) fb_w = 8;
    if (fb_h < 1) fb_h = 1;
    x0 = (int)(di.display_x & 1023u);
    y0 = (int)(di.display_y & 511u);
    x1 = x0 + fb_w - 1;
    y1 = y0 + fb_h - 1;
    if (x1 > VRAM_W - 1) x1 = VRAM_W - 1;
    if (y1 > VRAM_H - 1) y1 = VRAM_H - 1;
    rect_add(&s_d24_skip_fb, x0, y0, x1, y1);
}

static void depth24_clear_skipped_fb(void) {
    /* GL scissor-clears the skipped FB union. VK keeps texture uploads that
     * landed during depth24; the FB bands are overwritten by the next 15-bit
     * draws. (A full-image ClearColorImage would wipe those textures.) */
    rect_clear(&s_d24_skip_fb);
}

static void depth24_upload_policy(void) {
    int d24 = gpu_display_is_depth24();
    if (d24 && !s_depth24_skip_up) {
        s_up_nrects = 0;
        rect_clear(&s_d24_skip_fb);
    } else if (!d24 && s_depth24_skip_up) {
        s_up_nrects = 0;
        depth24_clear_skipped_fb();
        gpu_depth24_upload_span_reset();
    }
    s_depth24_skip_up = d24;
}

static void vkb_vram_transfer_in(int x, int y, int w, int h, const uint16_t *data) {
    sw_vram_transfer_in(x, y, w, h, data);
    if (!s_ctx_ok) return;
    depth24_upload_policy();
    if (s_depth24_skip_up && depth24_is_fb_transfer(x, y, w, h)) {
        /* Full-VRAM savestate restore: stage FBO; mark scanout band only. */
        if (w >= VRAM_W && h >= VRAM_H) {
            up_add_transfer(x, y, w, h);
            rect_clear(&s_d24_skip_fb);
            depth24_mark_scanout_band();
            return;
        }
        depth24_mark_scanout_band();
        return;
    }
    up_add_transfer(x, y, w, h);   /* exact touched rects, incl. per-pixel wrap */
}

void vk_renderer_restage_vram_after_savestate(void) {
    if (!s_ctx_ok || !s_vram) return;
    s_up_nrects = 0;
    rect_clear(&s_d24_skip_fb);
    s_depth24_skip_up = 0;
    up_add_transfer(0, 0, VRAM_W, VRAM_H);
    flush_cpu_upload();
    if (gpu_display_is_depth24()) {
        s_depth24_skip_up = 1;
        depth24_mark_scanout_band();
    }
}
static void vkb_vram_transfer_out(int x, int y, int w, int h, uint16_t *data) {
    ensure_cpu();   /* sync GPU-rendered content down to the CPU mirror first */
    for (int row = 0; row < h; row++)
        for (int col = 0; col < w; col++)
            data[row * w + col] = s_vram ? s_vram[(y + row) * VRAM_W + (x + col)] : 0;
}
static void vkb_vram_write(int x, int y, uint16_t pixel) {
    sw_vram_write(x, y, pixel);
    if (!s_ctx_ok) return;
    depth24_upload_policy();
    up_add(x & (VRAM_W - 1), y & (VRAM_H - 1),
           x & (VRAM_W - 1), y & (VRAM_H - 1));
}
static uint16_t vkb_vram_read(int x, int y) {
    ensure_cpu();   /* gated by s_gpu_dirty; cheap when nothing was drawn */
    return s_vram ? s_vram[y * VRAM_W + x] : 0;
}

/* render_display: CPU readout path (screenshots / debug server / the 24-bit
 * FMV present). Sync the GPU-authoritative VRAM down, then read out via the
 * software renderer on the shared array — mirrors the GL backend exactly. */
static int vkb_render_display(uint32_t *o, int p, int dx, int dy, int dw, int dh) {
    ensure_cpu();
    return sw_render_display(o, p, dx, dy, dw, dh);
}
static int vkb_render_display_hires(uint32_t *o, int p, int dx, int dy, int dw, int dh) {
    ensure_cpu();
    return sw_render_display_hires(o, p, dx, dy, dw, dh);
}

/* ---- geometry batch (Phase 2) ------------------------------------------ */
/* Set the full-image viewport and the draw-area scissor (in framebuffer px). */
static void set_vp_scissor(VkCommandBuffer cb) {
    int S = s_scale;
    VkViewport vp = { 0.f, 0.f, (float)(VRAM_W * S), (float)(VRAM_H * S), 0.f, 1.f };
    p_vkCmdSetViewport(cb, 0, 1, &vp);
    int x1 = s_da_x1 < 0 ? 0 : s_da_x1, y1 = s_da_y1 < 0 ? 0 : s_da_y1;
    int x2 = s_da_x2 >= VRAM_W ? VRAM_W - 1 : s_da_x2;
    int y2 = s_da_y2 >= VRAM_H ? VRAM_H - 1 : s_da_y2;
    int w = x2 - x1 + 1, h = y2 - y1 + 1;
    if (w < 0) w = 0; if (h < 0) h = 0;
    VkRect2D sc = { { x1 * S, y1 * S }, { (uint32_t)(w * S), (uint32_t)(h * S) } };
    p_vkCmdSetScissor(cb, 0, 1, &sc);
}

/* ---- native-wide compositor helpers ------------------------------------- */
/* X-translation (native px) from canonical VRAM space into the active wide
 * surface: local_x = vram_x - base_x + OFFSET. Same as GL/SW wide_dx(). */
static int view_enabled, view_shift, view_pad_left, view_pad_right;
static int wide_dx(void) { return s_wide_offset + view_shift - s_wide_cur_base; }

/* 2D-backdrop stretch (ported from the GL/SW renderers). The game's sky and
 * water backdrops bypass the GTE, so in native-wide they stay 4:3 wide and
 * leave black margins. Gated per primitive exactly like GL's bd_prim_gate; the
 * gate is a batch key, and the wide mirror of a gated batch is scaled about
 * the 4:3 centre so it fills the wide frame. Canonical VRAM is untouched. */
extern int psx_ws_prim_in_backdrop(void);
extern int gpu_ws_nw_flat_backdrop_enabled(void);
static int s_fg_bd_bits;             /* gate bits of the command being replayed */
static int s_geo_gate, s_tb_gate;    /* gate of the open untextured / textured batch */
static int vk_bd_bits(void) {
    if (s_fg_replay) return s_fg_bd_bits;
    if (s_wide_cur < 0 || s_wide_w <= 0) return 0;
    int b = 0;
    if (psx_ws_prim_in_backdrop() == 1) b |= 1;
    if (gpu_ws_nw_flat_backdrop_enabled()) b |= 2;
    return b;
}
static int vk_bd_gate(int textured, int is_line) {
    if (s_wide_cur < 0 || s_wide_w <= 0) return 0;
    if (!textured && is_line) return 0;      /* ropes/wires belong to the 3D scene */
    const int b = vk_bd_bits();
    if (!textured && (b & 2)) return 1;
    return b & 1;
}
/* Vertex transform of a wide-mirror draw: clip_x = (x + xoff) / xhalf - 1. */
static void wide_xform(int gate, float *xoff, float *xhalf) {
    const float half = (float)s_wide_w / 2.0f;
    *xoff = (float)wide_dx(); *xhalf = half;
    const int native_w = s_wide_w - 2 * s_wide_offset;
    if (!gate || native_w <= 0 || native_w >= s_wide_w) return;
    const float k = (float)s_wide_w / (float)native_w;
    const float centre = (float)s_wide_cur_base + (float)native_w / 2.0f;
    *xhalf = half / k;
    *xoff = *xhalf * (1.0f + (float)view_shift / half) - centre;
}

static void wide_free_all(void) {
    int any = 0;
    for (int i = 0; i < VK_WIDE_MAX_SURF; i++) any |= (s_wide_img[i] != VK_NULL_HANDLE);
    /* Queued mirror draws reference these surfaces, and so do clears and copies
     * already recorded in the shared command buffer, which is not submitted
     * until the next sync: destroying the images under it crashed that submit
     * (a display-width change at a door or scene switch). Submit it and wait. */
    pass_close();
    if (any) { s_work_pending = 1; vk_gpu_sync_internal(); }
    for (int i = 0; i < VK_WIDE_MAX_SURF; i++) {
        if (s_wide_fb[i])      { p_vkDestroyFramebuffer(s_dev, s_wide_fb[i], NULL); s_wide_fb[i] = VK_NULL_HANDLE; }
        if (s_wide_view[i])    { p_vkDestroyImageView(s_dev, s_wide_view[i], NULL); s_wide_view[i] = VK_NULL_HANDLE; }
        if (s_wide_img[i])     { p_vkDestroyImage(s_dev, s_wide_img[i], NULL); s_wide_img[i] = VK_NULL_HANDLE; }
        if (s_wide_ms_view[i]) { p_vkDestroyImageView(s_dev, s_wide_ms_view[i], NULL); s_wide_ms_view[i] = VK_NULL_HANDLE; }
        if (s_wide_ms_img[i])  { p_vkDestroyImage(s_dev, s_wide_ms_img[i], NULL); s_wide_ms_img[i] = VK_NULL_HANDLE; }
        if (s_wide_ms_mem[i])  { p_vkFreeMemory(s_dev, s_wide_ms_mem[i], NULL); s_wide_ms_mem[i] = VK_NULL_HANDLE; }
        if (s_wide_mem[i])     { p_vkFreeMemory(s_dev, s_wide_mem[i], NULL); s_wide_mem[i] = VK_NULL_HANDLE; }
        if (s_wide_ds_view[i]) { p_vkDestroyImageView(s_dev, s_wide_ds_view[i], NULL); s_wide_ds_view[i] = VK_NULL_HANDLE; }
        if (s_wide_ds_img[i])  { p_vkDestroyImage(s_dev, s_wide_ds_img[i], NULL); s_wide_ds_img[i] = VK_NULL_HANDLE; }
        if (s_wide_ds_mem[i])  { p_vkFreeMemory(s_dev, s_wide_ds_mem[i], NULL); s_wide_ds_mem[i] = VK_NULL_HANDLE; }
        s_wide_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        s_wide_base[i] = -1;
    }
    s_wide_cur = -1;
    s_fg_surf = -1; s_fg_hold = 0; s_fg_n = 0;
    s_fg_surfs[0] = s_fg_surfs[1] = s_fg_surfs[2] = -1;
    s_fgq_n = s_fgq_i = 0;
}

/* Find (or lazily create) the wide surface for base_x. Returns the surface
 * index or -1. A new surface = RGBA8 color image (attachment + blit src/dst)
 * + its own stencil image + a framebuffer on the SHARED render pass; color is
 * cleared black, stencil to 0, then both are parked in the layouts the render
 * pass expects (color/DS ATTACHMENT_OPTIMAL). */
static int wide_surf_for(int base_x) {
    if (s_wide_w <= 0 || !s_ready) return -1;
    if (s_fg_replay && s_fg_surf >= 0) return s_fg_surf;   /* generated picture */
    for (int i = 0; i < VK_WIDE_MAX_SURF; i++)
        if (s_wide_img[i] && s_wide_base[i] == base_x) return i;
    for (int i = 0; i < VK_WIDE_MAX_SURF; i++) {
        if (s_wide_img[i]) continue;
        int S = s_scale, w = s_wide_w * S, h = VRAM_H * S;
        if (!make_image(VK_FORMAT_R8G8B8A8_UNORM, w, h,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, &s_wide_img[i], &s_wide_mem[i], &s_wide_view[i]))
            return -1;
        const int msaa = s_msaa > 1;
        s_make_samples = s_msaa;      /* stencil (and MS colour) match the pass */
        int ok = make_image(s_ds_format, w, h,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        DS_ASPECT, &s_wide_ds_img[i], &s_wide_ds_mem[i], &s_wide_ds_view[i]);
        if (ok && msaa)
            ok = make_image(VK_FORMAT_R8G8B8A8_UNORM, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, &s_wide_ms_img[i], &s_wide_ms_mem[i], &s_wide_ms_view[i]);
        s_make_samples = 1;
        if (!ok) return -1;
        VkImageView views[3] = { msaa ? s_wide_ms_view[i] : s_wide_view[i], s_wide_ds_view[i], s_wide_view[i] };
        VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fi.renderPass = msaa ? s_rpass_ms : s_rpass; fi.attachmentCount = msaa ? 3 : 2; fi.pAttachments = views;
        fi.width = (uint32_t)w; fi.height = (uint32_t)h; fi.layers = 1;
        if (p_vkCreateFramebuffer(s_dev, &fi, NULL, &s_wide_fb[i]) != VK_SUCCESS)
            return -1;
        /* Clear + park in attachment layouts (mirrors the s_ds init chain). */
        VkCommandBuffer cb = begin_oneshot();
        s_wide_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
        img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black = {{0, 0, 0, 0}};
        VkImageSubresourceRange crng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        p_vkCmdClearColorImage(cb, s_wide_img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &crng);
        img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkImageMemoryBarrier db = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        db.image = s_wide_ds_img[i];
        db.subresourceRange.aspectMask = DS_ASPECT;
        db.subresourceRange.levelCount = 1; db.subresourceRange.layerCount = 1;
        if (!msaa) {
            db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; db.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            db.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, 0, NULL, 0, NULL, 1, &db);
            VkClearDepthStencilValue dsv = { 0.0f, 0 };
            VkImageSubresourceRange srng = { DS_ASPECT, 0, 1, 0, 1 };
            p_vkCmdClearDepthStencilImage(cb, s_wide_ds_img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &dsv, 1, &srng);
            db.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            db.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, NULL, 0, NULL, 1, &db);
        } else {
            /* Multisampled images: park both in their attachment layouts, then
             * clear them inside the MS pass (which also resolves the black). */
            db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, NULL, 0, NULL, 1, &db);
            VkImageLayout ml = VK_IMAGE_LAYOUT_UNDEFINED;
            img_to(cb, s_wide_ms_img[i], &ml, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkRenderPassBeginInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            rp.renderPass = s_rpass_ms; rp.framebuffer = s_wide_fb[i];
            rp.renderArea.extent.width = (uint32_t)w; rp.renderArea.extent.height = (uint32_t)h;
            p_vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
            VkClearAttachment ca[2] = {0};
            ca[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ca[1].aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            VkClearRect cr = { { { 0, 0 }, { (uint32_t)w, (uint32_t)h } }, 0, 1 };
            p_vkCmdClearAttachments(cb, 2, ca, 1, &cr);
            p_vkCmdEndRenderPass(cb);
        }
        end_oneshot(cb);
        s_wide_base[i] = base_x;
        return i;
    }
    return -1;  /* more distinct buffers than VK_WIDE_MAX_SURF — shouldn't happen */
}

/* Begin the mirror render pass on the active wide surface inside an already-
 * recording one-shot CB (canonical pass already ended). Sets the wide viewport
 * and the mirror scissor: FULL width X (the margins are the point) but the
 * DRAW-AREA Y band — native-wide only widens X; a full-height Y let draws bleed
 * across a vertical double-buffer band boundary (the GL band-flicker lesson). */
static void wide_pass_begin_surf(VkCommandBuffer cb, int i, int wide_w) {
    int S = s_scale;
    s_perf_cur.wide_passes++;
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    /* Stencil self-barrier for the WIDE DS image (same hazard class as
     * begin_geo_pass covers for the canonical DS across one-shot submits). */
    VkImageMemoryBarrier db = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    db.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    db.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    db.srcQueueFamilyIndex = db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    db.image = s_wide_ds_img[i];
    db.subresourceRange.aspectMask = DS_ASPECT;
    db.subresourceRange.levelCount = 1; db.subresourceRange.layerCount = 1;
    db.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    db.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        0, 0, NULL, 0, NULL, 1, &db);
    color_self_barrier(cb, s_wide_img[i]);
    if (s_msaa > 1) color_self_barrier(cb, s_wide_ms_img[i]);
    s_pipe_ms = s_msaa > 1;
    VkRenderPassBeginInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rp.renderPass = s_msaa > 1 ? s_rpass_ms : s_rpass; rp.framebuffer = s_wide_fb[i];
    rp.renderArea.extent.width = (uint32_t)(wide_w * S);
    rp.renderArea.extent.height = (uint32_t)(VRAM_H * S);
    p_vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp = { 0.f, 0.f, (float)(wide_w * S), (float)(VRAM_H * S), 0.f, 1.f };
    p_vkCmdSetViewport(cb, 0, 1, &vp);
}
static void wide_scissor(VkCommandBuffer cb, int wide_w, int da_y1, int da_y2, int pad_l, int pad_r) {
    int S = s_scale;
    int y1 = da_y1 < 0 ? 0 : da_y1;
    int y2 = da_y2 >= VRAM_H ? VRAM_H - 1 : da_y2;
    int hh = y2 - y1 + 1; if (hh < 0) hh = 0;
    VkRect2D sc = { { pad_l * S, y1 * S },
        { (uint32_t)((wide_w - pad_l - pad_r) * S), (uint32_t)(hh * S) } };
    p_vkCmdSetScissor(cb, 0, 1, &sc);
}
static void wide_pass_begin(VkCommandBuffer cb) {
    wide_pass_begin_surf(cb, s_wide_cur, s_wide_w);
    wide_scissor(cb, s_wide_w, s_da_y1, s_da_y2, view_pad_left, view_pad_right);
}

/* ---- pass coalescing (see begin_oneshot) -------------------------------- */
static int s_pass_open;          /* canonical VRAM pass open in the work CB */
typedef struct {
    uint8_t  tex, check, mask;
    int8_t   semi;               /* -1 opaque, else PSX blend mode */
    int16_t  surf, y1, y2, pad_l, pad_r;
    int      wide_w;
    uint32_t base, n;
    GeoPush  gp;
    TexPush  tp;
    VkDescriptorSet hd_ds;
    uint8_t stencil_fix;
} WideRec;
#define WQ_MAX 4096
static WideRec s_wq[WQ_MAX];
static int s_wq_n;

static void canon_pass_open(VkCommandBuffer cb) {
    if (s_pass_open) return;
    vram_to(cb, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_geo_pass(cb);
    s_pass_open = 1;
}
static WideRec *wq_push(int tex, int semi, int check, int mask, uint32_t base, uint32_t n) {
    if (s_wq_n >= WQ_MAX) pass_close();
    WideRec *r = &s_wq[s_wq_n++];
    r->tex = (uint8_t)tex; r->semi = (int8_t)semi; r->check = (uint8_t)check; r->mask = (uint8_t)mask;
    r->surf = (int16_t)s_wide_cur; r->wide_w = s_wide_w;
    r->y1 = (int16_t)s_da_y1; r->y2 = (int16_t)s_da_y2;
    r->pad_l = (int16_t)view_pad_left; r->pad_r = (int16_t)view_pad_right;
    r->base = base; r->n = n;
    return r;
}
static void pass_close(void) {
    if (!s_pass_open && !s_wq_n) return;
    VkCommandBuffer cb = s_work_cb;
    if (!cb) { s_pass_open = 0; s_wq_n = 0; return; }   /* cannot happen */
    const uint64_t t_in = SDL_GetPerformanceCounter();
    if (s_pass_open) {
        p_vkCmdEndRenderPass(cb);
        s_pass_open = 0;
        vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
    int cur = -1, n = s_wq_n;
    s_wq_n = 0;
    const VkShaderStageFlags TS = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDeviceSize off = 0;
    for (int k = 0; k < n; k++) {
        WideRec *r = &s_wq[k];
        if (r->surf < 0 || s_wide_img[r->surf] == VK_NULL_HANDLE) continue;
        if (r->surf != cur) {
            if (cur >= 0) p_vkCmdEndRenderPass(cb);
            cur = r->surf;
            wide_pass_begin_surf(cb, cur, r->wide_w);
        }
        wide_scissor(cb, r->wide_w, r->y1, r->y2, r->pad_l, r->pad_r);
        int blend = (r->semi < 0) ? 0 : (r->semi + 1);
        if (!r->tex) {
            p_vkCmdBindVertexBuffers(cb, 0, 1, &s_vbuf, &off);
            bind_masked(cb, 0, 0, blend, r->check, r->mask);
            p_vkCmdPushConstants(cb, s_pl_geo, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof r->gp, &r->gp);
            p_vkCmdDraw(cb, r->n, 1, r->base, 0);
            continue;
        }
        {
            VkDescriptorSet sets[2] = { s_ds_tex, r->hd_ds ? r->hd_ds : s_hd_none_ds };
            p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl_tex, 0, 2, sets, 0, NULL);
        }
        p_vkCmdBindVertexBuffers(cb, 0, 1, &s_tbuf, &off);
        TexPush tp = r->tp;
        if (r->semi < 0) {
            bind_masked(cb, 1, 0, 0, r->check, r->mask);
            tp.semipass = 0;
            p_vkCmdPushConstants(cb, s_pl_tex, TS, 0, sizeof tp, &tp);
            p_vkCmdDraw(cb, r->n, 1, r->base, 0);
            if (!r->stencil_fix) continue;
            bind_masked_stencil_only(cb, 1, 0, r->check, 1);
        } else {
            bind_masked(cb, 1, 0, 0, r->check, r->mask);
            tp.semipass = 1;
            p_vkCmdPushConstants(cb, s_pl_tex, TS, 0, sizeof tp, &tp);
            p_vkCmdDraw(cb, r->n, 1, r->base, 0);
            bind_masked(cb, 1, 0, blend, r->check, 1);
        }
        tp.semipass = 2;
        p_vkCmdPushConstants(cb, s_pl_tex, TS, 0, sizeof tp, &tp);
        p_vkCmdDraw(cb, r->n, 1, r->base, 0);
    }
    if (cur >= 0) p_vkCmdEndRenderPass(cb);
    s_t_close += SDL_GetPerformanceCounter() - t_in;
}

/* Tight bbox of the open untextured batch (added to s_pack_dirty at flush). */
static DirtyRect s_geo_bbox;

/* Flush accumulated untextured geometry: one render pass draws the whole batch
 * (blend/mask per the batch keys), then the image returns to TRANSFER_SRC. */
static void flush_geometry(void) {
    if (s_vcount == 0 || !s_ready) return;
    s_perf_cur.geo_flushes++;
    const uint64_t t_in = SDL_GetPerformanceCounter();
    uint32_t base = s_vbase, n = s_vcount;
    s_vbase += n; s_vcount = 0;   /* consumed region stays untouched until gpu_sync */
    int semi = s_geo_semi, blend = (semi < 0) ? 0 : (semi + 1);
    VkCommandBuffer cb = work_cb();
    if (!s_fg_replay) {
        VkDeviceSize off = 0;
        canon_pass_open(cb);
        p_vkCmdBindVertexBuffers(cb, 0, 1, &s_vbuf, &off);
        set_vp_scissor(cb);
        bind_masked(cb, 0, 0, blend, s_geo_check, s_geo_mask);
        GeoPush gp = { px_shift(), 0.0f, 512.0f };
        p_vkCmdPushConstants(cb, s_pl_geo, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof gp, &gp);
        p_vkCmdDraw(cb, n, 1, base, 0);
    }
    if (s_wide_cur >= 0 && !s_geo_mirror_suppress) {   /* native-wide mirror (queued) */
        WideRec *r = wq_push(0, semi, s_geo_check, s_geo_mask, base, n);
        GeoPush wgp = { px_shift(), 0.0f, 0.0f };
        wide_xform(s_geo_gate, &wgp.xoff, &wgp.xhalf);
        r->gp = wgp;
    }
    s_geo_mirror_suppress = 0;
    end_oneshot(cb);
    s_t_flush += SDL_GetPerformanceCounter() - t_in;
    if (s_fg_replay) { rect_clear(&s_geo_bbox); return; }   /* VRAM untouched */
    if (s_geo_bbox.set) {
        int x0 = s_geo_bbox.x0 < s_da_x1 ? s_da_x1 : s_geo_bbox.x0;
        int y0 = s_geo_bbox.y0 < s_da_y1 ? s_da_y1 : s_geo_bbox.y0;
        int x1 = s_geo_bbox.x1 > s_da_x2 ? s_da_x2 : s_geo_bbox.x1;
        int y1 = s_geo_bbox.y1 > s_da_y2 ? s_da_y2 : s_geo_bbox.y1;
        rect_add(&s_pack_dirty, x0, y0, x1, y1);
        rect_clear(&s_geo_bbox);
    }
    s_gpu_dirty = 1;
}

static inline void col555(uint16_t c, float out[3]) {
    out[0] = ((c)       & 0x1F) / 31.0f;
    out[1] = ((c >> 5)  & 0x1F) / 31.0f;
    out[2] = ((c >> 10) & 0x1F) / 31.0f;
}
static inline void col888(uint32_t c, float out[3]) {
    out[0] = ( c        & 0xFF) / 255.0f;
    out[1] = ((c >> 8)  & 0xFF) / 255.0f;
    out[2] = ((c >> 16) & 0xFF) / 255.0f;
}
static inline void push_vert(float x, float y, const float rgb[3]) {
    Vert *v = &s_vmap[s_vbase + s_vcount++];
    v->x = x; v->y = y; v->r = rgb[0]; v->g = rgb[1]; v->b = rgb[2];
    v->a = (float)s_geo_mask;
    rect_add(&s_geo_bbox, (int)floorf(x), (int)floorf(y), (int)ceilf(x), (int)ceilf(y));
}
/* Ensure room for `need` verts (flush mid-batch only between primitives). */
static void ensure_room(uint32_t need) {
    if (s_vbase + s_vcount + need > VK_VBUF_VERTS) {
        flush_geometry();
        if (s_vbase + need > VK_VBUF_VERTS)
            gpu_sync();   /* buffer exhausted: wait idle, cursors recycle to 0 */
    }
}
/* Open / continue an untextured batch with the current blend/mask state; drains
 * the textured batch and any differently-keyed untextured batch first (PS1
 * submission order). */
static void geo_prim_begin(int is_line) {
    flush_cpu_upload();   /* pending CPU writes precede this prim (PS1 order) */
    flush_tex_batch();
    int semi = s_semi_en ? s_semi_mode : -1;
    if (s_mask_check) { s_mask_check_ever = 1; s_mask_check_flip = s_flip_count; }
    const int gate = vk_bd_gate(0, is_line);
    if (s_vcount > 0 && (semi != s_geo_semi || s_mask_set != s_geo_mask || s_mask_check != s_geo_check ||
                         gate != s_geo_gate))
        flush_geometry();
    if (s_vcount == 0) { s_geo_semi = semi; s_geo_mask = s_mask_set; s_geo_check = s_mask_check; s_geo_gate = gate; }
}

static void tri3(float ax,float ay,const float ca[3],
                 float bx,float by,const float cb_[3],
                 float cx,float cy,const float cc[3]) {
    if (!s_ready) return;
    ensure_room(3);
    push_vert(ax, ay, ca); push_vert(bx, by, cb_); push_vert(cx, cy, cc);
}
static void quad(float x0,float y0,float x1,float y1,float x2,float y2,float x3,float y3,
                 const float c[3]) {
    /* (x0,y0)-(x1,y1)-(x2,y2)-(x3,y3) as two tris */
    tri3(x0,y0,c, x1,y1,c, x2,y2,c);
    tri3(x0,y0,c, x2,y2,c, x3,y3,c);
}

/* Per-prim render state. Batch keys are compared at draw-append time (the draw
 * funcs flush when they change), so the setters only record state. */
static void vkb_set_semi_transparency(int e, int m) { s_semi_en = e ? 1 : 0; s_semi_mode = m & 3; }
static void vkb_set_mask_bits(int s, int c) { s_mask_set = s ? 1 : 0; s_mask_check = c ? 1 : 0; }
static void vkb_set_texture_window(uint32_t raw) {
    s_tw_mask_x = (int)(raw & 0x1F);        s_tw_mask_y = (int)((raw >> 5) & 0x1F);
    s_tw_off_x  = (int)((raw >> 10) & 0x1F); s_tw_off_y  = (int)((raw >> 15) & 0x1F);
}
static void vkb_set_color_modulation(int r, int g, int b, int raw) {
    s_mod_r = r; s_mod_g = g; s_mod_b = b; s_mod_raw = raw ? 1 : 0;
}

/* Sub-pixel / perspective overrides for the next triangle. */
static void vkb_set_precise_triangle(int enabled,
                                     int32_t x0,int32_t y0, int32_t x1,int32_t y1,
                                     int32_t x2,int32_t y2) {
    s_pc_valid = enabled ? 1 : 0;
    if (s_pc_valid) {
        /* 16.16 native VRAM px -> float; the VK pipeline is float end-to-end. */
        s_pc_x[0] = (float)x0 / 65536.0f;  s_pc_y[0] = (float)y0 / 65536.0f;
        s_pc_x[1] = (float)x1 / 65536.0f;  s_pc_y[1] = (float)y1 / 65536.0f;
        s_pc_x[2] = (float)x2 / 65536.0f;  s_pc_y[2] = (float)y2 / 65536.0f;
    }
}
static float s_pq_scale;              /* s_pq[i] * s_pq_scale = 1 / camera depth; 0 = unknown */
static int   s_projected_uv;          /* s_proj_u/v replace the integer UVs (clipped faces) */
static float s_proj_u[3], s_proj_v[3];
void vk_renderer_set_triangle_depth_scale(float q) { s_pq_scale = s_pq_valid && q > 0.0f ? q : 0.0f; }
static void vkb_set_perspective_triangle(int enabled, float q0, float q1, float q2) {
    s_pq_scale = 0.0f;
    s_pq_valid = (enabled && q0 > 0.0f && q1 > 0.0f && q2 > 0.0f) ? 1 : 0;
    s_pq[0] = q0; s_pq[1] = q1; s_pq[2] = q2;
    /* The corrected-triangle tally lives in the software rasterizer and backs
     * gpu_texture_correction_hits() — the "is this doing anything on this
     * title" counter. Forward so it reads the same on every backend. */
    sw_set_perspective_triangle(enabled, q0, q1, q2);
}
/* The override describes exactly one triangle; drop it once submitted so a
 * later prim can never inherit it. */
static inline void precise_consumed(void) { s_pc_valid = 0; s_pq_valid = 0; s_pq_scale = 0.0f; s_projected_uv = 0; }

/* Position a vertex: the sub-pixel value when geometry_correction supplied one
 * for this triangle, else the integer GP0 position. */
#define PCX(i, v) (s_pc_valid ? s_pc_x[i] : (float)(v))
#define PCY(i, v) (s_pc_valid ? s_pc_y[i] : (float)(v))

static void vkb_draw_flat_triangle(int x0,int y0,int x1,int y1,int x2,int y2,uint16_t c){
    geo_prim_begin(0);
    float col[3]; col555(c, col);
    tri3(PCX(0,x0),PCY(0,y0),col, PCX(1,x1),PCY(1,y1),col, PCX(2,x2),PCY(2,y2),col);
    precise_consumed();
}
static void vkb_draw_gouraud_triangle(int x0,int y0,uint16_t c0,int x1,int y1,uint16_t c1,int x2,int y2,uint16_t c2){
    geo_prim_begin(0);
    float a[3],b[3],cc[3]; col555(c0,a); col555(c1,b); col555(c2,cc);
    tri3(PCX(0,x0),PCY(0,y0),a, PCX(1,x1),PCY(1,y1),b, PCX(2,x2),PCY(2,y2),cc);
    precise_consumed();
}
/* Full-screen-overlay wide pass (pause gray-filter / load fade): draw a flat
 * rect covering the FULL wide width [0, wide_w) x [y, y+h) directly into the
 * active wide surface (positions already wide-space, u_xoff = 0). Mirrors GL's
 * wide_flat_rect_direct: full wide width (the overlay must dim the revealed
 * margins too) but only the draw-area rows, like SW rt_wide() — a full-height
 * scissor let a rect that canonically clips at a vertical double-buffer band
 * boundary paint the OTHER band's rows. The 6 verts are consumed privately (s_vbase bumped)
 * so they never join a canonical batch. */
static void wide_overlay_rect(int y, int h, uint16_t c, int semi) {
    if (s_wide_cur < 0 || !s_ready) return;
    ensure_room(6);
    float col[3]; col555(c, col);
    uint32_t base = s_vbase + s_vcount;   /* == s_vbase (batches drained) */
    float x0 = 0.0f, x1 = (float)s_wide_w, fy0 = (float)y, fy1 = (float)(y + h);
    push_vert(x0, fy0, col); push_vert(x1, fy0, col); push_vert(x1, fy1, col);
    push_vert(x0, fy0, col); push_vert(x1, fy1, col); push_vert(x0, fy1, col);
    s_vbase += s_vcount; s_vcount = 0;    /* consume privately */
    int blend = (semi < 0) ? 0 : (semi + 1);
    VkCommandBuffer cb = begin_oneshot();
    wide_pass_begin(cb);   /* band scissor: full wide width, draw-area rows */
    bind_masked(cb, 0, 0, blend, s_mask_check, s_mask_set);
    GeoPush gp = { px_shift(), 0.0f, (float)s_wide_w / 2.0f };
    p_vkCmdPushConstants(cb, s_pl_geo, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof gp, &gp);
    VkDeviceSize off = 0;
    p_vkCmdBindVertexBuffers(cb, 0, 1, &s_vbuf, &off);
    p_vkCmdDraw(cb, 6, 1, base, 0);
    p_vkCmdEndRenderPass(cb);
    end_oneshot(cb);
}

static void vkb_draw_flat_rect(int x,int y,int w,int h,uint16_t c){
    /* Full-screen 2D overlay under native-wide: the canonical rect must NOT
     * mirror 1:1 (it would cover only the translated 4:3 span) — suppress the
     * batch mirror and emit one full-wide-width rect instead, so the revealed
     * margins are dimmed/faded too. Same detection as GL/SW. */
    int overlay = 0;
    if (s_wide_cur >= 0) {
        int native_w = s_wide_w - 2 * s_wide_offset;
        int lx = x - s_wide_cur_base, rx = x + w - s_wide_cur_base;
        overlay = (native_w > 0 && lx <= 0 && rx >= native_w);
    }
    if (overlay) { flush_tex_batch(); flush_geometry(); }
    geo_prim_begin(0);
    float col[3]; col555(c, col);
    if (overlay) s_geo_mirror_suppress = 1;
    quad((float)x,(float)y,(float)(x+w),(float)y,(float)(x+w),(float)(y+h),(float)x,(float)(y+h),col);
    if (overlay) {
        int semi = s_semi_en ? s_semi_mode : -1;
        flush_geometry();                  /* canonical only (mirror suppressed) */
        wide_overlay_rect(y, h, c, semi);
    }
}
/* 1px line as a thin quad along the segment. */
static void line_quad(int x0,int y0,int x1,int y1,const float ca[3],const float cb_[3]){
    float dx = (float)(x1 - x0), dy = (float)(y1 - y0);
    float len = dx*dx + dy*dy;
    float nx = 0.5f, ny = 0.5f;
    if (len > 0.0001f) { float il = 0.5f / (float)sqrt(len); nx = -dy * il; ny = dx * il; }
    ensure_room(6);
    float ax0 = x0 + nx, ay0 = y0 + ny, bx0 = x0 - nx, by0 = y0 - ny;
    float ax1 = x1 + nx, ay1 = y1 + ny, bx1 = x1 - nx, by1 = y1 - ny;
    tri3(ax0,ay0,ca, ax1,ay1,cb_, bx1,by1,cb_);
    tri3(ax0,ay0,ca, bx1,by1,cb_, bx0,by0,ca);
}
static void vkb_draw_line(int x0,int y0,int x1,int y1,uint16_t c){
    geo_prim_begin(1);
    float col[3]; col555(c, col); line_quad(x0,y0,x1,y1,col,col);
}
static void vkb_draw_shaded_line(int x0,int y0,uint16_t c0,int x1,int y1,uint16_t c1){
    geo_prim_begin(1);
    float a[3],b[3]; col555(c0,a); col555(c1,b); line_quad(x0,y0,x1,y1,a,b);
}

/* ---- textured primitives + VRAM->VRAM copy (Phase 3) ------------------- */
/* Flush the textured batch: two STP-split passes (pass 1 opaque texels, pass 2
 * semi-transparent texels with PS1 blending). Per-prim texture state rode in the
 * vertex; the batch keys (semi/mask/twin/filter) drive the push constants and
 * pipeline. The raw mirror was made coherent at append time (flush_pack_if_sampling). */
static void flush_tex_batch(void) {
    if (s_tcount == 0 || !s_ready) return;
    s_perf_cur.tex_flushes++;
    const uint64_t t_in = SDL_GetPerformanceCounter();
    uint32_t base = s_tbase, n = s_tcount;
    s_tbase += n; s_tcount = 0;   /* consumed region stays untouched until gpu_sync */
    int semi = s_tb_semi, blend = (semi < 0) ? 0 : (semi + 1);
    if (s_raw_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) pass_close();   /* no barrier inside a pass */
    VkCommandBuffer cb = work_cb();
    img_to(cb, s_raw_img, &s_raw_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VkDeviceSize off = 0;
    TexPush tp = {0};
    tp.shift = px_shift(); tp.xoff = 0.0f; tp.xhalf = 512.0f;
    tp.maskset = s_tb_mask; tp.filter = s_tb_filter;
    tp.twin[0] = s_tb_twin[0]; tp.twin[1] = s_tb_twin[1]; tp.twin[2] = s_tb_twin[2]; tp.twin[3] = s_tb_twin[3];
    const int fix = stencil_fix_needed();
    if (fix) s_stat_fix_on++; else s_stat_fix_off++;
    static int dbg_nocanon = -1;
    if (dbg_nocanon < 0) { const char *e = getenv("PSX_VK_NO_CANON"); dbg_nocanon = e && *e == '1'; }
    if (!s_fg_replay && !(dbg_nocanon && s_wide_cur >= 0)) {   /* canonical VRAM pass (skipped for a generated picture) */
    canon_pass_open(cb);
    {
        VkDescriptorSet sets[2] = { s_ds_tex, s_tb_hd_ds ? s_tb_hd_ds : s_hd_none_ds };
        p_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pl_tex, 0, 2, sets, 0, NULL);
    }
    p_vkCmdBindVertexBuffers(cb, 0, 1, &s_tbuf, &off);
    set_vp_scissor(cb);
    if (semi < 0) {
        /* OPAQUE batch: the STP bit does not gate COLOUR (every texel is
         * opaque), so colour draws in ONE ordered pass over all texels —
         * painter's order across the batch is exact — followed by a
         * COLOUR-MASKED pass that only fixes the STENCIL for STP=1 texels.
         * (The old whole-batch STP-split let a behind prim's STP=1 texels
         * paint OVER a front prim's STP=0 colour — the water-sparkle field
         * speckled every hill in the Tomba2 attract demo. Mirrors GL's
         * tex_batch_draw_passes.) */
        bind_masked(cb, 1, 0, 0, s_tb_check, s_tb_mask);
        tp.semipass = 0;
        p_vkCmdPushConstants(cb, s_pl_tex, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof tp, &tp);
        p_vkCmdDraw(cb, n, 1, base, 0);
        if (fix) {
            bind_masked_stencil_only(cb, 1, 0, s_tb_check, 1);
            tp.semipass = 2;
            p_vkCmdPushConstants(cb, s_pl_tex, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof tp, &tp);
            p_vkCmdDraw(cb, n, 1, base, 0);
        }
    } else {
        /* SEMI batch: genuine two-pass — STP=0 texels opaque, STP=1 texels
         * blended per the PSX mode. Cross-prim order is kept by isolating
         * semi prims to one per batch (gpu_textured_triangle), so the two
         * passes never straddle overlapping prims. */
        bind_masked(cb, 1, 0, 0, s_tb_check, s_tb_mask);
        tp.semipass = 1;
        p_vkCmdPushConstants(cb, s_pl_tex, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof tp, &tp);
        p_vkCmdDraw(cb, n, 1, base, 0);
        bind_masked(cb, 1, 0, blend, s_tb_check, 1);
        tp.semipass = 2;
        p_vkCmdPushConstants(cb, s_pl_tex, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof tp, &tp);
        p_vkCmdDraw(cb, n, 1, base, 0);
    }
    }   /* !s_fg_replay */
    if (s_wide_cur >= 0) {                 /* native-wide mirror (queued) */
        WideRec *r = wq_push(1, semi, s_tb_check, s_tb_mask, base, n);
        wide_xform(s_tb_gate, &tp.xoff, &tp.xhalf);
        r->tp = tp; r->hd_ds = s_tb_hd_ds; r->stencil_fix = (uint8_t)fix;
    }
    end_oneshot(cb);
    s_gpu_dirty = 1;
    s_t_flush += SDL_GetPerformanceCounter() - t_in;
}

/* Shared PS1 uv-sampling model (limits + mirrored-2D compensation) — one
 * implementation for GL/VK/SW, see gpu_uv.h. */
#include "gpu_uv.h"

/* ---- HD replacement texture cache ---------------------------------------- */
static int hd_vk_init(void) {
    VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                       VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ci.bindingCount = 1; ci.pBindings = &b;
    if (p_vkCreateDescriptorSetLayout(s_dev, &ci, NULL, &s_dsl_hd) != VK_SUCCESS) return 0;
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 16.0f;
    if (p_vkCreateSampler(s_dev, &sci, NULL, &s_hd_samp) != VK_SUCCESS) return 0;
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, HD_VK_CAP + 1 };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = HD_VK_CAP + 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (p_vkCreateDescriptorPool(s_dev, &pci, NULL, &s_hd_pool) != VK_SUCCESS) return 0;
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = s_hd_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &s_dsl_hd;
    if (p_vkAllocateDescriptorSets(s_dev, &ai, &s_hd_none_ds) != VK_SUCCESS) return 0;
    for (int i = 0; i < HD_VK_CAP; i++)
        if (p_vkAllocateDescriptorSets(s_dev, &ai, &s_hdv[i].ds) != VK_SUCCESS) return 0;
    /* 1x1 dummy bound by draws that have no replacement. */
    if (!make_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &s_hd_none_img, &s_hd_none_mem, &s_hd_none_view)) return 0;
    {
        VkCommandBuffer cb = begin_oneshot();
        img_barrier(cb, s_hd_none_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue z = {{0, 0, 0, 0}};
        VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        p_vkCmdClearColorImage(cb, s_hd_none_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &rng);
        img_barrier(cb, s_hd_none_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        end_oneshot(cb);
    }
    VkDescriptorImageInfo ii = { s_hd_samp, s_hd_none_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = s_hd_none_ds; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    p_vkUpdateDescriptorSets(s_dev, 1, &w, 0, NULL);
    const char *e = getenv("PSX_HD_VK_CACHE_MIB");
    const long mib = e ? atol(e) : 768;
    s_hd_budget = (size_t)(mib < 64 ? 64 : mib) << 20;
    s_hd_ready = 1;
    return 1;
}
static void hd_vk_free(HdVkTex *t) {
    if (t->view) p_vkDestroyImageView(s_dev, t->view, NULL);
    if (t->img)  p_vkDestroyImage(s_dev, t->img, NULL);
    if (t->mem)  p_vkFreeMemory(s_dev, t->mem, NULL);
    s_hd_bytes -= t->bytes;
    t->view = VK_NULL_HANDLE; t->img = VK_NULL_HANDLE; t->mem = VK_NULL_HANDLE;
    t->key = t->gen = 0; t->bytes = 0; t->w = t->h = 0;
}
/* Queue idle: destroy retired images and release their slots. */
static void hd_vk_reap(void) {
    for (int i = 0; i < s_hd_retired_n; i++) {
        if (s_hd_retired[i].view) p_vkDestroyImageView(s_dev, s_hd_retired[i].view, NULL);
        if (s_hd_retired[i].img)  p_vkDestroyImage(s_dev, s_hd_retired[i].img, NULL);
        if (s_hd_retired[i].mem)  p_vkFreeMemory(s_dev, s_hd_retired[i].mem, NULL);
        s_hd_slot_busy[s_hd_retired[i].slot] = 0;
    }
    s_hd_retired_n = 0;
}
static void hd_vk_retire(int slot) {
    HdVkTex *t = &s_hdv[slot];
    s_hd_retired[s_hd_retired_n].img = t->img; s_hd_retired[s_hd_retired_n].mem = t->mem;
    s_hd_retired[s_hd_retired_n].view = t->view; s_hd_retired[s_hd_retired_n].slot = slot;
    s_hd_retired_n++;
    s_hd_slot_busy[slot] = 1;
    s_hd_bytes -= t->bytes;
    t->view = VK_NULL_HANDLE; t->img = VK_NULL_HANDLE; t->mem = VK_NULL_HANDLE;
    t->key = t->gen = 0; t->bytes = 0; t->w = t->h = 0;
}
static void hd_vk_shutdown(void) {
    if (!s_dev) return;
    hd_vk_reap();
    for (int i = 0; i < HD_VK_CAP; i++) if (s_hdv[i].img) hd_vk_free(&s_hdv[i]);
    if (s_hd_none_view) { p_vkDestroyImageView(s_dev, s_hd_none_view, NULL); s_hd_none_view = VK_NULL_HANDLE; }
    if (s_hd_none_img)  { p_vkDestroyImage(s_dev, s_hd_none_img, NULL); s_hd_none_img = VK_NULL_HANDLE; }
    if (s_hd_none_mem)  { p_vkFreeMemory(s_dev, s_hd_none_mem, NULL); s_hd_none_mem = VK_NULL_HANDLE; }
    if (s_hd_pool) { p_vkDestroyDescriptorPool(s_dev, s_hd_pool, NULL); s_hd_pool = VK_NULL_HANDLE; }
    if (s_hd_samp) { p_vkDestroySampler(s_dev, s_hd_samp, NULL); s_hd_samp = VK_NULL_HANDLE; }
    if (s_dsl_hd)  { p_vkDestroyDescriptorSetLayout(s_dev, s_dsl_hd, NULL); s_dsl_hd = VK_NULL_HANDLE; }
    s_hd_ready = 0; s_hd_bytes = 0;
}
static unsigned hd_vk_hint_of(uint64_t key, uint64_t gen) {
    uint64_t h = (key ^ (gen * 0x9E3779B97F4A7C15ull)) * 0xD6E8FEB86659FD93ull;
    return (unsigned)(h >> 40) & (HD_VK_HINT - 1);
}
static int hd_vk_find(uint64_t key, uint64_t gen) {
    const unsigned hi = hd_vk_hint_of(key, gen);
    const int g = (int)s_hd_hint[hi] - 1;
    if (g >= 0 && s_hdv[g].img && s_hdv[g].key == key && s_hdv[g].gen == gen) return g;
    for (int i = 0; i < HD_VK_CAP; i++)
        if (s_hdv[i].img && s_hdv[i].key == key && s_hdv[i].gen == gen) {
            s_hd_hint[hi] = (uint16_t)(i + 1);
            return i;
        }
    return -1;
}
/* The texture-pack decoder asks this before queueing a PNG decode: the GPU
 * copy outlives its smaller CPU cache. */
int vk_renderer_hd_texture_cached(uint64_t key, uint64_t generation, uint32_t *width, uint32_t *height) {
    if (!s_hd_ready) return 0;
    const int i = hd_vk_find(key, generation);
    if (i < 0) return 0;
    *width = s_hdv[i].w; *height = s_hdv[i].h;
    return 1;
}
/* Cache slot of a replacement image, uploading it (with mips) on first use.
 * -1 = not available (pixels not decoded yet, or it cannot be cached). */
static int hd_vk_texture(const GpuHdTextureImage *im) {
    int slot = hd_vk_find(im->cache_key, im->generation);
    if (slot >= 0) { s_hdv[slot].last = ++s_hd_clock; return slot; }
    if (!im->rgba || !im->width || !im->height || im->stride != im->width * 4u ||
        im->width > 8192u || im->height > 8192u) return -1;
    uint32_t mips = 1;
    for (uint32_t m = im->width > im->height ? im->width : im->height; m > 1 && mips < 8; m >>= 1) mips++;
    const size_t bytes = (size_t)im->stride * im->height * 4u / 3u;
    if (bytes > s_hd_budget) return -1;
    for (int i = 0; i < HD_VK_CAP && slot < 0; i++) if (!s_hdv[i].img && !s_hd_slot_busy[i]) slot = i;
    if (s_hd_bytes + bytes > s_hd_budget || slot < 0) {
        /* Retire least-recently-used images (never one used this frame is
         * guaranteed only by LRU order; retired images stay alive until the
         * queue is idle, so pending draws are safe). */
        int guard = 0;
        while ((s_hd_bytes + bytes > s_hd_budget || slot < 0) && guard++ < 64) {
            int old = -1;
            for (int i = 0; i < HD_VK_CAP; i++)
                if (s_hdv[i].img && (old < 0 || s_hdv[i].last < s_hdv[old].last)) old = i;
            if (old < 0) break;
            hd_vk_retire(old);
            gpu_hd_textures_note_gl_cache(1);
            if (slot < 0) {
                /* No free slot at all: the retired one frees only at a sync. */
                flush_tex_batch(); flush_geometry(); gpu_sync();
                for (int i = 0; i < HD_VK_CAP && slot < 0; i++) if (!s_hdv[i].img && !s_hd_slot_busy[i]) slot = i;
            }
        }
        if (slot < 0 || s_hd_bytes + bytes > s_hd_budget) return -1;
    }
    HdVkTex *t = &s_hdv[slot];
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent.width = im->width; ci.extent.height = im->height; ci.extent.depth = 1;
    ci.mipLevels = mips; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (p_vkCreateImage(s_dev, &ci, NULL, &t->img) != VK_SUCCESS) { t->img = VK_NULL_HANDLE; return -1; }
    VkMemoryRequirements req; p_vkGetImageMemoryRequirements(s_dev, t->img, &req);
    VkMemoryAllocateInfo mi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mi.allocationSize = req.size;
    mi.memoryTypeIndex = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBuffer sbuf = VK_NULL_HANDLE; VkDeviceMemory smem = VK_NULL_HANDLE; void *map = NULL;
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    if (mi.memoryTypeIndex == UINT32_MAX || p_vkAllocateMemory(s_dev, &mi, NULL, &t->mem) != VK_SUCCESS) {
        t->mem = VK_NULL_HANDLE; goto fail;
    }
    p_vkBindImageMemory(s_dev, t->img, t->mem, 0);
    vi.image = t->img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = mips; vi.subresourceRange.layerCount = 1;
    if (p_vkCreateImageView(s_dev, &vi, NULL, &t->view) != VK_SUCCESS) { t->view = VK_NULL_HANDLE; goto fail; }
    if (!make_staging((VkDeviceSize)im->stride * im->height, &sbuf, &smem, &map)) goto fail;
    memcpy(map, im->rgba, (size_t)im->stride * im->height);
    {
        VkCommandBuffer cb = begin_oneshot();
        VkImageMemoryBarrier br = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        br.srcQueueFamilyIndex = br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        br.image = t->img;
        br.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; br.subresourceRange.layerCount = 1;
        br.subresourceRange.baseMipLevel = 0; br.subresourceRange.levelCount = mips;
        br.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        br.srcAccessMask = 0; br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, NULL, 0, NULL, 1, &br);
        VkBufferImageCopy rc = {0};
        rc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; rc.imageSubresource.layerCount = 1;
        rc.imageExtent.width = im->width; rc.imageExtent.height = im->height; rc.imageExtent.depth = 1;
        p_vkCmdCopyBufferToImage(cb, sbuf, t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rc);
        int32_t mw = (int32_t)im->width, mh = (int32_t)im->height;
        br.subresourceRange.levelCount = 1;
        for (uint32_t m = 1; m <= mips; m++) {
            /* Level m-1 is complete: make it a blit source (or final). */
            br.subresourceRange.baseMipLevel = m - 1;
            br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            br.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; br.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, 0, NULL, 0, NULL, 1, &br);
            if (m < mips) {
                const int32_t nw = mw > 1 ? mw / 2 : 1, nh = mh > 1 ? mh / 2 : 1;
                VkImageBlit bl = {0};
                bl.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bl.srcSubresource.layerCount = 1;
                bl.srcSubresource.mipLevel = m - 1;
                bl.srcOffsets[1].x = mw; bl.srcOffsets[1].y = mh; bl.srcOffsets[1].z = 1;
                bl.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bl.dstSubresource.layerCount = 1;
                bl.dstSubresource.mipLevel = m;
                bl.dstOffsets[1].x = nw; bl.dstOffsets[1].y = nh; bl.dstOffsets[1].z = 1;
                p_vkCmdBlitImage(cb, t->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
                mw = nw; mh = nh;
            }
            br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; br.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            br.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0, 0, NULL, 0, NULL, 1, &br);
        }
        end_oneshot(cb);
    }
    defer_staging(sbuf, smem);
    {
        VkDescriptorImageInfo ii = { s_hd_samp, t->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = t->ds; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
        p_vkUpdateDescriptorSets(s_dev, 1, &w, 0, NULL);
    }
    t->key = im->cache_key; t->gen = im->generation; t->w = im->width; t->h = im->height;
    t->bytes = bytes; s_hd_bytes += bytes; t->last = ++s_hd_clock;
    s_hd_hint[hd_vk_hint_of(t->key, t->gen)] = (uint16_t)(slot + 1);
    s_hd_stat_upload++;
    gpu_hd_textures_note_gl_cache(0);
    return slot;
fail:
    if (t->view) { p_vkDestroyImageView(s_dev, t->view, NULL); t->view = VK_NULL_HANDLE; }
    if (t->img)  { p_vkDestroyImage(s_dev, t->img, NULL); t->img = VK_NULL_HANDLE; }
    if (t->mem)  { p_vkFreeMemory(s_dev, t->mem, NULL); t->mem = VK_NULL_HANDLE; }
    return -1;
}

/* Append a textured triangle to the batch (per-prim state in the vertex; flush
 * first if the batch keys differ or the raw mirror needs packing). col = 3x RGB
 * modulation per vertex; lim = uv bounds (NULL -> derive). */
static void gpu_textured_triangle(const int *xs, const int *ys, const int *us, const int *vs,
                                  const float *col, uint16_t texpage,
                                  uint16_t clut_x, uint16_t clut_y, int rawtex,
                                  int semi, const int *lim) {
    if (!s_ready) return;
    int lim_buf[4];
    int uv_buf[6];
    const int texfilter = (s_texfilter || ((!lim || s_projected_uv) && touch_ui_tex_smooth())) ? 1 : 0;
    if (!lim) {
        /* Poly path: exact sampled bounds from the ORIGINAL uvs, then the
         * center-sampling mirror compensation (rect prims arrive with their
         * own precomputed lim and pre-bumped uvs). */
        int *mu = uv_buf, *mv = uv_buf + 3;
        for (int i = 0; i < 3; i++) { mu[i] = us[i]; mv[i] = vs[i]; }
        psx_uv_tri_center_sample(xs, ys, mu, mv, s_pq_valid, lim_buf);
        us = mu; vs = mv;
        lim = lim_buf;
    }
    int base_x = (texpage & 0xF) * 64;
    int base_y = ((texpage >> 4) & 1) * 256;
    int depth  = (texpage >> 7) & 3; if (depth > 2) depth = 2;

    flush_geometry();   /* drain untextured (PS1 order) */
    flush_cpu_upload(); /* uploaded texels/CLUTs must be sampleable + order */
    flush_pack_if_sampling(base_x, base_y, depth, clut_x, clut_y);

    int twx = s_tw_mask_x, twy = s_tw_mask_y, tox = s_tw_off_x, toy = s_tw_off_y;
    /* A semi-transparent prim must NOT coalesce with its neighbours (its two
     * STP passes would straddle overlapping prims — painter's-order violation;
     * see flush_tex_batch). It drains the open batch and draws alone; the next
     * append flushes it out again. Opaque prims keep batching. */
    /* HD replacement for this draw, if the active texture pack has one. */
    VkDescriptorSet hd_ds = VK_NULL_HANDLE;
    float hd_src[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
    int hd_mode = 0;
    if (s_hd_ready && gpu_hd_textures_active()) {   /* replacements and/or dumping */
        const uint32_t window = (uint32_t)(twx | (twy << 5) | (tox << 10) | (toy << 15));
        /* Memo by the exact sampled bounds. On a miss the pack is asked for
         * exactly those bounds first: when one replacement file covers them
         * the file itself is used (uploaded once, no CPU-side assembly). Only
         * otherwise is a 16-texel-aligned footprint requested, so that the
         * assembled ("composed") image is the same one frame after frame
         * while the camera shifts the exact bounds by a texel. */
        uint64_t mk = ((uint64_t)texpage << 48) ^ ((uint64_t)clut_x << 36) ^ ((uint64_t)clut_y << 24) ^
                      ((uint64_t)lim[0] << 16) ^ ((uint64_t)lim[1] << 8) ^ (uint64_t)lim[2] ^
                      ((uint64_t)lim[3] << 40) ^ ((uint64_t)window * 0x9E3779B97F4A7C15ull) ^
                      (semi >= 0 ? 0x8000000000000000ull : 0);
        mk = (mk ^ (mk >> 29)) * 0xBF58476D1CE4E5B9ull; mk ^= mk >> 32; if (!mk) mk = 1;
        HdMemo *m = &s_hd_memo[mk & (HD_MEMO_N - 1)];
        int hit = 0;
        if (m->key == mk && !gpu_hd_textures_dump_enabled()) {
            if (m->slot < 0) hit = (s_flip_count - m->stamp) < 20u;
            else hit = s_hdv[m->slot].img && s_hdv[m->slot].key == m->tex_key && s_hdv[m->slot].gen == m->tex_gen;
        }
        if (hit) {
            if (m->slot >= 0) {
                s_hdv[m->slot].last = ++s_hd_clock;
                hd_ds = s_hdv[m->slot].ds; hd_mode = m->mode;
                hd_src[0] = m->src[0]; hd_src[1] = m->src[1]; hd_src[2] = m->src[2]; hd_src[3] = m->src[3];
            }
        } else {
            GpuHdTextureImage hd;
            int slot = -1;
            const int qlim[4] = { lim[0] & ~15, lim[1] & ~15, lim[2] | 15, lim[3] | 15 };
            const int *used = lim;
            s_hd_stat_miss++;
            const int ppw0 = depth == 0 ? 4 : depth == 1 ? 2 : 1;
            const int animated = !gpu_hd_textures_dump_enabled() &&
                ((window ? hd_rect_animated(base_x, base_y, base_x + 255 / ppw0, base_y + 255)
                         : hd_rect_animated(base_x + lim[0] / ppw0, base_y + lim[1],
                                            base_x + lim[2] / ppw0, base_y + lim[3])) ||
                 (depth < 2 && hd_rect_animated(clut_x, clut_y, clut_x + (depth == 0 ? 15 : 255), clut_y)));
            if (s_fg_replay) gpu_hd_textures_suppress_dumps(1);
            memset(&hd, 0, sizeof hd);
            int got = animated ? 0 : gpu_hd_textures_acquire_gl_draw(texpage, clut_x, clut_y, lim, window, semi >= 0, &hd);
            if (!animated && (!got || hd.alpha_mode == 4)) {
                gpu_hd_textures_release_image(&hd);
                used = qlim;
                got = gpu_hd_textures_acquire_gl_draw(texpage, clut_x, clut_y, qlim, window, semi >= 0, &hd);
            }
            if (got && hd.alpha_mode >= 1 && hd.alpha_mode <= 4 && hd.source_width && hd.source_height) {
                slot = hd_vk_texture(&hd);
                if (slot >= 0) {
                    hd_ds = s_hdv[slot].ds; hd_mode = hd.alpha_mode;
                    hd_src[0] = (float)hd.origin_u; hd_src[1] = (float)hd.origin_v;
                    hd_src[2] = (float)hd.source_width; hd_src[3] = (float)hd.source_height;
                    gpu_hd_textures_note_applied();
                    if (hd_mode == 4) s_hd_stat_composed++; else s_hd_stat_direct++;
                }
            }
            gpu_hd_textures_release_image(&hd);
            if (s_fg_replay) gpu_hd_textures_suppress_dumps(0);
            /* Remember the answer with the VRAM words it depends on. */
            const int ppw = depth == 0 ? 4 : depth == 1 ? 2 : 1;
            if (!m->key) s_hd_memo_live++;
            m->key = mk; m->slot = (int16_t)slot; m->mode = (uint8_t)hd_mode; m->stamp = s_flip_count;
            m->tex_key = slot >= 0 ? s_hdv[slot].key : 0; m->tex_gen = slot >= 0 ? s_hdv[slot].gen : 0;
            m->src[0] = hd_src[0]; m->src[1] = hd_src[1]; m->src[2] = hd_src[2]; m->src[3] = hd_src[3];
            if (window) { m->x0 = (int16_t)base_x; m->x1 = (int16_t)(base_x + 255 / ppw); m->y0 = (int16_t)base_y; m->y1 = (int16_t)(base_y + 255); }
            else { m->x0 = (int16_t)(base_x + used[0] / ppw); m->x1 = (int16_t)(base_x + used[2] / ppw);
                   m->y0 = (int16_t)(base_y + used[1]); m->y1 = (int16_t)(base_y + used[3]); }
            m->cy = (int16_t)clut_y; m->cx0 = (int16_t)clut_x;
            m->cx1 = (int16_t)(depth == 0 ? clut_x + 15 : depth == 1 ? clut_x + 255 : clut_x - 1);
            if (m->x0 < s_hd_bb[0]) s_hd_bb[0] = m->x0;
            if (m->y0 < s_hd_bb[1]) s_hd_bb[1] = m->y0;
            if (m->x1 > s_hd_bb[2]) s_hd_bb[2] = m->x1;
            if (m->y1 > s_hd_bb[3]) s_hd_bb[3] = m->y1;
            if (m->cx1 >= m->cx0) {
                if (m->cx0 < s_hd_bb[0]) s_hd_bb[0] = m->cx0;
                if (m->cy  < s_hd_bb[1]) s_hd_bb[1] = m->cy;
                if (m->cx1 > s_hd_bb[2]) s_hd_bb[2] = m->cx1;
                if (m->cy  > s_hd_bb[3]) s_hd_bb[3] = m->cy;
            }
        }
    }
    if (s_mask_check) { s_mask_check_ever = 1; s_mask_check_flip = s_flip_count; }
    int isolate = (semi >= 0);
    const int gate = vk_bd_gate(1, 0);
    if (s_tcount > 0 &&
        (isolate || gate != s_tb_gate || hd_ds != s_tb_hd_ds || semi != s_tb_semi || s_mask_set != s_tb_mask || s_mask_check != s_tb_check ||
         texfilter != s_tb_filter ||
         twx != s_tb_twin[0] || twy != s_tb_twin[1] || tox != s_tb_twin[2] || toy != s_tb_twin[3]))
        flush_tex_batch();
    if (s_tbase + s_tcount + 3 > VK_TBUF_VERTS) {
        flush_tex_batch();
        if (s_tbase + 3 > VK_TBUF_VERTS)
            gpu_sync();   /* buffer exhausted: wait idle, cursors recycle to 0 */
    }
    if (s_tcount == 0) {
        s_tb_semi = semi; s_tb_mask = s_mask_set; s_tb_check = s_mask_check; s_tb_filter = texfilter;
        s_tb_gate = gate; s_tb_hd_ds = hd_ds;
        s_tb_twin[0] = twx; s_tb_twin[1] = twy; s_tb_twin[2] = tox; s_tb_twin[3] = toy;
    }
    int bx0 = xs[0], bx1 = xs[0], by0 = ys[0], by1 = ys[0];
    for (int i = 0; i < 3; i++) {
        float *vp = s_tmap[s_tbase + s_tcount + i].v;
        vp[0] = s_pc_valid ? s_pc_x[i] : (float)xs[i];
        vp[1] = s_pc_valid ? s_pc_y[i] : (float)ys[i];
        vp[2] = s_projected_uv ? s_proj_u[i] : (float)us[i];
        vp[3] = s_projected_uv ? s_proj_v[i] : (float)vs[i];
        vp[4] = col[i*3+0];     vp[5] = col[i*3+1];     vp[6] = col[i*3+2]; vp[7] = 1.0f;
        vp[8]  = (float)base_x;  vp[9]  = (float)base_y;
        vp[10] = (float)clut_x;  vp[11] = (float)clut_y;
        vp[12] = (float)depth;   vp[13] = (float)rawtex;
        vp[14] = (float)lim[0];  vp[15] = (float)lim[1];
        vp[16] = (float)lim[2];  vp[17] = (float)lim[3];
        vp[18] = s_pq_valid ? s_pq[i] : 0.0f;   /* a_q; 0 = affine */
        vp[19] = hd_src[0]; vp[20] = hd_src[1]; vp[21] = hd_src[2]; vp[22] = hd_src[3];
        vp[23] = (float)hd_mode;
        if (xs[i] < bx0) bx0 = xs[i]; if (xs[i] > bx1) bx1 = xs[i];
        if (ys[i] < by0) by0 = ys[i]; if (ys[i] > by1) by1 = ys[i];
    }
    s_tcount += 3;
    if (s_fg_replay) return;   /* generated picture: no VRAM dirty tracking */
    /* A sub-pixel-corrected vertex lies in [int, int+1) of the integer position
     * it was rounded from, so widen the readback rect by one pixel. */
    if (s_pc_valid) { bx1 += 1; by1 += 1; }
    /* mark dirty for later readback/pack (clamped to draw area) */
    if (bx0 < s_da_x1) bx0 = s_da_x1; if (by0 < s_da_y1) by0 = s_da_y1;
    if (bx1 > s_da_x2) bx1 = s_da_x2; if (by1 > s_da_y2) by1 = s_da_y2;
    rect_add(&s_pack_dirty, bx0, by0, bx1, by1);
    s_gpu_dirty = 1;
}

static void gpu_textured_rect(int x,int y,int w,int h, int u0,int v0,int u1,int v1,
                              uint16_t clut_x,uint16_t clut_y,uint16_t tp,int semi) {
    if (w <= 0 || h <= 0) return;
    float mr = s_mod_r/255.0f, mg = s_mod_g/255.0f, mb = s_mod_b/255.0f;
    float col[9] = { mr,mg,mb, mr,mg,mb, mr,mg,mb };
    /* Mirrored quads arrive here as scaled rects with u0>u1 / v0>v1 (see
     * the GL backend): exact bounds from the original corners, then the
     * mirror bump (gpu_uv.h). */
    int lim[4];
    psx_uv_rect_limits(u0, v0, u1, v1, lim);
    psx_uv_rect_mirror_offset(&u0, &v0, &u1, &v1);
    int xs1[3]={x, x+w, x},    ys1[3]={y, y, y+h};
    int us1[3]={u0,u1,u0},     vs1[3]={v0,v0,v1};
    gpu_textured_triangle(xs1,ys1,us1,vs1,col,tp,clut_x,clut_y,s_mod_raw,semi,lim);
    int xs2[3]={x+w, x, x+w},  ys2[3]={y, y+h, y+h};
    int us2[3]={u1,u0,u1},     vs2[3]={v0,v1,v1};
    gpu_textured_triangle(xs2,ys2,us2,vs2,col,tp,clut_x,clut_y,s_mod_raw,semi,lim);
}

static void vkb_draw_textured_triangle(int x0,int y0,int u0,int v0,int x1,int y1,int u1,int v1,
                                       int x2,int y2,int u2,int v2,uint16_t cx,uint16_t cy,uint16_t tp){
    int xs[3]={x0,x1,x2}, ys[3]={y0,y1,y2}, us[3]={u0,u1,u2}, vs[3]={v0,v1,v2};
    float mr=s_mod_r/255.0f, mg=s_mod_g/255.0f, mb=s_mod_b/255.0f;
    float col[9]={mr,mg,mb, mr,mg,mb, mr,mg,mb};
    gpu_textured_triangle(xs,ys,us,vs,col,tp,cx,cy,s_mod_raw, s_semi_en?s_semi_mode:-1, NULL);
    precise_consumed();
}
static void vkb_draw_shaded_textured_triangle(int x0,int y0,int u0,int v0,uint32_t c0,
                                              int x1,int y1,int u1,int v1,uint32_t c1,
                                              int x2,int y2,int u2,int v2,uint32_t c2,
                                              uint16_t cx,uint16_t cy,uint16_t tp,int raw){
    int xs[3]={x0,x1,x2}, ys[3]={y0,y1,y2}, us[3]={u0,u1,u2}, vs[3]={v0,v1,v2};
    /* shaded-textured: per-vertex 24-bit colour modulates the texel (PS1 *2). */
    float col[9];
    col888(c0, &col[0]); col888(c1, &col[3]); col888(c2, &col[6]);
    gpu_textured_triangle(xs,ys,us,vs,col,tp,cx,cy,raw, s_semi_en?s_semi_mode:-1, NULL);
    precise_consumed();
}
/* Faces crossing the camera plane or the PS1's coordinate range: clip the
 * signed homogeneous polygon on the CPU (shared with the GL backend) and draw
 * what is left as ordinary perspective triangles, in order, into both passes. */
int vk_renderer_projective_supported(void) { return s_ctx_ok && s_ready && s_wide_cur >= 0; }
static void projected_triangle(const PSXProjectedVertex vertices[3], uint16_t texpage,
                               uint16_t cx, uint16_t cy, int raw, int semi, int perspective) {
    PSXProjectedVertex polygon[12];
    const double left = fmin(s_da_x1, -wide_dx()) - 1.0;
    const double right = fmax(s_da_x2 + 1, s_wide_w - wide_dx()) + 1.0;
    const int n = psx_projective_clip_triangle(vertices, polygon, left, right, s_da_y1 - 1.0, s_da_y2 + 1.0);
    int limits[4] = { (int)vertices[0].u, (int)vertices[0].v, (int)vertices[0].u, (int)vertices[0].v };
    for (int i = 1; i < 3; i++) {
        if (vertices[i].u < limits[0]) limits[0] = (int)vertices[i].u;
        if (vertices[i].v < limits[1]) limits[1] = (int)vertices[i].v;
        if (vertices[i].u > limits[2]) limits[2] = (int)vertices[i].u;
        if (vertices[i].v > limits[3]) limits[3] = (int)vertices[i].v;
    }
    for (int k = 1; k + 1 < n; k++) {
        const PSXProjectedVertex v[3] = { polygon[0], polygon[k], polygon[k + 1] };
        int xs[3], ys[3], us[3], vs[3]; float col[9];
        s_pc_valid = 1; s_pq_valid = perspective ? 1 : 0; s_projected_uv = 1;
        for (int i = 0; i < 3; i++) {
            s_pc_x[i] = (float)(v[i].x / v[i].z); s_pc_y[i] = (float)(v[i].y / v[i].z);
            xs[i] = (int)floorf(s_pc_x[i]); ys[i] = (int)floorf(s_pc_y[i]);
            s_pq[i] = (float)(1.0 / v[i].z);
            s_proj_u[i] = (float)v[i].u; s_proj_v[i] = (float)v[i].v;
            us[i] = (int)v[i].u; vs[i] = (int)v[i].v;
            col[i*3] = (float)v[i].r; col[i*3+1] = (float)v[i].g; col[i*3+2] = (float)v[i].b;
        }
        gpu_textured_triangle(xs, ys, us, vs, col, texpage, cx, cy, raw, semi, limits);
        precise_consumed();
    }
    precise_consumed();
}
static void fg_rec_projected(const PSXProjectedVertex v[3], uint16_t texpage, uint16_t cx, uint16_t cy,
                             int raw, int semi, int perspective);   /* frame generation */
void vk_renderer_draw_projected_triangle(const PSXProjectedVertex *vertices, uint16_t texpage,
                                         uint16_t cx, uint16_t cy, int raw, int semi, int perspective) {
    if (!vk_renderer_projective_supported()) return;
    fg_rec_projected(vertices, texpage, cx, cy, raw, semi, perspective);
    projected_triangle(vertices, texpage, cx, cy, raw, semi, perspective);
}
static void vkb_draw_textured_rect(int x,int y,int w,int h,int u,int v,uint16_t cx,uint16_t cy,uint16_t tp){
    gpu_textured_rect(x,y,w,h, u,v, u+w,v+h, cx,cy,tp, s_semi_en?s_semi_mode:-1);
}
static void vkb_draw_textured_rect_scaled(int x,int y,int w,int h,int u0,int v0,int u1,int v1,uint16_t cx,uint16_t cy,uint16_t tp){
    gpu_textured_rect(x,y,w,h, u0,v0, u1,v1, cx,cy,tp, s_semi_en?s_semi_mode:-1);
}

/* VRAM->VRAM copy: blit the source hr region to scratch (resolves overlap),
 * then blit scratch -> dest with STP/mask split. Mirrors the GL backend. */
static void vkb_copy_rect(int sx,int sy,int dx,int dy,int w,int h){
    if (s_mask_check) { s_mask_check_ever = 1; s_mask_check_flip = s_flip_count; }
    hd_memo_invalidate(dx & (VRAM_W - 1), dy & (VRAM_H - 1), (dx & (VRAM_W - 1)) + w - 1, (dy & (VRAM_H - 1)) + h - 1);
    if (!s_ctx_ok || w <= 0 || h <= 0) return;
    flush_cpu_upload();   /* the copy source may include pending CPU writes */
    flush_tex_batch(); flush_geometry();
    if (sx + w > VRAM_W) w = VRAM_W - sx;
    if (dx + w > VRAM_W) w = VRAM_W - dx;
    if (sy + h > VRAM_H) h = VRAM_H - sy;
    if (dy + h > VRAM_H) h = VRAM_H - dy;
    if (w <= 0 || h <= 0) return;
    s_perf_cur.copy_rects++;
    int S = s_scale;

    /* stage 1: copy source hr region into scratch at the same coords */
    VkCommandBuffer cb = begin_oneshot();
    vram_to(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    img_to(cb, s_scratch_img, &s_scratch_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageCopy ic = {0};
    ic.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; ic.srcSubresource.layerCount = 1;
    ic.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; ic.dstSubresource.layerCount = 1;
    ic.srcOffset.x = sx * S; ic.srcOffset.y = sy * S;
    ic.dstOffset.x = sx * S; ic.dstOffset.y = sy * S;
    ic.extent.width = w * S; ic.extent.height = h * S; ic.extent.depth = 1;
    p_vkCmdCopyImage(cb, s_vram_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     s_scratch_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
    end_oneshot(cb);

    /* stage 2: blit scratch -> dest (hr-res source: src_div=1, offset (sx-dx)*S). */
    blit_region(dx, dy, w, h, s_scratch_view, s_scratch_img, &s_scratch_layout,
                1, (sx - dx) * S, (sy - dy) * S, 0, 1);

    if (s_vram) {   /* keep CPU mirror coherent for non-GPU readers */
        for (int row = 0; row < h; row++)
            for (int col = 0; col < w; col++)
                s_vram[(dy + row) * VRAM_W + (dx + col)] = s_vram[(sy + row) * VRAM_W + (sx + col)];
    }
}

/* ---- native-wide vtable entry points ------------------------------------ */
/* Enable native-wide with a wide width + centering offset (native px), or
 * disable (wide_w <= 0). No-ops when nothing changed (gpu.c calls this on
 * every draw-area set via ws_nw_sync_target — Part-A lesson: the wide entry
 * points sit inside guest emulation, keep the common path free). */
static void vkb_wide_set_view(int enabled, int shift, int pad_left, int pad_right) {
    if (!enabled) shift = pad_left = pad_right = 0;
    if (view_enabled == enabled && view_shift == shift &&
        view_pad_left == pad_left && view_pad_right == pad_right) return;
    flush_tex_batch(); flush_geometry();
    view_enabled = enabled;
    view_shift = shift;
    view_pad_left = pad_left;
    view_pad_right = pad_right;
}

static void vkb_wide_configure(int wide_w, int offset) {
    if (!s_ready) return;
    if (wide_w == s_wide_w && offset == s_wide_offset) return;
    flush_tex_batch(); flush_geometry();
    if (wide_w <= 0) { wide_free_all(); s_wide_w = 0; s_wide_offset = 0; return; }
    if (wide_w != s_wide_w) wide_free_all();
    s_wide_w = wide_w;
    s_wide_offset = offset;
}

/* Select the wide surface to mirror into for the back buffer at base_x. */
static void vkb_wide_set_target(int base_x) {
    if (!s_ready) { s_wide_cur = -1; return; }
    if (s_wide_cur >= 0 && s_wide_base[s_wide_cur] == base_x) { s_wide_cur_base = base_x; return; }
    flush_tex_batch(); flush_geometry();   /* drain into the OLD target first */
    s_wide_cur = wide_surf_for(base_x);
    s_wide_cur_base = base_x;
}

/* Stop mirroring (offscreen draws that don't target a framebuffer). */
static void vkb_wide_disable_target(void) {
    if (s_wide_cur < 0) return;
    flush_tex_batch(); flush_geometry();
    s_wide_cur = -1;
}

/* Mirror a framebuffer clear: fill the full wide width over [y, y+h) of the
 * surface for base_x (revealed margins stay clean). Stencil clears to bit15,
 * like the canonical fill path. */
static void vkb_wide_clear(int base_x, int y, int h, uint16_t color) {
    if (!s_ready || s_wide_w <= 0) return;
    flush_tex_batch(); flush_geometry();
    int i = wide_surf_for(base_x);
    if (i < 0) return;
    s_perf_cur.wide_clears++;
    int S = s_scale, H = VRAM_H * S;
    int y0 = y * S, y1 = (y + h) * S;
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    if (y1 <= y0) return;
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    color_self_barrier(cb, s_wide_img[i]);
    if (s_msaa > 1) color_self_barrier(cb, s_wide_ms_img[i]);
    s_pipe_ms = s_msaa > 1;
    VkRenderPassBeginInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rp.renderPass = s_msaa > 1 ? s_rpass_ms : s_rpass; rp.framebuffer = s_wide_fb[i];
    rp.renderArea.extent.width = (uint32_t)(s_wide_w * S);
    rp.renderArea.extent.height = (uint32_t)H;
    p_vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
    VkClearAttachment ca[2] = {0};
    ca[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca[0].colorAttachment = 0;
    ca[0].clearValue.color.float32[0] = (float)((color)       & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[1] = (float)((color >> 5)  & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[2] = (float)((color >> 10) & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[3] = (color >> 15) & 1 ? 1.0f : 0.0f;
    ca[1].aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    ca[1].clearValue.depthStencil.depth = 0.0f;
    ca[1].clearValue.depthStencil.stencil = (color >> 15) & 1;
    VkClearRect cr = { { { 0, y0 }, { (uint32_t)(s_wide_w * S), (uint32_t)(y1 - y0) } }, 0, 1 };
    p_vkCmdClearAttachments(cb, 2, ca, 1, &cr);
    p_vkCmdEndRenderPass(cb);
    end_oneshot(cb);
}

/* Clear only the two synthetic reveal strips, leaving the guest-owned centre
 * intact. Mirrors the software/OpenGL opt-in transition cleanup. */
static void vkb_wide_clear_margins(int base_x, int y, int h, uint16_t color, int sides) {
    if (!s_ready || s_wide_w <= 0 || s_wide_offset <= 0) return;
    flush_tex_batch(); flush_geometry();
    int i = wide_surf_for(base_x);
    if (i < 0) return;
    s_perf_cur.wide_clears++;
    int S = s_scale, H = VRAM_H * S;
    int W = s_wide_w * S, margin = s_wide_offset * S;
    int y0 = y * S, y1 = (y + h) * S;
    if (y0 < 0) y0 = 0;
    if (y1 > H) y1 = H;
    if (y1 <= y0 || margin * 2 >= W) return;
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    color_self_barrier(cb, s_wide_img[i]);
    if (s_msaa > 1) color_self_barrier(cb, s_wide_ms_img[i]);
    s_pipe_ms = s_msaa > 1;
    VkRenderPassBeginInfo rp = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rp.renderPass = s_msaa > 1 ? s_rpass_ms : s_rpass; rp.framebuffer = s_wide_fb[i];
    rp.renderArea.extent.width = (uint32_t)W;
    rp.renderArea.extent.height = (uint32_t)H;
    p_vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
    VkClearAttachment ca[2] = {0};
    ca[0].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca[0].colorAttachment = 0;
    ca[0].clearValue.color.float32[0] = (float)((color)       & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[1] = (float)((color >> 5)  & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[2] = (float)((color >> 10) & 0x1F) / 31.0f;
    ca[0].clearValue.color.float32[3] = (color >> 15) & 1 ? 1.0f : 0.0f;
    ca[1].aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    ca[1].clearValue.depthStencil.depth = 0.0f;
    ca[1].clearValue.depthStencil.stencil = (color >> 15) & 1;
    VkClearRect cr[2];
    uint32_t ncr = 0;
    if (sides & 1)
        cr[ncr++] = (VkClearRect){ { { 0, y0 }, { (uint32_t)margin, (uint32_t)(y1 - y0) } }, 0, 1 };
    if (sides & 2)
        cr[ncr++] = (VkClearRect){ { { W - margin, y0 }, { (uint32_t)margin, (uint32_t)(y1 - y0) } }, 0, 1 };
    if (ncr) p_vkCmdClearAttachments(cb, 2, ca, ncr, cr);
    p_vkCmdEndRenderPass(cb);
    end_oneshot(cb);
}

/* Present source (CPU): read the wide surface band for the displayed buffer
 * into an ARGB8888 buffer, byte-compatible with sw_render_wide_display — used
 * by the shared CPU present fallback and the ws debug dump. The GPU-direct
 * vk_renderer_present_wide is the per-frame path; this readback is cold. */
static int vkb_render_wide_display(uint32_t *out, int pitch, int base_x,
                                   int disp_y, int disp_h) {
    if (!s_ready || s_wide_w <= 0) return 0;
    int i = -1;
    for (int k = 0; k < VK_WIDE_MAX_SURF; k++)
        if (s_wide_img[k] && s_wide_base[k] == base_x) { i = k; break; }
    if (i < 0) return 0;
    flush_cpu_upload(); flush_tex_batch(); flush_geometry();
    int S = s_scale, W = s_wide_w * S, H = VRAM_H * S;
    int out_h = disp_h * S, ry0 = disp_y * S;
    if (ry0 < 0) ry0 = 0;
    if (ry0 + out_h > H) out_h = H - ry0;
    if (out_h <= 0) return 0;
    VkBuffer buf; VkDeviceMemory mem; void *map;
    if (!make_staging((VkDeviceSize)W * out_h * 4, &buf, &mem, &map)) return 0;
    VkCommandBuffer cb = begin_oneshot();
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy rc = {0};
    rc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rc.imageSubresource.layerCount = 1;
    rc.imageOffset.y = ry0;
    rc.imageExtent.width = (uint32_t)W; rc.imageExtent.height = (uint32_t)out_h;
    rc.imageExtent.depth = 1;
    p_vkCmdCopyImageToBuffer(cb, s_wide_img[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &rc);
    img_to(cb, s_wide_img[i], &s_wide_layout[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    end_oneshot(cb);
    gpu_sync();   /* readback must be complete before the CPU reads it */
    const uint8_t *src = (const uint8_t *)map;
    int count = 0;
    for (int row = 0; row < out_h; row++) {
        uint32_t *dst = (uint32_t *)((uint8_t *)out + (size_t)row * pitch);
        const uint8_t *s8 = src + (size_t)row * W * 4;
        for (int col = 0; col < W; col++) {   /* RGBA8 -> 0xAARRGGBB, alpha forced */
            dst[col] = 0xFF000000u | ((uint32_t)s8[col * 4 + 0] << 16)
                     | ((uint32_t)s8[col * 4 + 1] << 8) | s8[col * 4 + 2];
            count++;
        }
    }
    free_staging(buf, mem);
    return count;
}

/* ---- 60 fps frame generation (draw-list interpolation) ------------------- */
/* Titles that draw a new picture every second vblank (30 fps) get one
 * generated picture in between. Every draw call of a guest frame is recorded;
 * at the display flip (GP1 05h) the finished list N is replayed with each
 * primitive moved halfway back toward its match in list N-1, into a private
 * wide surface. The next present shows that surface, the one after shows the
 * real frame N, so real frames are delayed by one vblank.
 *
 * vk_framegen_match.h finds where every corner was one frame earlier. When too
 * little matches (a camera cut) no picture is generated.
 *
 * Guest-visible state is never touched: the replay draws only the wide mirror
 * pass (s_fg_replay), never the VRAM image or the CPU mirror. It therefore
 * works only while the native-wide target is active (3D scenes in widescreen).
 */
#define FG_BASE 0x40000000     /* wide-surface key of the generated picture */
#include "vk_framegen_match.h"

static FgList s_fg_list[2];
static int s_fg_cur;                 /* index of the list being recorded */
static int s_fg_on = -1;
static unsigned s_fg_stat_flips, s_fg_stat_made, s_fg_stat_prims, s_fg_stat_matched;
static unsigned s_fg_stat_how[FG_HOW_FIELD + 1];

static int fg_enabled(void) {
    if (s_fg_on < 0) {
        const char *e = getenv("PSX_VK_FRAMEGEN");
        s_fg_on = (e && *e) ? (*e >= '0' && *e <= '3' ? *e - '0' : 1) : -2;   /* -2 = follow the settings panel */
    }
    return s_fg_on == -2 ? touch_ui_framegen() : s_fg_on;
}

static void fg_state_get(FgState *s) {
    s->semi_en = s_semi_en; s->semi_mode = s_semi_mode;
    s->mask_set = s_mask_set; s->mask_check = s_mask_check;
    s->tw[0] = s_tw_mask_x; s->tw[1] = s_tw_mask_y; s->tw[2] = s_tw_off_x; s->tw[3] = s_tw_off_y;
    s->mod[0] = s_mod_r; s->mod[1] = s_mod_g; s->mod[2] = s_mod_b; s->mod[3] = s_mod_raw;
    s->da[0] = s_da_x1; s->da[1] = s_da_y1; s->da[2] = s_da_x2; s->da[3] = s_da_y2;
    s->off[0] = s_off_x; s->off[1] = s_off_y;
    s->view[0] = view_enabled; s->view[1] = view_shift;
    s->view[2] = view_pad_left; s->view[3] = view_pad_right;
    s->wide_on = s_wide_cur >= 0; s->wide_base = s_wide_cur_base;
}

/* Batches must be drained before this. */
static void fg_state_set(const FgState *s) {
    s_semi_en = s->semi_en; s_semi_mode = s->semi_mode;
    s_mask_set = s->mask_set; s_mask_check = s->mask_check;
    s_tw_mask_x = s->tw[0]; s_tw_mask_y = s->tw[1]; s_tw_off_x = s->tw[2]; s_tw_off_y = s->tw[3];
    s_mod_r = s->mod[0]; s_mod_g = s->mod[1]; s_mod_b = s->mod[2]; s_mod_raw = s->mod[3];
    s_da_x1 = s->da[0]; s_da_y1 = s->da[1]; s_da_x2 = s->da[2]; s_da_y2 = s->da[3];
    s_off_x = s->off[0]; s_off_y = s->off[1];
    view_enabled = s->view[0]; view_shift = s->view[1];
    view_pad_left = s->view[2]; view_pad_right = s->view[3];
}

static void fg_reset(void) {
    for (int i = 0; i < 2; i++) {
        s_fg_list[i].n = s_fg_list[i].prims = s_fg_list[i].wide_prims = 0;
        s_fg_list[i].overflow = 0;
    }
    s_fg_hold = 0;
    s_fg_flow_last.ok = 0;
}

static FgCmd *fg_rec(int op) {
    if (s_fg_replay || !s_ready || !fg_enabled()) return NULL;
    FgList *l = &s_fg_list[s_fg_cur];
    if (!l->c) {
        l->c = (FgCmd *)malloc(sizeof(FgCmd) * FG_MAX_CMDS);
        if (!l->c) { s_fg_on = 0; return NULL; }
    }
    if (l->n == 0) fg_state_get(&l->init);
    if (l->n >= FG_MAX_CMDS) { l->overflow = 1; return NULL; }
    FgCmd *c = &l->c[l->n++];
    c->op = (uint8_t)op; c->used = 0; c->pc_valid = 0; c->pq_valid = 0; c->bd = 0; c->moved = 0; c->how = FG_HOW_NONE;
    c->iz[0] = c->iz[1] = c->iz[2] = 0.0f;
    c->offx = (int16_t)s_off_x; c->offy = (int16_t)s_off_y;
    c->key = c->loose = 0; c->next = -1;
    return c;
}

static void fg_rec_prim(int op, const int32_t *args) {
    FgCmd *c = fg_rec(op);
    if (!c) return;
    const int n = kFgArgs[op];
    c->bd = (uint8_t)vk_bd_bits();
    memcpy(c->a, args, sizeof(int32_t) * (size_t)n);
    if (kFgPos[op].nv == 3) {
        c->pc_valid = (uint8_t)s_pc_valid; c->pq_valid = (uint8_t)s_pq_valid;
        for (int i = 0; i < 3; i++) {
            c->pcx[i] = s_pc_x[i]; c->pcy[i] = s_pc_y[i]; c->pq[i] = s_pq[i];
            if (s_pq_valid && s_pq_scale > 0.0f) c->iz[i] = s_pq[i] * s_pq_scale;
        }
    }
    fg_set_keys(c);
    FgList *l = &s_fg_list[s_fg_cur];
    l->prims++;
    if (s_wide_cur >= 0) l->wide_prims++;
}

/* A clipped face is recorded whole: its homogeneous corners interpolate, the
 * clipped polygon of two frames does not. a[] as FG_STT; a[18] = flags. */
static void fg_rec_projected(const PSXProjectedVertex v[3], uint16_t texpage, uint16_t cx, uint16_t cy,
                             int raw, int semi, int perspective) {
    int32_t a[19];
    for (int i = 0; i < 3; i++) {
        a[i*5] = a[i*5 + 1] = 0;
        a[i*5 + 2] = (int32_t)v[i].u; a[i*5 + 3] = (int32_t)v[i].v;
        a[i*5 + 4] = (int32_t)(v[i].r * 255.0 + 0.5) | (int32_t)(v[i].g * 255.0 + 0.5) << 8 | (int32_t)(v[i].b * 255.0 + 0.5) << 16;
    }
    a[15] = cx; a[16] = cy; a[17] = texpage;
    a[18] = (raw ? 1 : 0) | (perspective ? 2 : 0) | (semi + 1) << 2;
    FgCmd *c = fg_rec(FG_PROJ);
    if (!c) return;
    memcpy(c->a, a, sizeof a);
    for (int i = 0; i < 3; i++) {     /* relative to the draw offset, like fg_corner */
        c->pcx[i] = (float)(v[i].x - (double)c->offx * v[i].z);
        c->pcy[i] = (float)(v[i].y - (double)c->offy * v[i].z);
        c->pq[i] = (float)v[i].z;
    }
    fg_set_keys(c);
    FgList *l = &s_fg_list[s_fg_cur];
    l->prims++;
    if (s_wide_cur >= 0) l->wide_prims++;
}

static void fg_rec_set(int op, int a0, int a1, int a2, int a3) {
    FgCmd *c = fg_rec(op);
    if (c) { c->a[0] = a0; c->a[1] = a1; c->a[2] = a2; c->a[3] = a3; }
}

/* Recording entry points (the backend vtable points here). */
static void fgw_set_semi_transparency(int e, int m) { fg_rec_set(FG_SEMI, e, m, 0, 0); vkb_set_semi_transparency(e, m); }
static void fgw_set_mask_bits(int s, int c) { fg_rec_set(FG_MASK, s, c, 0, 0); vkb_set_mask_bits(s, c); }
static void fgw_set_texture_window(uint32_t raw) { fg_rec_set(FG_TWIN, (int)raw, 0, 0, 0); vkb_set_texture_window(raw); }
static void fgw_set_color_modulation(int r, int g, int b, int raw) { fg_rec_set(FG_MOD, r, g, b, raw); vkb_set_color_modulation(r, g, b, raw); }
static void fgw_set_draw_area(int x1, int y1, int x2, int y2) { fg_rec_set(FG_DA, x1, y1, x2, y2); vkb_set_draw_area(x1, y1, x2, y2); }
static void fgw_set_draw_offset(int x, int y) { fg_rec_set(FG_OFF, x, y, 0, 0); vkb_set_draw_offset(x, y); }
static void fgw_wide_set_view(int en, int shift, int pl, int pr) { fg_rec_set(FG_VIEW, en, shift, pl, pr); vkb_wide_set_view(en, shift, pl, pr); }
static void fgw_wide_set_target(int base_x) { fg_rec_set(FG_WSET, base_x, 0, 0, 0); vkb_wide_set_target(base_x); }
static void fgw_wide_disable_target(void) { fg_rec_set(FG_WDIS, 0, 0, 0, 0); vkb_wide_disable_target(); }
static void fgw_wide_clear(int base_x, int y, int h, uint16_t color) { fg_rec_set(FG_WCLR, base_x, y, h, color); vkb_wide_clear(base_x, y, h, color); }
static void fgw_wide_clear_margins(int base_x, int y, int h, uint16_t color, int sides) {
    FgCmd *c = fg_rec(FG_WCLRM);
    if (c) { c->a[0] = base_x; c->a[1] = y; c->a[2] = h; c->a[3] = color; c->a[4] = sides; }
    vkb_wide_clear_margins(base_x, y, h, color, sides);
}
static void fgw_wide_configure(int wide_w, int offset) {
    if (wide_w != s_wide_w || offset != s_wide_offset) fg_reset();   /* surfaces are rebuilt */
    vkb_wide_configure(wide_w, offset);
}
static void fgw_draw_flat_triangle(int x0,int y0,int x1,int y1,int x2,int y2,uint16_t c) {
    int32_t a[7] = { x0,y0,x1,y1,x2,y2,c };
    fg_rec_prim(FG_FT, a);
    vkb_draw_flat_triangle(x0,y0,x1,y1,x2,y2,c);
}
static void fgw_draw_gouraud_triangle(int x0,int y0,uint16_t c0,int x1,int y1,uint16_t c1,int x2,int y2,uint16_t c2) {
    int32_t a[9] = { x0,y0,c0,x1,y1,c1,x2,y2,c2 };
    fg_rec_prim(FG_GT, a);
    vkb_draw_gouraud_triangle(x0,y0,c0,x1,y1,c1,x2,y2,c2);
}
static void fgw_draw_textured_triangle(int x0,int y0,int u0,int v0,int x1,int y1,int u1,int v1,
                                       int x2,int y2,int u2,int v2,uint16_t cx,uint16_t cy,uint16_t tp) {
    int32_t a[15] = { x0,y0,u0,v0,x1,y1,u1,v1,x2,y2,u2,v2,cx,cy,tp };
    fg_rec_prim(FG_TT, a);
    vkb_draw_textured_triangle(x0,y0,u0,v0,x1,y1,u1,v1,x2,y2,u2,v2,cx,cy,tp);
}
static void fgw_draw_shaded_textured_triangle(int x0,int y0,int u0,int v0,uint32_t c0,
                                              int x1,int y1,int u1,int v1,uint32_t c1,
                                              int x2,int y2,int u2,int v2,uint32_t c2,
                                              uint16_t cx,uint16_t cy,uint16_t tp,int raw) {
    int32_t a[19] = { x0,y0,u0,v0,(int32_t)c0,x1,y1,u1,v1,(int32_t)c1,x2,y2,u2,v2,(int32_t)c2,cx,cy,tp,raw };
    fg_rec_prim(FG_STT, a);
    vkb_draw_shaded_textured_triangle(x0,y0,u0,v0,c0,x1,y1,u1,v1,c1,x2,y2,u2,v2,c2,cx,cy,tp,raw);
}
static void fgw_draw_flat_rect(int x,int y,int w,int h,uint16_t c) {
    int32_t a[5] = { x,y,w,h,c };
    fg_rec_prim(FG_FR, a);
    vkb_draw_flat_rect(x,y,w,h,c);
}
static void fgw_draw_textured_rect(int x,int y,int w,int h,int u,int v,uint16_t cx,uint16_t cy,uint16_t tp) {
    int32_t a[9] = { x,y,w,h,u,v,cx,cy,tp };
    fg_rec_prim(FG_TR, a);
    vkb_draw_textured_rect(x,y,w,h,u,v,cx,cy,tp);
}
static void fgw_draw_textured_rect_scaled(int x,int y,int w,int h,int u0,int v0,int u1,int v1,
                                          uint16_t cx,uint16_t cy,uint16_t tp) {
    int32_t a[11] = { x,y,w,h,u0,v0,u1,v1,cx,cy,tp };
    fg_rec_prim(FG_TRS, a);
    vkb_draw_textured_rect_scaled(x,y,w,h,u0,v0,u1,v1,cx,cy,tp);
}
static void fgw_draw_line(int x0,int y0,int x1,int y1,uint16_t c) {
    int32_t a[5] = { x0,y0,x1,y1,c };
    fg_rec_prim(FG_LN, a);
    vkb_draw_line(x0,y0,x1,y1,c);
}
static void fgw_draw_shaded_line(int x0,int y0,uint16_t c0,int x1,int y1,uint16_t c1) {
    int32_t a[6] = { x0,y0,c0,x1,y1,c1 };
    fg_rec_prim(FG_SLN, a);
    vkb_draw_shaded_line(x0,y0,c0,x1,y1,c1);
}

/* Diagnostic (PSX_VK_FG_CSV=<file>): one line per frame saying how the motion
 * of every primitive was found and how much screen each class covers. "field"
 * is the guess: motion taken from the surroundings. */
static float fg_area(const FgCmd *c) {
    const int nv = kFgPos[c->op].nv;
    if (nv == 3) {
        const float x0 = (float)c->a[kFgPos[c->op].xi[0]], y0 = (float)c->a[kFgPos[c->op].yi[0]];
        const float x1 = (float)c->a[kFgPos[c->op].xi[1]], y1 = (float)c->a[kFgPos[c->op].yi[1]];
        const float x2 = (float)c->a[kFgPos[c->op].xi[2]], y2 = (float)c->a[kFgPos[c->op].yi[2]];
        const float a = fabsf((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5f;
        return a > 76800.0f ? 76800.0f : a;
    }
    if (c->op == FG_FR || c->op == FG_TR || c->op == FG_TRS) return fabsf((float)c->a[2] * (float)c->a[3]);
    return 0.0f;
}
static void fg_csv(const FgList *cur, int matched, int made) {
    static const char *const names[] = { "none", "still", "camera", "anchor", "mesh", "seed", "loose", "corner", "field" };
    static FILE *f; static int tried;
    if (!tried) { tried = 1; const char *e = getenv("PSX_VK_FG_CSV"); if (e && *e) { f = fopen(e, "w");
        if (f) {
            fprintf(f, "flip,made,prims,matched,camera_ok,camera_fit,camera_of,bd_dx,bd_dy");
            for (int k = 0; k <= FG_HOW_FIELD; k++) fprintf(f, ",n_%s,a_%s", names[k], names[k]);
            fprintf(f, "\n");
        } } }
    if (!f) return;
    int n[FG_HOW_FIELD + 1] = {0}; float a[FG_HOW_FIELD + 1] = {0};
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (c->op < FG_FIRST_PRIM || fg_is_backdrop(c)) continue;
        n[c->how]++; a[c->how] += fg_area(c);
    }
    fprintf(f, "%u,%d,%d,%d,%d,%d,%d,%.2f,%.2f", s_flip_count, made, cur->prims, matched,
            s_fg_flow_last.ok, s_fg_flow_last.inliers, s_fg_flow_last.n, s_fg_bd_dx, s_fg_bd_dy);
    for (int k = 0; k <= FG_HOW_FIELD; k++) fprintf(f, ",%d,%.0f", n[k], a[k]);
    fprintf(f, "\n");
    if ((s_flip_count & 31) == 0) fflush(f);
}

/* Draw `c` back from where it is by the part of its motion that has not
 * happened yet at time t. `p`: its partner (FG_PROJ, screen-fixed shapes).
 * bd_still: backdrop tiles stay where the new frame has them (see fg_underpaint). */
static void fg_emit(const FgCmd *c, const FgCmd *p, int bd_still) {
    int32_t a[20];
    memcpy(a, c->a, sizeof a);
    const int op = c->op, nv = kFgPos[op].nv;
    const float back = 1.0f - s_fg_t;
    s_fg_bd_bits = c->bd;
    s_pc_valid = 0; s_pq_valid = 0;
    if (op == FG_PROJ) {
        PSXProjectedVertex v[3];
        for (int i = 0; i < 3; i++) {
            float x = c->pcx[i], y = c->pcy[i], z = c->pq[i];
            if (p) { x -= (x - p->pcx[i]) * back; y -= (y - p->pcy[i]) * back; z -= (z - p->pq[i]) * back; }
            v[i].x = (double)x + (double)c->offx * z; v[i].y = (double)y + (double)c->offy * z; v[i].z = z;
            v[i].u = a[i*5 + 2]; v[i].v = a[i*5 + 3];
            v[i].r = (a[i*5 + 4] & 255) / 255.0; v[i].g = ((a[i*5 + 4] >> 8) & 255) / 255.0; v[i].b = ((a[i*5 + 4] >> 16) & 255) / 255.0;
        }
        projected_triangle(v, (uint16_t)a[17], (uint16_t)a[15], (uint16_t)a[16], a[18] & 1, ((a[18] >> 2) & 7) - 1, a[18] & 2);
        return;
    }
    float dx[3] = { 0, 0, 0 }, dy[3] = { 0, 0, 0 };
    if (fg_is_backdrop(c)) {
        /* One rigid layer; a shape that sits at exactly the same place in both
         * frames (letterbox bars, fades) is fixed to the screen instead. */
        if (!bd_still && s_fg_bd_have && !(p && fg_bd_fixed(c, p)))
            for (int i = 0; i < 3; i++) { dx[i] = s_fg_bd_dx; dy[i] = s_fg_bd_dy; }
    } else if (c->moved) {
        for (int i = 0; i < nv; i++) { dx[i] = c->vdx[i]; dy[i] = c->vdy[i]; }
    }
    if (nv == 3) {
        s_pq_valid = c->pq_valid;
        for (int i = 0; i < 3; i++) {
            const int xi = kFgPos[op].xi[i], yi = kFgPos[op].yi[i];
            s_pq[i] = c->pq[i];
            s_pc_x[i] = (c->pc_valid ? c->pcx[i] : (float)c->a[xi]) - dx[i] * back;
            s_pc_y[i] = (c->pc_valid ? c->pcy[i] : (float)c->a[yi]) - dy[i] * back;
            a[xi] = (int32_t)floorf(s_pc_x[i]); a[yi] = (int32_t)floorf(s_pc_y[i]);
        }
        s_pc_valid = 1;
    } else {
        for (int i = 0; i < nv; i++) {
            a[kFgPos[op].xi[i]] -= (int)floorf(dx[i] * back + 0.5f);
            a[kFgPos[op].yi[i]] -= (int)floorf(dy[i] * back + 0.5f);
        }
    }
    switch (op) {
    case FG_FT:  vkb_draw_flat_triangle(a[0],a[1],a[2],a[3],a[4],a[5],(uint16_t)a[6]); break;
    case FG_GT:  vkb_draw_gouraud_triangle(a[0],a[1],(uint16_t)a[2],a[3],a[4],(uint16_t)a[5],a[6],a[7],(uint16_t)a[8]); break;
    case FG_TT:  vkb_draw_textured_triangle(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],
                                            (uint16_t)a[12],(uint16_t)a[13],(uint16_t)a[14]); break;
    case FG_STT: vkb_draw_shaded_textured_triangle(a[0],a[1],a[2],a[3],(uint32_t)a[4],a[5],a[6],a[7],a[8],(uint32_t)a[9],
                                                   a[10],a[11],a[12],a[13],(uint32_t)a[14],
                                                   (uint16_t)a[15],(uint16_t)a[16],(uint16_t)a[17],a[18]); break;
    case FG_FR:  vkb_draw_flat_rect(a[0],a[1],a[2],a[3],(uint16_t)a[4]); break;
    case FG_TR:  vkb_draw_textured_rect(a[0],a[1],a[2],a[3],a[4],a[5],(uint16_t)a[6],(uint16_t)a[7],(uint16_t)a[8]); break;
    case FG_TRS: vkb_draw_textured_rect_scaled(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
                                               (uint16_t)a[8],(uint16_t)a[9],(uint16_t)a[10]); break;
    case FG_LN:  vkb_draw_line(a[0],a[1],a[2],a[3],(uint16_t)a[4]); break;
    case FG_SLN: vkb_draw_shaded_line(a[0],a[1],(uint16_t)a[2],a[3],a[4],(uint16_t)a[5]); break;
    default: break;
    }
}

/* Replay state commands into the generated picture. Returns 0 for a primitive. */
static int fg_replay_state(const FgCmd *c, int clears) {
    switch (c->op) {
    case FG_SEMI: vkb_set_semi_transparency(c->a[0], c->a[1]); return 1;
    case FG_MASK: vkb_set_mask_bits(c->a[0], c->a[1]); return 1;
    case FG_TWIN: vkb_set_texture_window((uint32_t)c->a[0]); return 1;
    case FG_MOD:  vkb_set_color_modulation(c->a[0], c->a[1], c->a[2], c->a[3]); return 1;
    case FG_DA:   vkb_set_draw_area(c->a[0], c->a[1], c->a[2], c->a[3]); return 1;
    case FG_OFF:  vkb_set_draw_offset(c->a[0], c->a[1]); return 1;
    case FG_VIEW: vkb_wide_set_view(c->a[0], c->a[1], c->a[2], c->a[3]); return 1;
    case FG_WSET:
        flush_tex_batch(); flush_geometry();
        s_wide_cur = s_fg_surf; s_wide_cur_base = c->a[0];
        return 1;
    case FG_WDIS:
        flush_tex_batch(); flush_geometry();
        s_wide_cur = -1;
        return 1;
    case FG_WCLR:  if (clears) vkb_wide_clear(c->a[0], c->a[1], c->a[2], (uint16_t)c->a[3]); return 1;
    case FG_WCLRM: if (clears) vkb_wide_clear_margins(c->a[0], c->a[1], c->a[2], (uint16_t)c->a[3], c->a[4]); return 1;
    default: return 0;
    }
}
/* The game draws just enough sky and sea to fill the screen. Shifting that
 * layer back uncovers a strip along the edge it came from, black in every
 * other picture. Lay the new frame's opaque backdrop down unshifted first:
 * the strip then shows sea, and everything else is drawn over. */
static void fg_underpaint(const FgList *cur, int from) {
    FgState at; fg_state_get(&at);
    const int wide = s_wide_cur, base = s_wide_cur_base;
    for (int i = from; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (fg_replay_state(c, 0)) continue;
        if (!fg_is_backdrop(c) || !fg_is_textured(c->op)) break;      /* the layer comes first, in one run */
        if (!s_semi_en) fg_emit(c, NULL, 1);
    }
    flush_tex_batch(); flush_geometry();
    fg_state_set(&at);
    s_wide_cur = wide; s_wide_cur_base = base;
}

/* Render the picture halfway between `prev` and `cur` into the private wide
 * surface. Returns 1 when it was drawn. */
static int fg_build(FgList *prev, FgList *cur, int k, float t) {
    flush_cpu_upload();
    flush_tex_batch(); flush_geometry();
    if (s_fg_surfs[k] < 0 || !s_wide_img[s_fg_surfs[k]]) s_fg_surfs[k] = wide_surf_for(FG_BASE + k);
    s_fg_surf = s_fg_surfs[k];
    if (s_fg_surf < 0) return 0;
    s_fg_t = t;

    FgState live;
    fg_state_get(&live);
    const int live_cur = s_wide_cur;
    const int live_pc = s_pc_valid, live_pq = s_pq_valid;
    float live_pcx[3], live_pcy[3], live_q[3];
    for (int i = 0; i < 3; i++) { live_pcx[i] = s_pc_x[i]; live_pcy[i] = s_pc_y[i]; live_q[i] = s_pq[i]; }

    s_fg_replay = 1;
    fg_state_set(&cur->init);
    s_wide_cur = cur->init.wide_on ? s_fg_surf : -1;
    s_wide_cur_base = cur->init.wide_base;
    int under = s_fg_bd_have && (fabsf(s_fg_bd_dx) > 0.01f || fabsf(s_fg_bd_dy) > 0.01f);
    for (int i = 0; i < cur->n; i++) {
        const FgCmd *c = &cur->c[i];
        if (fg_replay_state(c, 1)) continue;
        if (under && fg_is_backdrop(c) && fg_is_textured(c->op)) { under = 0; fg_underpaint(cur, i); }
        fg_emit(c, c->used && c->next >= 0 ? &prev->c[c->next] : NULL, 0);
    }
    flush_tex_batch(); flush_geometry();
    s_fg_replay = 0;

    fg_state_set(&live);
    s_wide_cur = live_cur; s_wide_cur_base = live.wide_base;
    s_pc_valid = live_pc; s_pq_valid = live_pq;
    for (int i = 0; i < 3; i++) { s_pc_x[i] = live_pcx[i]; s_pc_y[i] = live_pcy[i]; s_pq[i] = live_q[i]; }
    return 1;
}

/* 120 needs a display that refreshes that fast: pictures queue on a FIFO
 * swapchain, so a 60 Hz panel would halve the game's speed. */
static int fg_120_ok(void) {
    static int ok = -1;
    if (ok < 0) {
        const char *e = getenv("PSX_VK_FORCE_120");
SDL_DisplayMode dm;   /* psx_sdl.h keeps the SDL2 call shape on SDL3 too */
        const float hz = SDL_GetCurrentDisplayMode(0, &dm) == 0 ? (float)dm.refresh_rate : 0.0f;
        ok = (e && *e == '1') || hz >= 110.0f;
        fprintf(stdout, "psxrecomp: vulkan: display %.1f Hz, 120 fps mode %s\n", hz, ok ? "available" : "not available (needs a 120 Hz display)");
        fflush(stdout);
    }
    return ok && !s_fg_120_off;
}

/* GP1(05h): the guest switched the displayed buffer, so the list recorded
 * since the last switch is a finished frame. */
void vk_renderer_display_flip(void) {
    if (!s_ctx_ok || !s_ready) return;
    s_flip_count++;
    hd_flip_tick();
    if (!fg_enabled()) { if (s_fg_list[0].n || s_fg_list[1].n) fg_reset(); return; }
    FgList *cur = &s_fg_list[s_fg_cur], *prev = &s_fg_list[s_fg_cur ^ 1];
    if (cur->n == 0) return;                      /* nothing drawn: not a frame boundary */
    s_fg_stat_flips++;
    s_fg_hold = 0;
    if (s_wide_w > 0 && !cur->overflow && !prev->overflow &&
        cur->wide_prims >= 24 && prev->wide_prims >= 24) {
        int matched = fg_solve(prev, cur);
        matched += fg_solve_backdrop(prev, cur);
        for (int i = 0; i < cur->n; i++)
            if (cur->c[i].op >= FG_FIRST_PRIM && !fg_is_backdrop(&cur->c[i])) s_fg_stat_how[cur->c[i].how]++;
        s_fg_stat_prims += (unsigned)cur->prims; s_fg_stat_matched += (unsigned)matched;
        const int mode = fg_enabled();
        const int want = (mode >= 2 && fg_120_ok()) ? (mode == 2 ? 2 : 3) : 1;
        frame_pacer_set_pump(fgq_due, fgq_run);
        int made = 0;
        if (matched * 2 >= cur->prims)
            for (; made < want; made++)
                if (!fg_build(prev, cur, made, (float)(made + 1) / (float)(want + 1))) break;
        if (made == want) {
            s_fg_n = want; s_fg_next = 0;
            s_fg_hold = 1;
            s_fg_hold_stamp = s_present_idx;
            s_fg_stat_made++;
        }
        fg_csv(cur, matched, made == want);
    }
    if (s_fg_stat_flips >= 300) {
        const unsigned nf = s_fg_stat_flips;
        fprintf(stdout, "psxrecomp: vulkan: frame generation: %u of %u frames got an in-between picture, %u%% of primitives matched; "
                        "per frame: camera %u, anchors %u, mesh %u, whole mesh %u, scrolled %u, corners %u, guessed %u "
                        "(camera fit %d of %d; last frame: %d cmds, %d prims, %d wide; wide_w %d)\n",
                s_fg_stat_made, nf,
                s_fg_stat_prims ? (unsigned)((uint64_t)s_fg_stat_matched * 100u / s_fg_stat_prims) : 0u,
                s_fg_stat_how[FG_HOW_CAMERA] / nf, (s_fg_stat_how[FG_HOW_ANCHOR] + s_fg_stat_how[FG_HOW_STILL]) / nf,
                s_fg_stat_how[FG_HOW_MESH] / nf, s_fg_stat_how[FG_HOW_SEED] / nf, s_fg_stat_how[FG_HOW_LOOSE] / nf,
                s_fg_stat_how[FG_HOW_CORNER] / nf, s_fg_stat_how[FG_HOW_FIELD] / nf,
                s_fg_flow_last.inliers, s_fg_flow_last.n, cur->n, cur->prims, cur->wide_prims, s_wide_w);
        fflush(stdout);
        s_fg_stat_flips = s_fg_stat_made = s_fg_stat_prims = s_fg_stat_matched = 0;
        memset(s_fg_stat_how, 0, sizeof s_fg_stat_how);
    }
    s_fg_cur ^= 1;                                /* cur becomes prev; old prev is reused */
    s_fg_list[s_fg_cur].n = s_fg_list[s_fg_cur].prims = s_fg_list[s_fg_cur].wide_prims = 0;
    s_fg_list[s_fg_cur].overflow = 0;
}

static const GpuRenderBackend VK_BACKEND = {
    .name                          = "vulkan",
    .init                          = vkb_init,
    .set_scale                     = vkb_set_scale,
    .scale                         = vkb_scale,
    .set_texture_filter            = vkb_set_texture_filter,
    .texture_filter                = vkb_texture_filter,
    .set_semi_transparency         = fgw_set_semi_transparency,
    .set_mask_bits                 = fgw_set_mask_bits,
    .set_texture_window            = fgw_set_texture_window,
    .set_color_modulation          = fgw_set_color_modulation,
    .set_precise_triangle          = vkb_set_precise_triangle,
    .set_perspective_triangle      = vkb_set_perspective_triangle,
    .fill_rect                     = vkb_fill_rect,
    .copy_rect                     = vkb_copy_rect,
    .draw_flat_triangle            = fgw_draw_flat_triangle,
    .draw_gouraud_triangle         = fgw_draw_gouraud_triangle,
    .draw_textured_triangle        = fgw_draw_textured_triangle,
    .draw_shaded_textured_triangle = fgw_draw_shaded_textured_triangle,
    .draw_flat_rect                = fgw_draw_flat_rect,
    .draw_textured_rect            = fgw_draw_textured_rect,
    .draw_textured_rect_scaled     = fgw_draw_textured_rect_scaled,
    .draw_line                     = fgw_draw_line,
    .draw_shaded_line              = fgw_draw_shaded_line,
    .render_display                = vkb_render_display,
    .render_display_hires          = vkb_render_display_hires,
    .vram_write                    = vkb_vram_write,
    .vram_read                     = vkb_vram_read,
    .vram_transfer_in              = vkb_vram_transfer_in,
    .vram_transfer_out             = vkb_vram_transfer_out,
    .set_draw_area                 = fgw_set_draw_area,
    .get_draw_area                 = vkb_get_draw_area,
    .set_draw_offset               = fgw_set_draw_offset,
    .wide_configure                = fgw_wide_configure,
    .wide_set_view = fgw_wide_set_view,
    .wide_set_target               = fgw_wide_set_target,
    .wide_disable_target           = fgw_wide_disable_target,
    .wide_clear                    = fgw_wide_clear,
    .wide_clear_margins            = fgw_wide_clear_margins,
    .render_wide_display           = vkb_render_wide_display,
};

/* Returned unconditionally (like gl_backend_get): the table is selected before
 * the context exists; vk_renderer_init_context creates the device, and on
 * failure main.cpp switches the facade back to software. */
const GpuRenderBackend *vk_backend_get(void) { return &VK_BACKEND; }

#endif /* PSX_HAVE_VULKAN */
