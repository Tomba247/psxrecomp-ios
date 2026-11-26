/* A direct Metal implementation of exactly the Vulkan functions that
 * gpu_vk_renderer.c calls through its p_vk* pointers (PSX_METAL=1). It is not
 * a general Vulkan driver: one queue, 2D images, the renderer's own ten
 * shaders, no synchronisation beyond Metal's own hazard tracking. A use it
 * does not know is reported once and refused, not guessed at.
 * Compiled with -fobjc-arc. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vk_shaders_spv.h"
#include "gpu_metal_shaders.h"      /* kMetalShaders */

#define BAD(...) do { static int said; if (!said) { said = 1; fprintf(stdout, "psxrecomp: metal: " __VA_ARGS__); fputc('\n', stdout); fflush(stdout); } } while (0)
#define O(T, h) ((__bridge T *)(void *)(h))
#define GIVE(T, out, obj) (*(out) = (T)(void *)CFBridgingRetain(obj))
#define DROP(h) do { if (h) CFBridgingRelease((void *)(h)); } while (0)

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_lib;
static id<MTLCommandBuffer> g_last;
static int g_depth_shader;
void psx_metal_set_depth_shader(int on) { g_depth_shader = on; }
extern void touch_ui_metal_ran(void);
static id<MTLSamplerState> g_lin, g_near;

@interface MMem : NSObject { @public id<MTLBuffer> buf; } @end
@implementation MMem @end
@interface MBuf : NSObject { @public id<MTLBuffer> buf; VkDeviceSize off, size; } @end
@implementation MBuf @end
@interface MImg : NSObject { @public id<MTLTexture> tex; VkFormat fmt; uint32_t w, h, levels; int screen; } @end
@implementation MImg @end
/* img: a screen picture, whose texture is another one each time. */
@interface MView : NSObject { @public id<MTLTexture> tex; VkFormat fmt; MImg *img; } @end
@implementation MView @end
static id<MTLTexture> vtex(MView *v) { return v->tex ? v->tex : v->img->tex; }
@interface MSamp : NSObject { @public id<MTLSamplerState> s; } @end
@implementation MSamp @end
@interface MShader : NSObject { @public NSString *name; int comp; } @end
@implementation MShader @end
@interface MSet : NSObject { @public id<MTLTexture> tex[4]; id<MTLSamplerState> samp[4]; } @end
@implementation MSet @end
@interface MPass : NSObject { @public VkAttachmentDescription att[4]; uint32_t n; int color, ds, resolve; } @end
@implementation MPass @end
@interface MFB : NSObject { @public MView *v[4]; uint32_t n, w, h; } @end
@implementation MFB @end
@interface MPipe : NSObject { @public id<MTLRenderPipelineState> rp; id<MTLComputePipelineState> cp; id<MTLDepthStencilState> ds;
    MTLPrimitiveType prim; float bias, blend[4]; int vbuf, pushv, pushf; } @end
@implementation MPipe @end
@interface MFence : NSObject { @public id<MTLCommandBuffer> cb; int signaled; } @end
@implementation MFence @end
@interface MSurf : NSObject { @public CAMetalLayer *layer; } @end
@implementation MSurf @end
@interface MSwap : NSObject { @public MImg *img[3]; uint32_t next, w, h; CAMetalLayer *layer; id<CAMetalDrawable> held[3]; } @end
@implementation MSwap @end
@interface MObj : NSObject @end
@implementation MObj @end
@interface MCmd : NSObject { @public
    id<MTLCommandBuffer> cb; id<MTLRenderCommandEncoder> re; id<MTLBlitCommandEncoder> be;
    MPipe *pipe, *cpipe; MSet *sets[2], *csets[2]; uint8_t push[128]; uint32_t npush;
    MBuf *vb; VkDeviceSize vboff; MTLViewport vp; VkRect2D sc; uint32_t sref;
    uint32_t tw, th; MTLPixelFormat cf, df, sf; NSUInteger samples; }
@end
@implementation MCmd @end

static MTLPixelFormat pf(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return MTLPixelFormatRGBA8Unorm;
    case VK_FORMAT_B8G8R8A8_UNORM: return MTLPixelFormatBGRA8Unorm;
    case VK_FORMAT_R16_UINT: return MTLPixelFormatR16Uint;
    case VK_FORMAT_S8_UINT: return MTLPixelFormatStencil8;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return MTLPixelFormatDepth32Float_Stencil8;
    default: BAD("format %d is not known", (int)f); return MTLPixelFormatInvalid;
    }
}
static uint32_t bpp(VkFormat f) { return f == VK_FORMAT_R16_UINT ? 2 : 4; }
static int has_depth(MTLPixelFormat f) { return f == MTLPixelFormatDepth32Float_Stencil8; }
static int has_stencil(MTLPixelFormat f) { return f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatStencil8; }

/* ---- instance, device ---------------------------------------------------- */
static MObj *g_tok;
static VkResult m_vkEnumerateInstanceExtensionProperties(const char *l, uint32_t *n, VkExtensionProperties *p) { *n = 0; return VK_SUCCESS; }
static VkResult m_vkEnumerateInstanceLayerProperties(uint32_t *n, VkLayerProperties *p) { *n = 0; return VK_SUCCESS; }
static VkResult m_vkCreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a, VkInstance *out) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return VK_ERROR_INITIALIZATION_FAILED;
        g_queue = [g_dev newCommandQueueWithMaxCommandBufferCount:256];
        NSError *e = nil;
        g_lib = [g_dev newLibraryWithSource:@(kMetalShaders) options:[MTLCompileOptions new] error:&e];
        if (!g_lib) { fprintf(stdout, "psxrecomp: metal: shaders: %s\n", e.localizedDescription.UTF8String); return VK_ERROR_INITIALIZATION_FAILED; }
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        g_near = [g_dev newSamplerStateWithDescriptor:sd];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        g_lin = [g_dev newSamplerStateWithDescriptor:sd];
        g_tok = [MObj new];
        *out = (VkInstance)(__bridge void *)g_tok;
        fprintf(stdout, "psxrecomp: metal: direct Metal, no MoltenVK\n"); fflush(stdout);
        return VK_SUCCESS;
    }
}
static void m_vkDestroyInstance(VkInstance i, const VkAllocationCallbacks *a) {}
static VkResult m_vkEnumeratePhysicalDevices(VkInstance i, uint32_t *n, VkPhysicalDevice *d) { if (d && *n) d[0] = (VkPhysicalDevice)i; *n = 1; return VK_SUCCESS; }
static void m_vkGetPhysicalDeviceProperties(VkPhysicalDevice d, VkPhysicalDeviceProperties *p) {
    memset(p, 0, sizeof *p);
    p->apiVersion = VK_API_VERSION_1_1; p->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    snprintf(p->deviceName, sizeof p->deviceName, "%s (Metal)", g_dev.name.UTF8String);
    VkSampleCountFlags s = 1;
    if ([g_dev supportsTextureSampleCount:2]) s |= 2;
    if ([g_dev supportsTextureSampleCount:4]) s |= 4;
    p->limits.framebufferColorSampleCounts = p->limits.framebufferDepthSampleCounts = p->limits.framebufferStencilSampleCounts = s;
    p->limits.maxImageDimension2D = p->limits.maxFramebufferWidth = p->limits.maxFramebufferHeight = 16384;
    p->limits.maxPushConstantsSize = 128; p->limits.maxSamplerAnisotropy = 16;
}
static void m_vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice d, uint32_t *n, VkQueueFamilyProperties *p) {
    if (p && *n) { memset(p, 0, sizeof *p); p->queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT; p->queueCount = 1; }
    *n = 1;
}
static void m_vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice d, VkPhysicalDeviceMemoryProperties *p) {
    memset(p, 0, sizeof *p);
    p->memoryHeapCount = 1; p->memoryHeaps[0].size = (VkDeviceSize)4 << 30; p->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    p->memoryTypeCount = 2;
    p->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    p->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
}
static void m_vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice d, VkFormat f, VkFormatProperties *p) {
    memset(p, 0, sizeof *p);
    if (f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT) p->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    else if (f == VK_FORMAT_R8G8B8A8_UNORM || f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_R16_UINT) p->optimalTilingFeatures = ~0u & ~VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
}
static VkResult m_vkCreateDevice(VkPhysicalDevice d, const VkDeviceCreateInfo *c, const VkAllocationCallbacks *a, VkDevice *out) { *out = (VkDevice)d; return VK_SUCCESS; }
static void m_vkDestroyDevice(VkDevice d, const VkAllocationCallbacks *a) {}
static void m_vkGetDeviceQueue(VkDevice d, uint32_t f, uint32_t i, VkQueue *q) { *q = (VkQueue)d; }

/* ---- surface, swapchain (headless pictures; the screen comes later) ------ */
static VkResult m_vkCreateHeadlessSurfaceEXT(VkInstance i, const VkHeadlessSurfaceCreateInfoEXT *c, const VkAllocationCallbacks *a, VkSurfaceKHR *s) { GIVE(VkSurfaceKHR, s, [MSurf new]); return VK_SUCCESS; }
VkSurfaceKHR psx_metal_surface(void *layer) { MSurf *s = [MSurf new]; s->layer = (__bridge CAMetalLayer *)layer; VkSurfaceKHR out; GIVE(VkSurfaceKHR, &out, s); return layer ? out : VK_NULL_HANDLE; }
static void m_vkDestroySurfaceKHR(VkInstance i, VkSurfaceKHR s, const VkAllocationCallbacks *a) { DROP(s); }
static VkResult m_vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice d, uint32_t f, VkSurfaceKHR s, VkBool32 *ok) { *ok = VK_TRUE; return VK_SUCCESS; }
static VkResult m_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice d, VkSurfaceKHR s, VkSurfaceCapabilitiesKHR *c) {
    memset(c, 0, sizeof *c); c->minImageCount = 2; c->maxImageCount = 3; c->currentExtent.width = c->currentExtent.height = 0xFFFFFFFFu;
    CAMetalLayer *y = O(MSurf, s)->layer;
    if (y && y.drawableSize.width >= 1) { c->currentExtent.width = (uint32_t)y.drawableSize.width; c->currentExtent.height = (uint32_t)y.drawableSize.height; }
    c->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; c->maxImageArrayLayers = 1; return VK_SUCCESS;
}
static VkResult m_vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice d, VkSurfaceKHR s, uint32_t *n, VkSurfaceFormatKHR *f) {
    if (f && *n) { f[0].format = VK_FORMAT_B8G8R8A8_UNORM; f[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; } *n = 1; return VK_SUCCESS;
}
static VkResult m_vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice d, VkSurfaceKHR s, uint32_t *n, VkPresentModeKHR *m) { if (m && *n) m[0] = VK_PRESENT_MODE_FIFO_KHR; *n = 1; return VK_SUCCESS; }
static MImg *new_img(VkFormat f, uint32_t w, uint32_t h, uint32_t levels, uint32_t samples, MTLTextureUsage u) {
    MTLTextureDescriptor *t = [MTLTextureDescriptor new];
    t.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    t.pixelFormat = pf(f); t.width = w; t.height = h; t.mipmapLevelCount = levels; t.sampleCount = samples;
    t.storageMode = MTLStorageModePrivate; t.usage = u;
    MImg *m = [MImg new]; m->tex = [g_dev newTextureWithDescriptor:t]; m->fmt = f; m->w = w; m->h = h; m->levels = levels;
    return m->tex ? m : nil;
}
static VkResult m_vkCreateSwapchainKHR(VkDevice d, const VkSwapchainCreateInfoKHR *c, const VkAllocationCallbacks *a, VkSwapchainKHR *out) {
    @autoreleasepool {
        MSwap *s = [MSwap new]; s->w = c->imageExtent.width; s->h = c->imageExtent.height; s->layer = O(MSurf, c->surface)->layer;
        if (s->layer) {
            s->layer.device = g_dev; s->layer.pixelFormat = pf(c->imageFormat); s->layer.framebufferOnly = NO; s->layer.maximumDrawableCount = 3; s->layer.opaque = YES;
            for (int i = 0; i < 3; i++) { MImg *m = [MImg new]; m->fmt = c->imageFormat; m->w = s->w; m->h = s->h; m->levels = 1; m->screen = 1; s->img[i] = m; }
            GIVE(VkSwapchainKHR, out, s); return VK_SUCCESS;
        }
        for (int i = 0; i < 3; i++) if (!(s->img[i] = new_img(c->imageFormat, c->imageExtent.width, c->imageExtent.height, 1, 1, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead))) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        GIVE(VkSwapchainKHR, out, s); return VK_SUCCESS;
    }
}
static void m_vkDestroySwapchainKHR(VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks *a) { DROP(s); }
static VkResult m_vkGetSwapchainImagesKHR(VkDevice d, VkSwapchainKHR s, uint32_t *n, VkImage *im) { if (im) for (uint32_t i = 0; i < 3 && i < *n; i++) im[i] = (VkImage)(__bridge void *)O(MSwap, s)->img[i]; *n = 3; return VK_SUCCESS; }
static VkResult m_vkAcquireNextImageKHR(VkDevice d, VkSwapchainKHR s, uint64_t t, VkSemaphore se, VkFence f, uint32_t *i) {
    @autoreleasepool {
        MSwap *w = O(MSwap, s);
        if (w->layer) {
            if ((uint32_t)w->layer.drawableSize.width != w->w || (uint32_t)w->layer.drawableSize.height != w->h) return VK_ERROR_OUT_OF_DATE_KHR;
            id<CAMetalDrawable> dr = [w->layer nextDrawable];
            if (!dr) return VK_NOT_READY;
            w->held[w->next] = dr; w->img[w->next]->tex = dr.texture;
        }
        *i = w->next; w->next = (w->next + 1) % 3; return VK_SUCCESS;
    }
}
static VkResult m_vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR *p) {
    @autoreleasepool {
        MSwap *w = O(MSwap, p->pSwapchains[0]); uint32_t i = p->pImageIndices[0];
        if (!w->layer || !w->held[i]) return VK_SUCCESS;
        id<MTLCommandBuffer> b = [g_queue commandBuffer];
        [b presentDrawable:w->held[i]]; [b commit];
        w->held[i] = nil; w->img[i]->tex = nil;      /* given back to the layer */
        static unsigned shown;
        if (++shown == 300) touch_ui_metal_ran();
        return VK_SUCCESS;
    }
}

/* ---- memory, buffers, images --------------------------------------------- */
static VkResult m_vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo *i, const VkAllocationCallbacks *a, VkDeviceMemory *out) {
    MMem *m = [MMem new];
    if (i->memoryTypeIndex == 1) { m->buf = [g_dev newBufferWithLength:(NSUInteger)(i->allocationSize ? i->allocationSize : 4) options:MTLResourceStorageModeShared]; if (!m->buf) return VK_ERROR_OUT_OF_HOST_MEMORY; }
    GIVE(VkDeviceMemory, out, m); return VK_SUCCESS;
}
static void m_vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { DROP(m); }
static VkResult m_vkMapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize off, VkDeviceSize n, VkMemoryMapFlags f, void **p) { MMem *x = O(MMem, m); if (!x->buf) return VK_ERROR_MEMORY_MAP_FAILED; *p = (uint8_t *)x->buf.contents + off; return VK_SUCCESS; }
static void m_vkUnmapMemory(VkDevice d, VkDeviceMemory m) {}
static VkResult m_vkCreateBuffer(VkDevice d, const VkBufferCreateInfo *c, const VkAllocationCallbacks *a, VkBuffer *out) { MBuf *b = [MBuf new]; b->size = c->size; GIVE(VkBuffer, out, b); return VK_SUCCESS; }
static void m_vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { DROP(b); }
static void m_vkGetBufferMemoryRequirements(VkDevice d, VkBuffer b, VkMemoryRequirements *r) { r->size = (O(MBuf, b)->size + 255) & ~(VkDeviceSize)255; r->alignment = 256; r->memoryTypeBits = 2; }
static VkResult m_vkBindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) { MBuf *x = O(MBuf, b); x->buf = O(MMem, m)->buf; x->off = off; return VK_SUCCESS; }
static VkResult m_vkCreateImage(VkDevice d, const VkImageCreateInfo *c, const VkAllocationCallbacks *a, VkImage *out) {
    @autoreleasepool {
        MTLTextureUsage u = MTLTextureUsageShaderRead;
        if (!has_stencil(pf(c->format)) && c->format != VK_FORMAT_R16_UINT) u |= MTLTextureUsageRenderTarget;
        if (has_stencil(pf(c->format)) || (c->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) || c->format == VK_FORMAT_R16_UINT) u |= MTLTextureUsageRenderTarget;
        if (c->usage & VK_IMAGE_USAGE_STORAGE_BIT) u |= MTLTextureUsageShaderWrite;
        MImg *m = new_img(c->format, c->extent.width, c->extent.height, c->mipLevels, c->samples, u);
        if (!m) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        GIVE(VkImage, out, m); return VK_SUCCESS;
    }
}
static void m_vkDestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks *a) { DROP(i); }
static void m_vkGetImageMemoryRequirements(VkDevice d, VkImage i, VkMemoryRequirements *r) { r->size = 4096; r->alignment = 4096; r->memoryTypeBits = 1; }
static VkResult m_vkBindImageMemory(VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize off) { return VK_SUCCESS; }
static VkResult m_vkCreateImageView(VkDevice d, const VkImageViewCreateInfo *c, const VkAllocationCallbacks *a, VkImageView *out) {
    MView *v = [MView new]; MImg *m = O(MImg, c->image); v->fmt = m->fmt;
    uint32_t n = c->subresourceRange.levelCount == VK_REMAINING_MIP_LEVELS ? m->levels - c->subresourceRange.baseMipLevel : c->subresourceRange.levelCount;
    v->img = m;
    v->tex = m->screen ? nil : n == m->levels ? m->tex : [m->tex newTextureViewWithPixelFormat:m->tex.pixelFormat textureType:m->tex.textureType levels:NSMakeRange(c->subresourceRange.baseMipLevel, n) slices:NSMakeRange(0, 1)];
    GIVE(VkImageView, out, v); return VK_SUCCESS;
}
static void m_vkDestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks *a) { DROP(v); }
static VkResult m_vkCreateSampler(VkDevice d, const VkSamplerCreateInfo *c, const VkAllocationCallbacks *a, VkSampler *out) {
    MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
    s.minFilter = c->minFilter == VK_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    s.magFilter = c->magFilter == VK_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    s.mipFilter = c->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    s.lodMinClamp = c->minLod; s.lodMaxClamp = c->maxLod > 1000 ? 1000 : c->maxLod;
    if (c->anisotropyEnable) s.maxAnisotropy = (NSUInteger)(c->maxAnisotropy < 1 ? 1 : c->maxAnisotropy);
    MTLSamplerAddressMode m = c->addressModeU == VK_SAMPLER_ADDRESS_MODE_REPEAT ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
    s.sAddressMode = s.tAddressMode = m;
    MSamp *x = [MSamp new]; x->s = [g_dev newSamplerStateWithDescriptor:s];
    GIVE(VkSampler, out, x); return VK_SUCCESS;
}
static void m_vkDestroySampler(VkDevice d, VkSampler s, const VkAllocationCallbacks *a) { DROP(s); }

/* ---- shaders, descriptors, passes ---------------------------------------- */
static VkResult m_vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *c, const VkAllocationCallbacks *a, VkShaderModule *out) {
    static const struct { const uint32_t *code; uint32_t size; const char *name; } k[] = {
        {spv_geo_vert, sizeof spv_geo_vert, "geo_vert"}, {spv_geo_frag, sizeof spv_geo_frag, "geo_frag"},
        {spv_geo_tex_vert, sizeof spv_geo_tex_vert, "geo_tex_vert"}, {spv_geo_tex_frag, sizeof spv_geo_tex_frag, "geo_tex_frag"},
        {spv_blit_vert, sizeof spv_blit_vert, "blit_vert"}, {spv_blit_frag, sizeof spv_blit_frag, "blit_frag"},
        {spv_present_vert, sizeof spv_present_vert, "present_vert"}, {spv_present_frag, sizeof spv_present_frag, "present_frag"},
        {spv_overlay_frag, sizeof spv_overlay_frag, "overlay_frag"}, {spv_pack_comp, sizeof spv_pack_comp, "pack_comp"} };
    for (unsigned i = 0; i < sizeof k / sizeof *k; i++) {
        if (k[i].size != c->codeSize) continue;
        unsigned diff = 0;      /* the depth variant of geo_tex_vert differs in one word */
        for (unsigned w = 0; w < k[i].size / 4; w++) diff += k[i].code[w] != c->pCode[w];
        if (diff > 1) continue;
        MShader *s = [MShader new]; s->name = @(k[i].name); GIVE(VkShaderModule, out, s); return VK_SUCCESS;
    }
    BAD("a shader of %zu bytes is not one of the renderer's", c->codeSize);
    return VK_ERROR_INITIALIZATION_FAILED;
}
static void m_vkDestroyShaderModule(VkDevice d, VkShaderModule s, const VkAllocationCallbacks *a) { DROP(s); }
#define TOKEN(Name, Info, T) \
static VkResult m_vkCreate##Name(VkDevice d, const Info *c, const VkAllocationCallbacks *a, T *out) { GIVE(T, out, [MObj new]); return VK_SUCCESS; } \
static void m_vkDestroy##Name(VkDevice d, T h, const VkAllocationCallbacks *a) { DROP(h); }
TOKEN(DescriptorSetLayout, VkDescriptorSetLayoutCreateInfo, VkDescriptorSetLayout)
TOKEN(PipelineLayout, VkPipelineLayoutCreateInfo, VkPipelineLayout)
TOKEN(Semaphore, VkSemaphoreCreateInfo, VkSemaphore)
@interface MPool : NSObject { @public NSMutableArray *sets; } @end
@implementation MPool @end
static VkResult m_vkCreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo *c, const VkAllocationCallbacks *a, VkDescriptorPool *out) { MPool *p = [MPool new]; p->sets = [NSMutableArray new]; GIVE(VkDescriptorPool, out, p); return VK_SUCCESS; }
static void m_vkDestroyDescriptorPool(VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks *a) { DROP(p); }
static VkResult m_vkAllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo *i, VkDescriptorSet *out) {
    for (uint32_t k = 0; k < i->descriptorSetCount; k++) { MSet *s = [MSet new]; [O(MPool, i->descriptorPool)->sets addObject:s]; out[k] = (VkDescriptorSet)(__bridge void *)s; }
    return VK_SUCCESS;
}
static void m_vkUpdateDescriptorSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    for (uint32_t k = 0; k < n; k++) {
        if (!w[k].pImageInfo || w[k].dstBinding > 3) { BAD("descriptor of an unknown kind"); continue; }
        MSet *s = O(MSet, w[k].dstSet); const VkDescriptorImageInfo *ii = w[k].pImageInfo;
        s->tex[w[k].dstBinding] = ii->imageView ? vtex(O(MView, ii->imageView)) : nil;
        if (ii->sampler) s->samp[w[k].dstBinding] = O(MSamp, ii->sampler)->s;
    }
}
static VkResult m_vkCreateRenderPass(VkDevice d, const VkRenderPassCreateInfo *c, const VkAllocationCallbacks *a, VkRenderPass *out) {
    if (c->attachmentCount > 4 || c->subpassCount != 1) { BAD("render pass of an unknown shape"); return VK_ERROR_INITIALIZATION_FAILED; }
    MPass *p = [MPass new]; p->n = c->attachmentCount; memcpy(p->att, c->pAttachments, sizeof(VkAttachmentDescription) * p->n);
    const VkSubpassDescription *s = c->pSubpasses;
    p->color = s->colorAttachmentCount ? (int)s->pColorAttachments[0].attachment : -1;
    p->ds = s->pDepthStencilAttachment ? (int)s->pDepthStencilAttachment->attachment : -1;
    p->resolve = s->pResolveAttachments ? (int)s->pResolveAttachments[0].attachment : -1;
    GIVE(VkRenderPass, out, p); return VK_SUCCESS;
}
static void m_vkDestroyRenderPass(VkDevice d, VkRenderPass p, const VkAllocationCallbacks *a) { DROP(p); }
static VkResult m_vkCreateFramebuffer(VkDevice d, const VkFramebufferCreateInfo *c, const VkAllocationCallbacks *a, VkFramebuffer *out) {
    MFB *f = [MFB new]; f->n = c->attachmentCount > 4 ? 4 : c->attachmentCount; f->w = c->width; f->h = c->height;
    for (uint32_t i = 0; i < f->n; i++) f->v[i] = O(MView, c->pAttachments[i]);
    GIVE(VkFramebuffer, out, f); return VK_SUCCESS;
}
static void m_vkDestroyFramebuffer(VkDevice d, VkFramebuffer f, const VkAllocationCallbacks *a) { DROP(f); }

/* ---- pipelines ------------------------------------------------------------ */
static id<MTLFunction> func(NSString *name) {
    MTLFunctionConstantValues *v = [MTLFunctionConstantValues new]; bool b = g_depth_shader != 0;
    [v setConstantValue:&b type:MTLDataTypeBool atIndex:0];
    NSError *e = nil; id<MTLFunction> f = [g_lib newFunctionWithName:name constantValues:v error:&e];
    if (!f) fprintf(stdout, "psxrecomp: metal: %s: %s\n", name.UTF8String, e.localizedDescription.UTF8String);
    return f;
}
static MTLBlendFactor bf(VkBlendFactor f) {
    switch (f) {
    case VK_BLEND_FACTOR_ZERO: return MTLBlendFactorZero;
    case VK_BLEND_FACTOR_ONE: return MTLBlendFactorOne;
    case VK_BLEND_FACTOR_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case VK_BLEND_FACTOR_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
    case VK_BLEND_FACTOR_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
    default: BAD("blend factor %d is not known", (int)f); return MTLBlendFactorOne;
    }
}
static MTLBlendOperation bo(VkBlendOp o) {
    switch (o) {
    case VK_BLEND_OP_ADD: return MTLBlendOperationAdd;
    case VK_BLEND_OP_SUBTRACT: return MTLBlendOperationSubtract;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
    default: BAD("blend operation %d is not known", (int)o); return MTLBlendOperationAdd;
    }
}
static MTLStencilDescriptor *sten(const VkStencilOpState *s) {
    static const MTLStencilOperation op[8] = { MTLStencilOperationKeep, MTLStencilOperationZero, MTLStencilOperationReplace, MTLStencilOperationIncrementClamp,
        MTLStencilOperationDecrementClamp, MTLStencilOperationInvert, MTLStencilOperationIncrementWrap, MTLStencilOperationDecrementWrap };
    MTLStencilDescriptor *d = [MTLStencilDescriptor new];
    d.stencilCompareFunction = (MTLCompareFunction)s->compareOp;       /* the same order in both */
    d.stencilFailureOperation = op[s->failOp]; d.depthFailureOperation = op[s->depthFailOp]; d.depthStencilPassOperation = op[s->passOp];
    d.readMask = s->compareMask; d.writeMask = s->writeMask;
    return d;
}
static VkResult m_vkCreateGraphicsPipelines(VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo *c, const VkAllocationCallbacks *a, VkPipeline *out) {
    @autoreleasepool {
        if (n != 1) { BAD("several pipelines at once"); return VK_ERROR_INITIALIZATION_FAILED; }
        MPipe *p = [MPipe new]; MPass *rp = O(MPass, c->renderPass);
        MTLRenderPipelineDescriptor *r = [MTLRenderPipelineDescriptor new];
        for (uint32_t i = 0; i < c->stageCount; i++) {
            id<MTLFunction> f = func(O(MShader, c->pStages[i].module)->name);
            if (c->pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT) r.vertexFunction = f; else r.fragmentFunction = f;
        }
        const VkPipelineVertexInputStateCreateInfo *vi = c->pVertexInputState;
        if (vi && vi->vertexBindingDescriptionCount) {
            MTLVertexDescriptor *v = [MTLVertexDescriptor new];
            v.layouts[0].stride = vi->pVertexBindingDescriptions[0].stride;
            for (uint32_t i = 0; i < vi->vertexAttributeDescriptionCount; i++) {
                const VkVertexInputAttributeDescription *x = &vi->pVertexAttributeDescriptions[i];
                MTLVertexFormat f = x->format == VK_FORMAT_R32_SFLOAT ? MTLVertexFormatFloat : x->format == VK_FORMAT_R32G32_SFLOAT ? MTLVertexFormatFloat2 : MTLVertexFormatFloat4;
                v.attributes[x->location].format = f; v.attributes[x->location].offset = x->offset; v.attributes[x->location].bufferIndex = 0;
            }
            r.vertexDescriptor = v; p->vbuf = 1;
        }
        if (rp->color >= 0) {
            r.colorAttachments[0].pixelFormat = pf(rp->att[rp->color].format);
            const VkPipelineColorBlendAttachmentState *b = c->pColorBlendState->pAttachments;
            MTLColorWriteMask m = MTLColorWriteMaskNone;
            if (b->colorWriteMask & 1) m |= MTLColorWriteMaskRed;
            if (b->colorWriteMask & 2) m |= MTLColorWriteMaskGreen;
            if (b->colorWriteMask & 4) m |= MTLColorWriteMaskBlue;
            if (b->colorWriteMask & 8) m |= MTLColorWriteMaskAlpha;
            r.colorAttachments[0].writeMask = m;
            if (b->blendEnable) {
                r.colorAttachments[0].blendingEnabled = YES;
                r.colorAttachments[0].sourceRGBBlendFactor = bf(b->srcColorBlendFactor); r.colorAttachments[0].destinationRGBBlendFactor = bf(b->dstColorBlendFactor);
                r.colorAttachments[0].rgbBlendOperation = bo(b->colorBlendOp);
                r.colorAttachments[0].sourceAlphaBlendFactor = bf(b->srcAlphaBlendFactor); r.colorAttachments[0].destinationAlphaBlendFactor = bf(b->dstAlphaBlendFactor);
                r.colorAttachments[0].alphaBlendOperation = bo(b->alphaBlendOp);
            }
            memcpy(p->blend, c->pColorBlendState->blendConstants, sizeof p->blend);
        }
        if (rp->ds >= 0) {
            MTLPixelFormat f = pf(rp->att[rp->ds].format);
            if (has_depth(f)) r.depthAttachmentPixelFormat = f;
            if (has_stencil(f)) r.stencilAttachmentPixelFormat = f;
        }
        r.rasterSampleCount = c->pMultisampleState ? c->pMultisampleState->rasterizationSamples : 1;
        MTLDepthStencilDescriptor *ds = [MTLDepthStencilDescriptor new];
        const VkPipelineDepthStencilStateCreateInfo *z = c->pDepthStencilState;
        if (z && z->depthTestEnable) { ds.depthCompareFunction = (MTLCompareFunction)z->depthCompareOp; ds.depthWriteEnabled = z->depthWriteEnable; }
        if (z && z->stencilTestEnable) { ds.frontFaceStencil = sten(&z->front); ds.backFaceStencil = sten(&z->back); }
        p->ds = [g_dev newDepthStencilStateWithDescriptor:ds];
        p->prim = c->pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
        if (c->pRasterizationState->depthBiasEnable) p->bias = c->pRasterizationState->depthBiasConstantFactor;
        NSError *e = nil; p->rp = [g_dev newRenderPipelineStateWithDescriptor:r error:&e];
        if (!p->rp) { fprintf(stdout, "psxrecomp: metal: pipeline: %s\n", e.localizedDescription.UTF8String); fflush(stdout); return VK_ERROR_INITIALIZATION_FAILED; }
        GIVE(VkPipeline, out, p); return VK_SUCCESS;
    }
}
static VkResult m_vkCreateComputePipelines(VkDevice d, VkPipelineCache pc, uint32_t n, const VkComputePipelineCreateInfo *c, const VkAllocationCallbacks *a, VkPipeline *out) {
    MPipe *p = [MPipe new]; NSError *e = nil;
    p->cp = [g_dev newComputePipelineStateWithFunction:func(O(MShader, c->stage.module)->name) error:&e];
    if (!p->cp) return VK_ERROR_INITIALIZATION_FAILED;
    GIVE(VkPipeline, out, p); return VK_SUCCESS;
}
static void m_vkDestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) { DROP(p); }

/* ---- command buffers ------------------------------------------------------ */
@interface MCPool : NSObject { @public NSMutableArray *cmds; } @end
@implementation MCPool @end
static void enc_end(MCmd *c) { if (c->re) { [c->re endEncoding]; c->re = nil; } if (c->be) { [c->be endEncoding]; c->be = nil; } }
static id<MTLBlitCommandEncoder> blit(MCmd *c) { if (!c->be) { enc_end(c); c->be = [c->cb blitCommandEncoder]; } return c->be; }
static VkResult m_vkCreateCommandPool(VkDevice d, const VkCommandPoolCreateInfo *c, const VkAllocationCallbacks *a, VkCommandPool *out) { MCPool *p = [MCPool new]; p->cmds = [NSMutableArray new]; GIVE(VkCommandPool, out, p); return VK_SUCCESS; }
static void m_vkDestroyCommandPool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a) { DROP(p); }
/* The renderer drops its one-shot buffers here and never uses them again; its
 * per-frame ones live in another pool, which it resets one by one. */
static VkResult m_vkResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) { if (f & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) [O(MCPool, p)->cmds removeAllObjects]; return VK_SUCCESS; }
static VkResult m_vkAllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo *i, VkCommandBuffer *out) {
    for (uint32_t k = 0; k < i->commandBufferCount; k++) { MCmd *c = [MCmd new]; [O(MCPool, i->commandPool)->cmds addObject:c]; out[k] = (VkCommandBuffer)(__bridge void *)c; }
    return VK_SUCCESS;
}
static void m_vkFreeCommandBuffers(VkDevice d, VkCommandPool p, uint32_t n, const VkCommandBuffer *c) { for (uint32_t k = 0; k < n; k++) if (c[k]) [O(MCPool, p)->cmds removeObjectIdenticalTo:O(MCmd, c[k])]; }
static VkResult m_vkBeginCommandBuffer(VkCommandBuffer h, const VkCommandBufferBeginInfo *i) { @autoreleasepool { MCmd *c = O(MCmd, h); c->re = nil; c->be = nil; c->pipe = c->cpipe = nil; c->cb = [g_queue commandBuffer]; return c->cb ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY; } }
static VkResult m_vkEndCommandBuffer(VkCommandBuffer h) { enc_end(O(MCmd, h)); return VK_SUCCESS; }
static VkResult m_vkResetCommandBuffer(VkCommandBuffer h, VkCommandBufferResetFlags f) { MCmd *c = O(MCmd, h); enc_end(c); c->cb = nil; return VK_SUCCESS; }
static void done(id<MTLCommandBuffer> b) { if (b.status == MTLCommandBufferStatusError) BAD("a command buffer failed: %s", b.error.localizedDescription.UTF8String); }
static VkResult m_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *s, VkFence f) {
    @autoreleasepool {
        for (uint32_t k = 0; k < n; k++) for (uint32_t i = 0; i < s[k].commandBufferCount; i++) {
            MCmd *c = O(MCmd, s[k].pCommandBuffers[i]); enc_end(c);
            if (c->cb) { [c->cb addCompletedHandler:^(id<MTLCommandBuffer> b) { done(b); }]; [c->cb commit]; g_last = c->cb; c->cb = nil; }
        }
        if (f) { O(MFence, f)->cb = g_last; O(MFence, f)->signaled = 1; }
        return VK_SUCCESS;
    }
}
static VkResult m_vkQueueWaitIdle(VkQueue q) { @autoreleasepool { [g_last waitUntilCompleted]; return VK_SUCCESS; } }
static VkResult m_vkDeviceWaitIdle(VkDevice d) { return m_vkQueueWaitIdle(NULL); }
static VkResult m_vkCreateFence(VkDevice d, const VkFenceCreateInfo *c, const VkAllocationCallbacks *a, VkFence *out) { GIVE(VkFence, out, [MFence new]); return VK_SUCCESS; }
static void m_vkDestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks *a) { DROP(f); }
static VkResult m_vkWaitForFences(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t t) { for (uint32_t k = 0; k < n; k++) [O(MFence, f[k])->cb waitUntilCompleted]; return VK_SUCCESS; }
static VkResult m_vkResetFences(VkDevice d, uint32_t n, const VkFence *f) { for (uint32_t k = 0; k < n; k++) O(MFence, f[k])->cb = nil; return VK_SUCCESS; }
static void m_vkCmdPipelineBarrier(VkCommandBuffer h, VkPipelineStageFlags s, VkPipelineStageFlags t, VkDependencyFlags f, uint32_t a, const VkMemoryBarrier *am, uint32_t b, const VkBufferMemoryBarrier *bm, uint32_t c, const VkImageMemoryBarrier *cm) {}

/* ---- passes ---------------------------------------------------------------- */
static MTLLoadAction la(VkAttachmentLoadOp o) { return o == VK_ATTACHMENT_LOAD_OP_LOAD ? MTLLoadActionLoad : o == VK_ATTACHMENT_LOAD_OP_CLEAR ? MTLLoadActionClear : MTLLoadActionDontCare; }
static MTLStoreAction sa(VkAttachmentStoreOp o) { return o == VK_ATTACHMENT_STORE_OP_STORE ? MTLStoreActionStore : MTLStoreActionDontCare; }
static void m_vkCmdBeginRenderPass(VkCommandBuffer h, const VkRenderPassBeginInfo *b, VkSubpassContents sc) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); MPass *p = O(MPass, b->renderPass); MFB *f = O(MFB, b->framebuffer); enc_end(c);
        MTLRenderPassDescriptor *r = [MTLRenderPassDescriptor renderPassDescriptor];
        c->cf = c->df = c->sf = MTLPixelFormatInvalid; c->samples = 1;
        if (p->color >= 0) {
            MTLRenderPassColorAttachmentDescriptor *x = r.colorAttachments[0]; const VkAttachmentDescription *t = &p->att[p->color];
            x.texture = vtex(f->v[p->color]); x.loadAction = la(t->loadOp); x.storeAction = sa(t->storeOp);
            if (b->clearValueCount > (uint32_t)p->color) { const float *k = b->pClearValues[p->color].color.float32; x.clearColor = MTLClearColorMake(k[0], k[1], k[2], k[3]); }
            if (p->resolve >= 0) { x.resolveTexture = vtex(f->v[p->resolve]); x.storeAction = t->storeOp == VK_ATTACHMENT_STORE_OP_STORE ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionMultisampleResolve; }
            c->cf = x.texture.pixelFormat; c->samples = x.texture.sampleCount;
            c->tw = (uint32_t)x.texture.width; c->th = (uint32_t)x.texture.height;
        }
        if (p->ds >= 0) {
            id<MTLTexture> t = vtex(f->v[p->ds]); const VkAttachmentDescription *k = &p->att[p->ds];
            if (has_depth(t.pixelFormat)) { r.depthAttachment.texture = t; r.depthAttachment.loadAction = la(k->loadOp); r.depthAttachment.storeAction = sa(k->storeOp); c->df = t.pixelFormat; }
            if (has_stencil(t.pixelFormat)) { r.stencilAttachment.texture = t; r.stencilAttachment.loadAction = la(k->stencilLoadOp); r.stencilAttachment.storeAction = sa(k->stencilStoreOp); c->sf = t.pixelFormat; }
        }
        c->re = [c->cb renderCommandEncoderWithDescriptor:r];
    }
}
static void m_vkCmdEndRenderPass(VkCommandBuffer h) { enc_end(O(MCmd, h)); }
static void m_vkCmdBindPipeline(VkCommandBuffer h, VkPipelineBindPoint b, VkPipeline p) { MCmd *c = O(MCmd, h); if (b == VK_PIPELINE_BIND_POINT_COMPUTE) c->cpipe = O(MPipe, p); else c->pipe = O(MPipe, p); }
static void m_vkCmdBindDescriptorSets(VkCommandBuffer h, VkPipelineBindPoint b, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *dy) {
    MCmd *c = O(MCmd, h);
    for (uint32_t k = 0; k < n && first + k < 2; k++) { if (b == VK_PIPELINE_BIND_POINT_COMPUTE) c->csets[first + k] = O(MSet, s[k]); else c->sets[first + k] = O(MSet, s[k]); }
}
static void m_vkCmdBindVertexBuffers(VkCommandBuffer h, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { MCmd *c = O(MCmd, h); c->vb = O(MBuf, b[0]); c->vboff = o[0]; }
static void m_vkCmdPushConstants(VkCommandBuffer h, VkPipelineLayout l, VkShaderStageFlags s, uint32_t off, uint32_t n, const void *p) { MCmd *c = O(MCmd, h); if (off + n <= sizeof c->push) { memcpy(c->push + off, p, n); c->npush = off + n; } }
static void m_vkCmdSetViewport(VkCommandBuffer h, uint32_t f, uint32_t n, const VkViewport *v) { O(MCmd, h)->vp = (MTLViewport){v->x, v->y, v->width, v->height, v->minDepth, v->maxDepth}; }
static void m_vkCmdSetScissor(VkCommandBuffer h, uint32_t f, uint32_t n, const VkRect2D *r) { O(MCmd, h)->sc = *r; }
static void m_vkCmdSetStencilReference(VkCommandBuffer h, VkStencilFaceFlags f, uint32_t r) { O(MCmd, h)->sref = r; }
static void m_vkCmdDraw(VkCommandBuffer h, uint32_t n, uint32_t inst, uint32_t first, uint32_t fi) {
    MCmd *c = O(MCmd, h);
    if (!c->re || !c->pipe || !n) return;
    int64_t x0 = c->sc.offset.x < 0 ? 0 : c->sc.offset.x, y0 = c->sc.offset.y < 0 ? 0 : c->sc.offset.y;
    int64_t x1 = (int64_t)c->sc.offset.x + c->sc.extent.width, y1 = (int64_t)c->sc.offset.y + c->sc.extent.height;
    if (x1 > c->tw) x1 = c->tw;
    if (y1 > c->th) y1 = c->th;
    if (x1 <= x0 || y1 <= y0) return;
    id<MTLRenderCommandEncoder> e = c->re; MPipe *p = c->pipe;
    [e setRenderPipelineState:p->rp]; [e setDepthStencilState:p->ds];
    [e setViewport:c->vp]; [e setScissorRect:(MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)}];
    [e setStencilReferenceValue:c->sref]; [e setBlendColorRed:p->blend[0] green:p->blend[1] blue:p->blend[2] alpha:p->blend[3]];
    [e setDepthBias:p->bias slopeScale:0 clamp:0]; [e setCullMode:MTLCullModeNone];
    if (p->vbuf && c->vb) [e setVertexBuffer:c->vb->buf offset:(NSUInteger)(c->vb->off + c->vboff) atIndex:0];
    [e setVertexBytes:c->push length:64 atIndex:1]; [e setFragmentBytes:c->push length:64 atIndex:1];
    for (int s = 0; s < 2; s++) if (c->sets[s]) { [e setFragmentTexture:c->sets[s]->tex[0] atIndex:(NSUInteger)s]; [e setFragmentSamplerState:c->sets[s]->samp[0] ? c->sets[s]->samp[0] : g_near atIndex:(NSUInteger)s]; }
    [e drawPrimitives:p->prim vertexStart:first vertexCount:n];
}
static void m_vkCmdDispatch(VkCommandBuffer h, uint32_t x, uint32_t y, uint32_t z) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); if (!c->cpipe) return; enc_end(c);
        id<MTLComputeCommandEncoder> e = [c->cb computeCommandEncoder];
        [e setComputePipelineState:c->cpipe->cp]; [e setBytes:c->push length:64 atIndex:1];
        if (c->csets[0]) for (int b = 0; b < 2; b++) [e setTexture:c->csets[0]->tex[b] atIndex:(NSUInteger)b];
        [e dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [e endEncoding];
    }
}
/* Mid-pass clears and scaled copies are drawn. */
static id<MTLRenderPipelineState> util_pipe(NSString *v, NSString *f, MTLPixelFormat cf, MTLPixelFormat df, MTLPixelFormat sf, NSUInteger samples, int write) {
    static NSMutableDictionary *all;
    if (!all) all = [NSMutableDictionary new];
    NSString *k = [NSString stringWithFormat:@"%@ %@ %lu %lu %lu %lu %d", v, f, (unsigned long)cf, (unsigned long)df, (unsigned long)sf, (unsigned long)samples, write];
    id<MTLRenderPipelineState> p = all[k];
    if (p) return p;
    MTLRenderPipelineDescriptor *r = [MTLRenderPipelineDescriptor new];
    r.vertexFunction = func(v); r.fragmentFunction = func(f); r.rasterSampleCount = samples;
    r.colorAttachments[0].pixelFormat = cf; r.colorAttachments[0].writeMask = write ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    r.depthAttachmentPixelFormat = df; r.stencilAttachmentPixelFormat = sf;
    NSError *e = nil; p = [g_dev newRenderPipelineStateWithDescriptor:r error:&e];
    if (!p) { BAD("helper pipeline: %s", e.localizedDescription.UTF8String); return nil; }
    all[k] = p; return p;
}
static void m_vkCmdClearAttachments(VkCommandBuffer h, uint32_t n, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); if (!c->re) return;
        id<MTLRenderCommandEncoder> e = c->re; int color = 0, depth = 0, stencil = 0; float col[4] = {0}, z = 0; uint32_t s = 0;
        for (uint32_t k = 0; k < n; k++) {
            if (a[k].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) { color = 1; memcpy(col, a[k].clearValue.color.float32, 16); }
            if (a[k].aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) { depth = 1; z = a[k].clearValue.depthStencil.depth; }
            if (a[k].aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) { stencil = 1; s = a[k].clearValue.depthStencil.stencil; }
        }
        id<MTLRenderPipelineState> p = util_pipe(@"clear_vert", @"clear_frag", c->cf, c->df, c->sf, c->samples, color);
        if (!p) return;
        MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
        d.depthCompareFunction = MTLCompareFunctionAlways; d.depthWriteEnabled = depth && c->df != MTLPixelFormatInvalid;
        if (stencil) { MTLStencilDescriptor *t = [MTLStencilDescriptor new]; t.depthStencilPassOperation = MTLStencilOperationReplace; d.frontFaceStencil = d.backFaceStencil = t; }
        [e setRenderPipelineState:p]; [e setDepthStencilState:[g_dev newDepthStencilStateWithDescriptor:d]]; [e setStencilReferenceValue:s];
        [e setViewport:(MTLViewport){0, 0, c->tw, c->th, 0, 1}]; [e setScissorRect:(MTLScissorRect){0, 0, c->tw, c->th}];
        [e setDepthBias:0 slopeScale:0 clamp:0]; [e setCullMode:MTLCullModeNone];
        for (uint32_t k = 0; k < nr; k++) {
            const VkRect2D *q = &r[k].rect; float W = c->tw, H = c->th;
            float b[8] = { 2 * q->offset.x / W - 1, 1 - 2 * q->offset.y / H, 2 * (q->offset.x + (float)q->extent.width) / W - 1, 1 - 2 * (q->offset.y + (float)q->extent.height) / H, z, 0, 0, 0 };
            [e setVertexBytes:b length:sizeof b atIndex:0]; [e setFragmentBytes:col length:16 atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
    }
}
static void m_vkCmdClearColorImage(VkCommandBuffer h, VkImage i, VkImageLayout l, const VkClearColorValue *v, uint32_t n, const VkImageSubresourceRange *r) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); MImg *m = O(MImg, i); enc_end(c);
        for (uint32_t lv = 0; lv < m->levels; lv++) {
            MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
            p.colorAttachments[0].texture = m->tex; p.colorAttachments[0].level = lv; p.colorAttachments[0].loadAction = MTLLoadActionClear; p.colorAttachments[0].storeAction = MTLStoreActionStore;
            p.colorAttachments[0].clearColor = m->fmt == VK_FORMAT_R16_UINT ? MTLClearColorMake(v->uint32[0], 0, 0, 0) : MTLClearColorMake(v->float32[0], v->float32[1], v->float32[2], v->float32[3]);
            [[c->cb renderCommandEncoderWithDescriptor:p] endEncoding];
        }
    }
}
static void m_vkCmdClearDepthStencilImage(VkCommandBuffer h, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *v, uint32_t n, const VkImageSubresourceRange *r) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); MImg *m = O(MImg, i); enc_end(c);
        MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
        if (has_depth(m->tex.pixelFormat)) { p.depthAttachment.texture = m->tex; p.depthAttachment.loadAction = MTLLoadActionClear; p.depthAttachment.storeAction = MTLStoreActionStore; p.depthAttachment.clearDepth = v->depth; }
        p.stencilAttachment.texture = m->tex; p.stencilAttachment.loadAction = MTLLoadActionClear; p.stencilAttachment.storeAction = MTLStoreActionStore; p.stencilAttachment.clearStencil = v->stencil;
        [[c->cb renderCommandEncoderWithDescriptor:p] endEncoding];
    }
}
/* ---- copies ---------------------------------------------------------------- */
static void m_vkCmdCopyBufferToImage(VkCommandBuffer h, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n, const VkBufferImageCopy *r) {
    MCmd *c = O(MCmd, h); MBuf *s = O(MBuf, b); MImg *m = O(MImg, i);
    for (uint32_t k = 0; k < n; k++) {
        uint32_t row = (r[k].bufferRowLength ? r[k].bufferRowLength : r[k].imageExtent.width) * bpp(m->fmt);
        [blit(c) copyFromBuffer:s->buf sourceOffset:(NSUInteger)(s->off + r[k].bufferOffset) sourceBytesPerRow:row sourceBytesPerImage:row * (r[k].bufferImageHeight ? r[k].bufferImageHeight : r[k].imageExtent.height)
            sourceSize:MTLSizeMake(r[k].imageExtent.width, r[k].imageExtent.height, 1) toTexture:m->tex destinationSlice:0 destinationLevel:r[k].imageSubresource.mipLevel destinationOrigin:MTLOriginMake((NSUInteger)r[k].imageOffset.x, (NSUInteger)r[k].imageOffset.y, 0)];
    }
}
static void m_vkCmdCopyImageToBuffer(VkCommandBuffer h, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy *r) {
    MCmd *c = O(MCmd, h); MBuf *s = O(MBuf, b); MImg *m = O(MImg, i);
    for (uint32_t k = 0; k < n; k++) {
        uint32_t row = (r[k].bufferRowLength ? r[k].bufferRowLength : r[k].imageExtent.width) * bpp(m->fmt);
        [blit(c) copyFromTexture:m->tex sourceSlice:0 sourceLevel:r[k].imageSubresource.mipLevel sourceOrigin:MTLOriginMake((NSUInteger)r[k].imageOffset.x, (NSUInteger)r[k].imageOffset.y, 0)
            sourceSize:MTLSizeMake(r[k].imageExtent.width, r[k].imageExtent.height, 1) toBuffer:s->buf destinationOffset:(NSUInteger)(s->off + r[k].bufferOffset) destinationBytesPerRow:row destinationBytesPerImage:row * r[k].imageExtent.height];
    }
}
static void m_vkCmdCopyImage(VkCommandBuffer h, VkImage a, VkImageLayout la_, VkImage b, VkImageLayout lb, uint32_t n, const VkImageCopy *r) {
    MCmd *c = O(MCmd, h); MImg *x = O(MImg, a), *y = O(MImg, b);
    for (uint32_t k = 0; k < n; k++)
        [blit(c) copyFromTexture:x->tex sourceSlice:0 sourceLevel:r[k].srcSubresource.mipLevel sourceOrigin:MTLOriginMake((NSUInteger)r[k].srcOffset.x, (NSUInteger)r[k].srcOffset.y, 0) sourceSize:MTLSizeMake(r[k].extent.width, r[k].extent.height, 1)
            toTexture:y->tex destinationSlice:0 destinationLevel:r[k].dstSubresource.mipLevel destinationOrigin:MTLOriginMake((NSUInteger)r[k].dstOffset.x, (NSUInteger)r[k].dstOffset.y, 0)];
}
static void m_vkCmdBlitImage(VkCommandBuffer h, VkImage a, VkImageLayout la_, VkImage b, VkImageLayout lb, uint32_t n, const VkImageBlit *r, VkFilter f) {
    @autoreleasepool {
        MCmd *c = O(MCmd, h); MImg *x = O(MImg, a), *y = O(MImg, b); enc_end(c);
        for (uint32_t k = 0; k < n; k++) {
            uint32_t sl = r[k].srcSubresource.mipLevel, dl = r[k].dstSubresource.mipLevel;
            id<MTLTexture> src = x->levels == 1 ? x->tex : [x->tex newTextureViewWithPixelFormat:x->tex.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(sl, 1) slices:NSMakeRange(0, 1)];
            MTLRenderPassDescriptor *p = [MTLRenderPassDescriptor renderPassDescriptor];
            p.colorAttachments[0].texture = y->tex; p.colorAttachments[0].level = dl; p.colorAttachments[0].loadAction = MTLLoadActionLoad; p.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderPipelineState> u = util_pipe(@"present_vert", @"scale_frag", y->tex.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid, 1, 1);
            if (!u) return;
            float W = src.width, H = src.height;
            float rect[4] = { r[k].srcOffsets[0].x / W, r[k].srcOffsets[0].y / H, r[k].srcOffsets[1].x / W, r[k].srcOffsets[1].y / H };
            double dx = r[k].dstOffsets[0].x, dy = r[k].dstOffsets[0].y, dw = r[k].dstOffsets[1].x - dx, dh = r[k].dstOffsets[1].y - dy;
            id<MTLRenderCommandEncoder> e = [c->cb renderCommandEncoderWithDescriptor:p];
            [e setRenderPipelineState:u]; [e setViewport:(MTLViewport){dx, dy, dw, dh, 0, 1}];
            [e setFragmentBytes:rect length:16 atIndex:1]; [e setFragmentTexture:src atIndex:0]; [e setFragmentSamplerState:f == VK_FILTER_LINEAR ? g_lin : g_near atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [e endEncoding];
        }
    }
}

static PFN_vkVoidFunction m_vkGetDeviceProcAddr(VkDevice d, const char *n);
PFN_vkVoidFunction psx_metal_vkGetInstanceProcAddr(VkInstance i, const char *n) {
#define F(x) if (!strcmp(n, #x)) return (PFN_vkVoidFunction)m_##x;
#include "gpu_metal_layer_list.h"
    F(vkCreateHeadlessSurfaceEXT) F(vkGetDeviceProcAddr)
#undef F
    fprintf(stdout, "psxrecomp: metal: %s is not there\n", n); fflush(stdout);
    return NULL;
}
static PFN_vkVoidFunction m_vkGetDeviceProcAddr(VkDevice d, const char *n) { return psx_metal_vkGetInstanceProcAddr(NULL, n); }
