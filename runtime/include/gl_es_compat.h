/* gl_es_compat.h — desktop-GL names the GL renderer uses that OpenGL ES 3.0
 * headers do not declare. First goal: the renderer compiles on iOS while the
 * software renderer is the one that runs. Entries are added as the compiler
 * asks for them; each is the ES equivalent or a documented no-op. */
#ifndef GL_ES_COMPAT_H
#define GL_ES_COMPAT_H
#ifndef APIENTRY
#define APIENTRY
#endif
#ifndef GLAPIENTRY
#define GLAPIENTRY
#endif
#define glClearDepth(d) glClearDepthf((GLfloat)(d))
#define glDepthRange(n, f) glDepthRangef((GLfloat)(n), (GLfloat)(f))
/* No stereo back buffers on ES. */
#ifndef GL_BACK_LEFT
#define GL_BACK_LEFT GL_BACK
#endif
#ifndef GL_BACK_RIGHT
#define GL_BACK_RIGHT GL_BACK
#endif
/* ES 3.0 has neither call. glGetTexImage only feeds developer dump tools;
 * level-parameter queries read 0, which callers treat as "no texture". */
#ifndef GL_TEXTURE_WIDTH
#define GL_TEXTURE_WIDTH 0x1000
#define GL_TEXTURE_HEIGHT 0x1001
#endif
#define glGetTexImage(target, level, format, type, pixels) ((void)(pixels))
static inline void psx_es_get_tex_level_parameteriv(GLenum t, GLint l, GLenum p, GLint *o) {
    (void)t; (void)l; (void)p; if (o) *o = 0;
}
#define glGetTexLevelParameteriv(t, l, p, o) psx_es_get_tex_level_parameteriv((t), (l), (p), (o))
#endif
