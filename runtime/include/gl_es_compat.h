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
#endif
