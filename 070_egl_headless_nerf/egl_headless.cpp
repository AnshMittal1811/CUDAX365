#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <cstdio>

int main(){
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) return 1;
    eglInitialize(dpy, nullptr, nullptr);
    EGLint configAttribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE};
    EGLConfig config;
    EGLint num;
    eglChooseConfig(dpy, configAttribs, &config, 1, &num);
    EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext ctx = eglCreateContext(dpy, config, EGL_NO_CONTEXT, ctxAttribs);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    printf("rendered\n");
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    return 0;
}
