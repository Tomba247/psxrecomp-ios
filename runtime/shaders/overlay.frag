#version 450
/* Touch UI overlay: a premultiplied-alpha canvas blended over the finished
 * frame (blend ONE, ONE_MINUS_SRC_ALPHA). Uses present.vert. */
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
layout(set = 0, binding = 0) uniform sampler2D u_src;
void main() {
    o_col = texture(u_src, v_uv);
}
