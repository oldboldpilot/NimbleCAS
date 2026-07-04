// NimbleCAS plot shader.
//
// One pipeline draws ALL plot geometry as a triangle list. Each vertex carries
// a clip-space position (vec2) and an RGBA color (vec4), so gridlines, axes,
// line series (triangulated to quads for width) and scatter points (small
// quads) all render in a single draw call.
//
// Vertex buffer layout (matches renderer.js):
//   arrayStride = 24 bytes (6 x f32)
//   @location(0) position : vec2<f32>  offset 0
//   @location(1) color    : vec4<f32>  offset 8

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       col : vec4<f32>,
};

@vertex
fn vs_main(@location(0) position : vec2<f32>,
           @location(1) color    : vec4<f32>) -> VSOut {
  var out : VSOut;
  out.pos = vec4<f32>(position, 0.0, 1.0);
  out.col = color;
  return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  return in.col;
}
