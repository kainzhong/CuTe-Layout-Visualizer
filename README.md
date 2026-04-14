# CuTe Layout Visualizer

An interactive browser-based tool for visualizing [CUTLASS CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md) layouts and thread-value (TV) layouts.

**Try now:** https://kainzhong.github.io/CuTe-Layout-Visualizer/

## Features

- **Layout visualization** -- Render any CuTe layout specified as `shape:stride` with support for nested/hierarchical modes (e.g. `((2,4),(2,4)):((1,8),(2,16))`)
- **TV layout visualization** -- Map thread-value layouts onto a tile to see how threads and values are distributed across a 2D grid
- **Multiple tabs** -- Open several independent workspaces side by side
- **Presets** -- Built-in examples for common layout patterns
- **Auto stride** -- Omit the stride to default to column-major

## Usage

Enter a CuTe layout string in the input field using the standard notation:

```
(shape_0, shape_1):(stride_0, stride_1)
```

Nested shapes and strides are supported:

```
((2,4),(2,4)):((1,8),(2,16))
```

Click **Render** to visualize.
