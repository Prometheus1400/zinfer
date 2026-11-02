const c = @import("mlx_c");

pub const DType = enum(c.mlx_dtype) {
    boolean = c.MLX_BOOL,
    uint8 = c.MLX_UINT8,
    uint16 = c.MLX_UINT16,
    uint32 = c.MLX_UINT32,
    uint64 = c.MLX_UINT64,
    int8 = c.MLX_INT8,
    int16 = c.MLX_INT16,
    int32 = c.MLX_INT32,
    int64 = c.MLX_INT64,
    float16 = c.MLX_FLOAT16,
    float32 = c.MLX_FLOAT32,
    float64 = c.MLX_FLOAT64,
    bfloat16 = c.MLX_BFLOAT16,
    complex64 = c.MLX_COMPLEX64,
};
