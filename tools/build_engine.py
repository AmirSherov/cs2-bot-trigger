"""Build a batch-one RTX 4060 engine; requires TensorRT 10.13.3 and CUDA 12.

The version-compatible engine uses the separately shipped lean runtime. Keep the
DirectML model as a portable fallback for other adapters or incompatible drivers.
"""
from pathlib import Path
import tensorrt as trt

root = Path(__file__).resolve().parent.parent
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse((root / 'person-seg-center-320.onnx').read_bytes()):
    raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
config.builder_optimization_level = 3
config.max_aux_streams = 0
cache_path = root / 'build' / 'tensorrt-timing.cache'
config.set_timing_cache(config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b''), False)
engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError('TensorRT engine build failed')
destination = root / 'person-seg-rtx4060.engine'
destination.write_bytes(bytes(engine))
cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
print(destination)
