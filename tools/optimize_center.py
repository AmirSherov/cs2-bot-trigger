"""Compute only the four needed prototype pixels, preserving their receptive field.

The original 320x320 network and all detection/classification heads are retained.
For the prototype head: source[18:22] -> valid 3x3 -> 2x upsample ->
valid 3x3 -> 1x1 produces original prototype pixels [39:41,39:41].
"""
from pathlib import Path
import onnx
import numpy as np
from onnx import helper, numpy_helper
from onnxconverter_common import float16

root = Path(__file__).resolve().parent.parent
model = onnx.load(root / 'person-seg-fast-320.onnx')
nodes = {node.name: node for node in model.graph.node if node.name}
cv1 = nodes['/model.22/proto/cv1/conv/Conv']
cv2 = nodes['/model.22/proto/cv2/conv/Conv']
cv3 = nodes['/model.22/proto/cv3/conv/Conv']
upsample = nodes['/model.22/proto/upsample/ConvTranspose']

def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}

for node in (cv1, cv2):
    a = attributes(node)
    assert a['kernel_shape'] == [3, 3] and a['strides'] == [1, 1]
    assert a['pads'] == [1, 1, 1, 1] and a.get('dilations', [1, 1]) == [1, 1]
assert attributes(cv3)['kernel_shape'] == [1, 1]
assert attributes(cv3)['pads'] == [0, 0, 0, 0]
assert attributes(upsample)['kernel_shape'] == [2, 2]
assert attributes(upsample)['strides'] == [2, 2]
assert attributes(upsample)['pads'] == [0, 0, 0, 0]

def constant(name, data):
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(data, dtype=np.int64), name))
    return name

slice_node = helper.make_node('Slice', [cv1.input[0], constant('center_starts', [18, 18]),
    constant('center_ends', [22, 22]), constant('center_axes', [2, 3])],
    ['center_receptive_field'], name='center_receptive_field_slice')
cv1.input[0] = 'center_receptive_field'
for node in (cv1, cv2):
    for attr in node.attribute:
        if attr.name == 'pads':
            attr.ints[:] = [0, 0, 0, 0]
index = next(i for i, node in enumerate(model.graph.node) if node.name == cv1.name)
model.graph.node.insert(index, slice_node)
for tensor in model.graph.initializer:
    if tensor.name == 'fast_proto_shape':
        tensor.CopyFrom(numpy_helper.from_array(np.array([1, 32, 4], dtype=np.int64), tensor.name))
    elif tensor.name == 'fast_center_indices':
        tensor.CopyFrom(numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.int64), tensor.name))
del model.graph.value_info[:]
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(model)
model.metadata_props.add(key='prototype_receptive_field', value='input[18:22,18:22]; original proto[39:41,39:41]')
destination = root / 'person-seg-center-320.onnx'
onnx.save(model, destination)
print(destination)
half = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.checker.check_model(half)
destination = root / 'person-seg-center-fp16-320.onnx'
onnx.save(half, destination)
print(destination)
