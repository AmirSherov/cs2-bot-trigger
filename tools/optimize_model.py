"""Keep YOLO weights; compute the center-mask logit on GPU and return compact output."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

root = Path(__file__).resolve().parent.parent
source = root / 'person-seg-320.onnx'
destination = root / 'person-seg-fast-320.onnx'
model = onnx.load(source)
det, proto = [output.name for output in model.graph.output]
nodes = model.graph.node
def const(name, value):
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
    return name
def slice_channels(input_name, start, end, output):
    nodes.append(helper.make_node('Slice', [input_name,
        const(output+'_start', np.array([start], np.int64)),
        const(output+'_end', np.array([end], np.int64)),
        const(output+'_axis', np.array([1], np.int64))], [output]))

slice_channels(det, 0, 5, 'fast_boxes_person')
slice_channels(det, 5, 84, 'fast_other_classes')
nodes.append(helper.make_node('ReduceMax', ['fast_other_classes'], ['fast_other_max'], axes=[1], keepdims=1))
nodes.append(helper.make_node('Reshape', [proto, const('fast_proto_shape', np.array([1,32,6400],np.int64))], ['fast_flat_proto']))
# Match C++ half-pixel bilinear upsampling exactly at input point (160,160).
nodes.append(helper.make_node('Gather', ['fast_flat_proto',
    const('fast_center_indices', np.array([39*80+39,39*80+40,40*80+39,40*80+40],np.int64))], ['fast_center_features'], axis=2))
nodes.append(helper.make_node('Mul', ['fast_center_features',
    const('fast_center_weights', np.array([[[.140625,.234375,.234375,.390625]]],np.float32))], ['fast_weighted_features']))
axes = const('fast_sum_axis2', np.array([2],np.int64))
nodes.append(helper.make_node('ReduceSum', ['fast_weighted_features',axes], ['fast_center_proto'], keepdims=1))
slice_channels(det, 84, 116, 'fast_coefficients')
nodes.append(helper.make_node('Mul', ['fast_coefficients','fast_center_proto'], ['fast_center_products']))
nodes.append(helper.make_node('ReduceSum', ['fast_center_products',
    const('fast_sum_axis1', np.array([1],np.int64))], ['fast_center_logits'], keepdims=1))
nodes.append(helper.make_node('Concat', ['fast_boxes_person','fast_other_max','fast_center_logits'], ['center_output'], axis=1))
del model.graph.output[:]
model.graph.output.append(helper.make_tensor_value_info('center_output',TensorProto.FLOAT,[1,7,2100]))
model.metadata_props.add(key='center_only_output',value='xywh,person,max_other,center_mask_logit; center=160,160')
onnx.checker.check_model(model)
onnx.save(model,destination)
print(destination)
