import tensorflow as tf
from typing import Dict, List, Tuple, Any, Callable #可调用对象

class Activation(object):
  def __init__(self,
               name: str,
               op: Callable[[tf.Tensor], tf.Tensor], #前者为输入list，list的内容为Tensor，后者为Callable返回的对象，也为Tensor
               thresholds: List[Callable[[tf.Tensor], tf.Tensor]],
               is_pointwise: bool = True) -> None:
    """Store all arguments as object fields.
    Args:
      name: name string uniquely identifying a non-linearity, potentially parametrized, e.g. 'relu', 'gamma_relu_0.5'.
      op: tf op, e.g. tf.nn.relu, tf.nn.tanh or a more complex tf expression.
      thresholds: list of thresholds on the output values on the non-linearity, possibly depending on inputs, thus being
          functions of one argument, e.g. [lambda x: 0].
        Non-linearities considered for counting the number of linear regions are assumed non-decreasing and
          piecewise-linear. Under this assumption, we can determine the linear region of a neuron into which the input
          falls only by looking at the post-activation (non-linearity output) of the unit and comparing it to a list of
          thresholds, which will be the values of the non-linearity at it's discontinuous points.
        (Not all non-linearities in the project satisfy the above assumptions, since other expressivity metrics do not
         rely on linear regions.)
        E.g. ReLU(x) = max(0, x) has one threshold: 0, and the linear region of an input is determined by whether it
          falls below or above 0. ReLU6(x) = min(6, ReLU(x)) has 2 thresholds: [0, 6] and three respective linear
          regions: (-inf; 0), [0, 6),[6; +inf).
        In more complex cases a non-linearity can be not a pointwise function, which is why thresholds are functions of
          the whole tensor inputs, not just numbers. E.g. ReLU_median(x) = min(Median({x_i | x_i > 0}), ReLU(x)).
        Currently thresholds are simply determined manually from the symbolic expression of the non-linearity and not
          from the underlying tf op, which is why they have to be passed as an input parameter.
      is_pointwise: True to indicate that the non-linearity is pointwise. If true, only the diagonal elements of the
        jacobian are returned when derivative() is called.
    """
    self.op = op
    self.thresholds = thresholds
    self.name = name
    self.is_pointwise = is_pointwise

def _get_layer_transition(act: Activation, pre_activation: tf.Tensor, post_activation: tf.Tensor, eps: float, is_last: bool) -> tf.Tensor:
  with tf.variable_scope('patterns'):
    num_thresholds = len(act.thresholds)
    if num_thresholds == 1 and not is_last:
      t = act.thresholds[0]
      patterns = tf.greater(post_activation, t(pre_activation))
    else:
      # add current-layer patterns to the list.
      pattern_type = tf.bool if num_thresholds <= 1 else tf.int8
      patterns = tf.zeros(tf.shape(post_activation), pattern_type)
      if not is_last and num_thresholds != 0:
        if 'abs' in act.name or 'gamma' in act.name:
          # hack to handle exceptional non-monotonic activations
          patterns += _int8_greater(pre_activation, 0)
        else:
          for t in act.thresholds[:-1]:
            patterns += _int8_greater(post_activation, t(pre_activation))
          last_t = act.thresholds[-1](pre_activation)
          if len(act.thresholds) > 1:
            last_t -= eps
          patterns += _int8_greater(post_activation, last_t)
    return patterns

def combine_layer_props(props: Dict[str, List[tf.Tensor]]) -> None:
  for name, prop in props.items():
    prop = _combine_layer_prop(prop, name)
    props[name] = prop

def _combine_layer_prop(prop: Any[List[tf.Tensor], List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]], name: str) -> tf.Tensor:
  # cast last prop in case of bool / int8 transition types.
  type_first = prop[0].dtype
  type_last = prop[-1].dtype
  if type_first != type_last:
    prop[-1] = tf.cast(prop[-1], type_first)
  prop = tuple(prop)
  # ...
  prop = _concat(prop)
  if 'transition' in name:
    prop = _count_batch_transitions_tf(prop)
  # ...
  return prop

def _concat(prop: Tuple[tf.Tensor, ...]) -> tf.Tensor:
  if len(prop) == 1:
    prop = prop[0]
  elif len(prop) >= 2:
    prop = tf.concat(prop, axis=-1)
  return prop

def _count_batch_transitions_tf(array: tf.Tensor) -> tf.Tensor:
  with tf.variable_scope('transition'):
    array_prev = array[:-1]
    array_next = array[1:]
    axis = list(range(array.shape.ndims)[1:])
    mismatches = tf.not_equal(array_prev, array_next)
    transitions = tf.reduce_any(mismatches, axis=axis, keep_dims=True) #对除了第一维度之外的维度做逻辑或
    return transitions

def _count_batch_transition_torch(array: torch.Tensor) -> torch.Tensor:
    array_prev = array[:-1]
    array_next = array[1:]
    axis = list(range(array.shape.ndims)[1:])
    mismatches = torch.not_equal(array_prev, array_next)
    transitions = torch.any(mismatches, dim=axis, keepdim=True) #对除了第一维度之外的维度做逻辑或
    return transitions

def _int8_greater(tsr1: tf.Tensor, tsr2: tf.Tensor, name: str = None) -> tf.Tensor:
  return tf.cast(tf.greater(tsr1, tsr2), tf.int8, name)
