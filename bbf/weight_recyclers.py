import functools
import flax
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import optax
import csv
import os


def reset_momentum(momentum, mask):
  new_momentum = momentum if mask is None else momentum * (1.0 - mask)
  return new_momentum


def weight_reinit_zero(param, mask):
  if mask is None:
    return param
  else:
    new_param = jnp.zeros_like(param)
    param = jnp.where(mask == 1, new_param, param)
    return param


def weight_scale(param, mask, key, scale):
  param = jnp.where(mask == 1, scale * param, param)
  return param


def weight_reinit_random(param,
                         mask,
                         key,
                         weight_scaling=False,
                         scale=1.0,
                         weights_type='incoming',
                         sp_weight=0.5):
  """Randomly reinit recycled weights and may scale its norm.

  If scaling applied, the norm of recycled weights equals
  the average norm of non recycled weights per neuron multiplied by a scalar.

  Args:
    param: current param
    mask: incoming/outgoing mask for recycled weights
    key: random key to generate new random weights
    weight_scaling: if true scale recycled weights with the norm of non recycled
    scale: scale to multiply the new weights norm.
    weights_type: incoming or outgoing weights

  Returns:
  params: new params after weight recycle.
  """
  if mask is None or key is None:
    return param

  new_param = nn.initializers.xavier_uniform()(key, shape=param.shape)

  if weight_scaling:
    axes = list(range(param.ndim))
    if weights_type == 'outgoing':
      del axes[-2]
    else:
      del axes[-1]

    neuron_mask = jnp.mean(mask, axis=axes)

    non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    non_recycled_norm = jnp.sum(norm_per_neuron *
                                (1 - neuron_mask)) / non_dead_count
    non_recycled_norm = non_recycled_norm * scale

    normalized_new_param = _weight_normalization_per_neuron_norm(
        new_param, axes)
    new_param = normalized_new_param * non_recycled_norm

  param = jnp.where(mask == 1, (1-sp_weight) * new_param + sp_weight * param, param)
  return param


def _weight_normalization_per_neuron_norm(param, axes):
  norm_per_neuron = _get_norm_per_neuron(param, axes)
  norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
  normalized_param = param / norm_per_neuron
  return normalized_param


def _get_norm_per_neuron(param, axes):
  return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))


@functools.partial(jax.jit, static_argnames=('dead_neurons_threshold'))
def score2mask(activation, dead_neurons_threshold):
  """
    计算休眠单元掩码

    Args: 
      activation: 激活值,
      outgoing_score: 输出权重休眠得分,
      dead_neurons_threshold: 休眠阈值

    Returns:
      mask: 休眠单元掩码
  """
  # 按batch求激活值绝对值的均值
  reduce_axes = list(range(activation.ndim - 1))
  score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
  # 分数标准化
  score /= jnp.mean(score) + 1e-9
  return score <= dead_neurons_threshold


@functools.partial(jax.jit, static_argnames=())
def score2maskmid(activation):
    """
    计算休眠单元掩码，将前25%和后25%得分的单元标记为休眠单元

    Args: 
      activation: 激活值

    Returns:
      mask: 休眠单元掩码，True表示该单元为休眠状态
    """
    # 按batch求激活值绝对值的均值
    reduce_axes = list(range(activation.ndim - 1))
    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    # 分数标准化
    score /= jnp.mean(score) + 1e-9
    
    # 计算25%和75%的分位数
    q25 = jnp.percentile(score, 25)
    q75 = jnp.percentile(score, 75)
    q50 = jnp.percentile(score, 50)
    # q5 = jnp.percentile(score, 5)
    
    # 前25%（<=q25）和后25%（>=q75）的单元标记为休眠
    # return (score <= q25) | (score >= q75)
    # return (score >= q25) & (score <= q75)
    return score >= q50



@functools.partial(jax.jit, static_argnames=('dead_neurons_threshold'))
def score2maskSparse(activation, dead_neurons_threshold, last_av, last_count):
  """
    计算休眠单元掩码

    Args: 
      activation: 激活值,
      outgoing_score: 输出权重休眠得分,
      dead_neurons_threshold: 休眠阈值

    Returns:
      mask: 休眠单元掩码
  """
  # 按batch求激活值绝对值的均值
  # 减少高频低激活单元以及不激活单元
  reduce_axes = list(range(activation.ndim - 1))
  cur_av = jnp.sum(jnp.abs(activation), axis=reduce_axes)
  count = jnp.sum(jnp.abs(activation) > 0, axis=reduce_axes) + last_count
  frac_count = jnp.where(count == 0, 1, count)  # 避免除以0
  total_score = (cur_av + last_av) / frac_count
  mask = total_score <= dead_neurons_threshold
  last_av = jnp.where(mask, 0, last_av+cur_av)
  last_count = jnp.where(mask, 0, count)
  return mask, last_av, last_count



@jax.jit
def create_mask_helper(neuron_mask, current_param, next_param):
  """generate incoming and outgoing weight mask given dead neurons mask.

  Args:
    neuron_mask: mask of size equals the width of a layer.
    current_param: incoming weights of a layer.
    next_param: outgoing weights of a layer.

  Returns:
    incoming_mask
    outgoing_mask
  """
  def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
    """create a mask of weight matrix given 1D vector of neurons mask.

    Args:
      expansion_axis: List contains 1 axis. The dimension to expand the mask
        for dense layers (weight shape 2D).
      expansion_axes: List conrtains 3 axes. The dimensions to expand the
        score for convolutional layers (weight shape 4D).
      param: weight.
      neuron_mask: 1D mask that represents dead neurons(features).

    Returns:
      mask: mask of weight.
    """
    if param.ndim == 2:
      axes = expansion_axis
      # flatten layer
      # The size of neuron_mask is the same as the width of last conv layer.
      # This conv layer will be flatten and connected to dense layer.
      # we repeat each value of a feature map to cover the spatial dimension.
      if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
        num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
        neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
    elif param.ndim == 4:
      axes = expansion_axes
    mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
    for i in range(len(axes)):
      mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
    return mask

  incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
  outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
  return incoming_mask, outgoing_mask


@jax.jit
def compute_effective_rank(intermediates, path='projection/net_act/__call__'):
  # 展平中间结果并提取激活值
  activations_dict = flax.traverse_util.flatten_dict(intermediates, sep='/')
  activation = activations_dict[path][0]
  # 计算奇异值
  sv = jnp.linalg.svd(activation, compute_uv=False)
  # 归一化奇异值
  norm_sv = sv / jnp.sum(jnp.abs(sv))
  # 计算熵
  entropy = -jnp.sum(jnp.where(norm_sv > 0.0, norm_sv * jnp.log(norm_sv), 0.0))
  # 计算有效秩
  effective_rank = jnp.e ** entropy
  return effective_rank


@functools.partial(
    jax.jit,
    static_argnames=(
      'dead_neurons_threshold',
      'init_method_outgoing',
    )
)
def create_masks(
  param_dict,
  target_param_dict,
  activations_dict,
  grad_dict,
  key,
  dead_neurons_threshold,
  init_method_outgoing,
  last_av,
  last_count,
  total_count,
  ):
  """
    构建休眠单元掩码

    Args:
      param_dict: 网络参数字典,
      activations_dict: 激活值字典,
      key: 随机数,
      current_count: 当前休眠计数,
      total_count: 总休眠计数,
      dead_neurons_threshold: 休眠阈值,
      init_method_outgoing: 输出权重初始化方法

    Returns:
      incoming_mask_dict: 输入权重掩码字典, 
      outgoing_mask_dict: 输出权重掩码字典, 
      ingoing_random_keys_dict: 输入随机数字典,
      outgoing_random_keys_dict: 输出随机数字典, 
      param_dict: 网络参数字典, 
      current_count: 当前休眠计数, 
      total_count: 总休眠计数
      log_dict: 休眠单元统计日志字典
  """
  # 设置需要进行休眠单元查询的网络层
  reset_layers_map = {
    'encoder/ResidualStage_0/Conv_0': 'encoder/ResidualStage_0/Conv_1',
    'encoder/ResidualStage_0/Conv_1': 'encoder/ResidualStage_0/Conv_2',
    'encoder/ResidualStage_0/Conv_2': 'encoder/ResidualStage_0/Conv_3',
    'encoder/ResidualStage_0/Conv_3': 'encoder/ResidualStage_0/Conv_4',
    'encoder/ResidualStage_0/Conv_4': 'encoder/ResidualStage_1/Conv_0',
    'encoder/ResidualStage_1/Conv_0': 'encoder/ResidualStage_1/Conv_1',
    'encoder/ResidualStage_1/Conv_1': 'encoder/ResidualStage_1/Conv_2',
    'encoder/ResidualStage_1/Conv_2': 'encoder/ResidualStage_1/Conv_3',
    'encoder/ResidualStage_1/Conv_3': 'encoder/ResidualStage_1/Conv_4',
    'encoder/ResidualStage_1/Conv_4': 'encoder/ResidualStage_2/Conv_0',
    'encoder/ResidualStage_2/Conv_0': 'encoder/ResidualStage_2/Conv_1',
    'encoder/ResidualStage_2/Conv_1': 'encoder/ResidualStage_2/Conv_2',
    'encoder/ResidualStage_2/Conv_2': 'encoder/ResidualStage_2/Conv_3',
    'encoder/ResidualStage_2/Conv_3': 'encoder/ResidualStage_2/Conv_4',
    'encoder/ResidualStage_2/Conv_4': 'projection/net',
    'transition_model/ScanConvTMCell_0/Conv_0': 'transition_model/ScanConvTMCell_0/Conv_1',
    'transition_model/ScanConvTMCell_0/Conv_1': 'projection/net',
    'projection/net': ['head/advantage/net', 'head/value/net', 'predictor']
  }
  # 初始化输入休眠权重掩码
  incoming_mask_dict = {
      k: jnp.zeros_like(p) if p.ndim != 1 else None
      for k, p in param_dict.items()
  }
  # 初始化输出休眠权重掩码
  outgoing_mask_dict = {
      k: jnp.zeros_like(p) if p.ndim != 1 else None
      for k, p in param_dict.items()
  }
  # 初始化输入随机数字典
  ingoing_random_keys_dict = {k: None for k in param_dict}
  # 初始化输出随机数字典
  outgoing_random_keys_dict = {
      k: None for k in param_dict
  } if init_method_outgoing == 'random' else {}
  # 计算输入权重与输出权重的休眠掩码
  for k in reset_layers_map:
    # 获取当前层的输入权重
    param_key = 'params/' + k + '/kernel'
    param = param_dict[param_key]
    # 获取当前层的激活值
    # activation = activations_dict[k + '_act/__call__'][0]
    # 获取当前层的梯度
    grad = grad_dict[param_key]
    # 处理当前层后续与其他模块的连接
    next_keys = (reset_layers_map[k] if isinstance(reset_layers_map[k], list) else [reset_layers_map[k]])
    # 计算当前层的休眠掩码
    # neuron_mask = score2mask(grad, dead_neurons_threshold)
    neuron_mask = score2maskmid(grad)
    # neuron_mask, last_av[k], last_count[k] = score2maskSparse(activation, dead_neurons_threshold, last_av[k], last_count[k])
    # total_count[k] += neuron_mask
    # 计算当前层的休眠单元掩码
    # neuron_mask = jnp.logical_and(neuron_mask, grad_mask)
    # 分别计算不同后续层的权重重置掩码
    for next_k in next_keys:
      # 获取输出权重
      next_param_key = 'params/' + next_k + '/kernel'
      next_param = param_dict[next_param_key]
      # 计算输入权重与输出权重的重置掩码
      incoming_mask, outgoing_mask = create_mask_helper(
          neuron_mask, param, next_param)
      # 将输入权重掩码写入输入权重掩码字典的当前层位置
      incoming_mask_dict[param_key] = incoming_mask
      # 将输出权重掩码写入输出权重掩码字典的下一层位置
      outgoing_mask_dict[next_param_key] = outgoing_mask
      # 添加随机数到输入与输出随机数字典中
      key, subkey = random.split(key)
      ingoing_random_keys_dict[param_key] = subkey
      if init_method_outgoing == 'random':
        key, subkey = random.split(key)
        outgoing_random_keys_dict[next_param_key] = subkey
    # 重置休眠单元的偏置项为0
    bias_key = 'params/' + k + '/bias'
    new_bias = jnp.zeros_like(param_dict[bias_key])
    param_dict[bias_key] = jnp.where(neuron_mask, new_bias, param_dict[bias_key])
    target_param_dict[bias_key] = jnp.where(neuron_mask, new_bias, target_param_dict[bias_key])
  return (incoming_mask_dict, outgoing_mask_dict, ingoing_random_keys_dict,
          outgoing_random_keys_dict, param_dict, target_param_dict, last_av, last_count, total_count)


@functools.partial(
  jax.jit,
  static_argnames=(
    'dead_neurons_threshold',
    'init_method_outgoing',
  )
)
def jit_dnr(
  params,
  target_params,
  opt_state,
  intermediates,
  grad,
  rng,
  last_av,
  last_count,
  total_count,
  dead_neurons_threshold,
  init_method_outgoing,
  sp_weight,
):
  activations_score_dict = flax.traverse_util.flatten_dict(intermediates, sep='/')
  param_dict = flax.traverse_util.flatten_dict(params, sep='/')
  target_param_dict = flax.traverse_util.flatten_dict(target_params, sep='/')
  grad_dict = flax.traverse_util.flatten_dict(grad, sep='/')
  last_av = flax.traverse_util.flatten_dict(last_av, sep='/')
  last_count = flax.traverse_util.flatten_dict(last_count, sep='/')

  # create incoming and outgoing masks and reset bias of dead neurons.
  (
      incoming_mask_dict,
      outgoing_mask_dict,
      incoming_random_keys_dict,
      outgoing_random_keys_dict,
      param_dict,
      target_param_dict,
      last_av,
      last_count,
      total_count,
  ) = create_masks(
      param_dict,
      target_param_dict,
      activations_score_dict,
      grad_dict,
      rng,
      dead_neurons_threshold,
      init_method_outgoing,
      last_av,
      last_count,
      total_count,
  )
  
  params = flax.core.freeze(flax.traverse_util.unflatten_dict(param_dict, sep='/'))
  target_params = flax.core.freeze(flax.traverse_util.unflatten_dict(target_param_dict, sep='/'))
  incoming_random_keys = flax.core.freeze(flax.traverse_util.unflatten_dict(incoming_random_keys_dict, sep='/'))
  if init_method_outgoing == 'random':
    outgoing_random_keys = flax.core.freeze(flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/'))
  # reset incoming weights
  incoming_mask = flax.core.freeze(flax.traverse_util.unflatten_dict(incoming_mask_dict, sep='/'))
  reinit_fn = functools.partial(
      weight_reinit_random,
      weight_scaling=False,
      scale=1,
      weights_type='incoming',
      sp_weight=sp_weight)
  weight_random_reset_fn = jax.jit(functools.partial(jax.tree_map, reinit_fn))
  params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)
  target_params = weight_random_reset_fn(target_params, incoming_mask, incoming_random_keys)

  # reset outgoing weights
  outgoing_mask = flax.core.freeze(flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/'))
  if init_method_outgoing == 'random':
    reinit_fn = functools.partial(
        weight_reinit_random,
        weight_scaling=False,
        scale=1,
        weights_type='outgoing',
        sp_weight=sp_weight)
    # reinit_fn = functools.partial(weight_scale, scale=sp_weight)
    weight_random_reset_fn = jax.jit(functools.partial(jax.tree_map, reinit_fn))
    params = weight_random_reset_fn(params, outgoing_mask, outgoing_random_keys)
    target_params = weight_random_reset_fn(target_params, outgoing_mask, outgoing_random_keys)
  elif init_method_outgoing == 'zero':
    weight_zero_reset_fn = jax.jit(functools.partial(jax.tree_map, weight_reinit_zero))
    params = weight_zero_reset_fn(params, outgoing_mask)
    target_params = weight_zero_reset_fn(target_params, outgoing_mask)
  # else:
  #   raise ValueError(f'Invalid init method: {self.init_method_outgoing}')
  # reset mu, nu of adam optimizer for recycled weights.
  reset_momentum_fn = jax.jit(functools.partial(jax.tree_map, reset_momentum))
  new_mu = reset_momentum_fn(opt_state[0][1], incoming_mask)
  new_mu = reset_momentum_fn(new_mu, outgoing_mask)
  new_nu = reset_momentum_fn(opt_state[0][2], incoming_mask)
  new_nu = reset_momentum_fn(new_nu, outgoing_mask)
  opt_state_list = list(opt_state)
  opt_state_list[0] = optax.ScaleByAdamState(
      opt_state[0].count, mu=new_mu, nu=new_nu)
  opt_state = tuple(opt_state_list)
  return params, target_params, opt_state, last_av, last_count, total_count



@functools.partial(jax.jit, static_argnums=(0), device=jax.local_devices()[0])
def get_intermediates(network_def, support, params, batch):
  # TODO(gsokar) add a check if batch_size equals batch_size_statistics
  # then no need to sample a new batch from buffer.
  def apply_data(x):
    states = x
    filter_rep = lambda l, _: l.name is not None and 'act' in l.name
    return network_def.apply(
        params,
        states,
        do_rollout=False,
        support=support,
        key=jax.random.PRNGKey(0),
        capture_intermediates=filter_rep,
        mutable=['intermediates'])

  _, state = jax.vmap(apply_data)(batch)
  return state['intermediates']


@functools.partial(jax.jit,static_argnames=('dead_neurons_threshold'))
def log_deadunit_and_weight_mag(intermediates, grads, params, last_av, last_count, dead_neurons_threshold=0.0):
  """
    计算休眠单元个数和权重量级

    For conv layer we also log dead elements in the spatial dimension.

    Args:
      intermediates: 每层神经元的激活值.
      params: 网络参数
      threshold: 休眠阈值

    Returns:
      log_dict_elements_per_neuron
      log_dict_neurons
  """
  activations_dict = flax.traverse_util.flatten_dict(intermediates, sep='/')
  param_dict = flax.traverse_util.flatten_dict(params, sep='/')
  grad_dict = flax.traverse_util.flatten_dict(grads, sep='/')
  # 设置需要进行休眠单元查询的网络层
  reset_layers_map = {
    'encoder/ResidualStage_0/Conv_0': 'encoder/ResidualStage_0/Conv_1',
    'encoder/ResidualStage_0/Conv_1': 'encoder/ResidualStage_0/Conv_2',
    'encoder/ResidualStage_0/Conv_2': 'encoder/ResidualStage_0/Conv_3',
    'encoder/ResidualStage_0/Conv_3': 'encoder/ResidualStage_0/Conv_4',
    'encoder/ResidualStage_0/Conv_4': 'encoder/ResidualStage_1/Conv_0',
    'encoder/ResidualStage_1/Conv_0': 'encoder/ResidualStage_1/Conv_1',
    'encoder/ResidualStage_1/Conv_1': 'encoder/ResidualStage_1/Conv_2',
    'encoder/ResidualStage_1/Conv_2': 'encoder/ResidualStage_1/Conv_3',
    'encoder/ResidualStage_1/Conv_3': 'encoder/ResidualStage_1/Conv_4',
    'encoder/ResidualStage_1/Conv_4': 'encoder/ResidualStage_2/Conv_0',
    'encoder/ResidualStage_2/Conv_0': 'encoder/ResidualStage_2/Conv_1',
    'encoder/ResidualStage_2/Conv_1': 'encoder/ResidualStage_2/Conv_2',
    'encoder/ResidualStage_2/Conv_2': 'encoder/ResidualStage_2/Conv_3',
    'encoder/ResidualStage_2/Conv_3': 'encoder/ResidualStage_2/Conv_4',
    'encoder/ResidualStage_2/Conv_4': 'projection/net',
    'transition_model/ScanConvTMCell_0/Conv_0': 'transition_model/ScanConvTMCell_0/Conv_1',
    'transition_model/ScanConvTMCell_0/Conv_1': 'projection/net',
    'projection/net': ['head/advantage/net', 'head/value/net', 'predictor']
  }
  # 初始化神经元计数
  total_neurons, total_deadneurons = 0., 0.
  weight_avg, weight_count = 0, 0
  # 获取log字典
  log_dict = {}
  # 计算输入权重与输出权重的重置掩码
  for k in reset_layers_map.keys():
    # 获取当前层的输入权重
    param_key = 'params/' + k + '/kernel'
    param = param_dict[param_key]
    grad = grad_dict[param_key]
    weight_avg += jnp.sum(jnp.abs(param))
    weight_count += jnp.size(param)
    # 获取当前层的激活值
    # activation = activations_dict[k + '_act/__call__'][0]
    # 处理当前层后续与其他模块的连接
    next_keys = (reset_layers_map[k] if isinstance(reset_layers_map[k], list) else [reset_layers_map[k]])
    # # 获取输出权重得分
    # outgoing_score = jnp.zeros(activation.shape[1])
    for next_k in next_keys:
      # 获取输出权重
      next_param_key = 'params/' + next_k + '/kernel'
      next_param = param_dict[next_param_key]
      weight_avg += jnp.sum(jnp.abs(next_param))
      weight_count += jnp.size(next_param)
    #   outgoing_score += jnp.sum(jnp.abs(next_param), axis=1)
    # 计算当前层的休眠掩码
    # neuron_mask = score2mask(grad, dead_neurons_threshold)
    neuron_mask = score2maskmid(grad)
    # neuron_mask, _, _ = score2maskSparse(activation, dead_neurons_threshold, last_av[k], last_count[k])
    # 计算当前层的休眠单元掩码
    # neuron_mask = jnp.logical_and(neuron_mask, grad_mask)
    # 层大小
    layer_size = neuron_mask.shape[0]
    # 该层休眠单元计数
    deadneurons_count = jnp.count_nonzero(neuron_mask)
    # 总单元数增加
    total_neurons += layer_size
    # 总休眠单元数增加
    total_deadneurons += deadneurons_count
    log_dict[f'dead_feature_percentage/{param_key}'] = deadneurons_count / layer_size * 100.
    log_dict[f'dead_feature_count/{param_key}'] = deadneurons_count

  log_dict[f'feature/total'] = total_neurons
  log_dict[f'feature/deadcount'] = total_deadneurons
  log_dict[f'dead_feature_percentage'] = total_deadneurons / total_neurons * 100.
  log_dict[f'weight_magnitude'] = weight_avg / weight_count
  log_dict[f'grad'] = jnp.mean(jnp.abs(grad_dict['params/projection/net/kernel']))
  grads = []
  for k in grad_dict.keys():
    if 'kernel' in k:
      grads.append(jnp.mean(jnp.abs(grad_dict[k])))
  log_dict['total_grad'] = jnp.mean(jnp.array(grads))
  return log_dict


def write_log(intermediates, grads, params, last_av, last_count, threshold, base_dir):
  """
    获取休眠单元统计数据并将其写入文件

    Args:
      intermediates: 每层神经元的激活值.
      threshold: 休眠阈值
      params: 网络参数
      base_dir: 主文件夹路径

    Returns:
      无
  """
  # 根据阈值统计休眠单元个数
  log_dict = log_deadunit_and_weight_mag(intermediates, grads, params, last_av, last_count, threshold)
  effective_rank = compute_effective_rank(intermediates)
  # print(log_dict)
  # 如果没有休眠单元则直接返回
  if log_dict is None:
    return
  # 否则根据字典读取休眠单元百分比
  stats = []
  header = []
  for k, v in log_dict.items():
      header.append(k)
      stats.append(v)
  header.append('effective_rank')
  stats.append(effective_rank)
  filename = base_dir + '/inter.csv'
  file_exists = os.path.exists(filename)
  # 将统计数据写入文件
  with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
      writer.writerow(header)
    writer.writerow(stats)