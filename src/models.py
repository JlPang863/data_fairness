from dataclasses import dataclass
from types import SimpleNamespace
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from jax.nn.initializers import normal, he_normal
from typing import Any, Callable, Optional, Sequence, Tuple, Type
import numpy as np


########################################################################################################################
#  SimpleCNN
########################################################################################################################

class MLP(nn.Module):
  features: Sequence[int]
  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):
    # import pdb
    # pdb.set_trace()
    x = x.reshape(x.shape[0], -1)
    for feat in self.features:
      x = nn.Dense(feat, dtype=self.dtype)(x)
      x = nn.relu(x)
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)

    return x


class SimpleCNN(nn.Module):
  num_channels: Sequence[int]
  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):  # train is a dummy argument, model does not have different train and eval modes
    for nc in self.num_channels:
      x = nn.Conv(nc, (5, 5), padding='SAME', dtype=self.dtype, kernel_init=he_normal())(x)
      x = nn.relu(x)
      x = nn.Conv(nc, (3, 3), (2, 2), 'SAME', dtype=self.dtype, kernel_init=he_normal())(x)
      x = nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype, kernel_init=normal())(x)
    return x


########################################################################################################################
#  ResNet18: based on flax and elegy implementations of ResNet V1
########################################################################################################################

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm()(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  lowres: bool = True
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    # norm = partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
    norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype)

    x = conv(self.num_filters,
             (3, 3) if self.lowres else (7, 7),
             (1, 1) if self.lowres else (2, 2),
             padding='SAME',
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    if not self.lowres:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i, strides=strides, conv=conv, norm=norm, act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)


########################################################################################################################
#  Vision Transformer
########################################################################################################################


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.
  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.
    Args:
      inputs: Inputs to the layer.
    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.
  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.
    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.
    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.
  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.
    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.
    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  resnet: Optional[Any] = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  head_bias_init: float = 0.
  encoder: Type[nn.Module] = Encoder
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train=True):

    x = inputs
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add ƒit here.
      if self.classifier == 'token':
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier == 'unpooled':
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      logits = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
      return logits, x # logits, embedding
    else:
      return x # logits


# class ViT_last(nn.Module):
#   """VisionTransformer."""

#   num_classes: int
#   hidden_size: int
#   head_bias_init: float = 0.
  


#   @nn.compact
#   def __call__(self, inputs, *, train=True):

#     x = inputs
#     if self.num_classes:
#       x = nn.Dense(
#           features=self.num_classes,
#           name='head',
#           kernel_init=nn.initializers.zeros,
#           bias_init=nn.initializers.constant(self.head_bias_init))(x)
#       return x # logits
#     else:
#       raise ValueError('num_class invalid')

ViT_S8 = partial(
  VisionTransformer, 
  hidden_size=384, 
  patches=SimpleNamespace(size=(32, 32)), 
  transformer=dict(
    attention_dropout_rate = 0.0,
    dropout_rate = 0.0,
    mlp_dim = 1536,
    num_heads = 6,
    num_layers = 12,    
  ),
  model_name="ViT-S/8"
)

ViT_B8 = partial(
  VisionTransformer, 
  hidden_size=768, 
  patches=SimpleNamespace(size=(32, 32)),
  transformer=dict(
    attention_dropout_rate = 0.0,
    dropout_rate = 0.0,
    mlp_dim = 3072,
    num_heads = 12,
    num_layers = 12,    
  ),
  model_name="ViT-B/8"
)

ViT_B8_lowres = partial(
  VisionTransformer, 
  hidden_size=768, 
  patches=SimpleNamespace(size=(8, 8)), 
  transformer=dict(
    attention_dropout_rate = 0.0,
    dropout_rate = 0.0,
    mlp_dim = 3072,
    num_heads = 12,
    num_layers = 12,    
  ),
  model_name="ViT-B/8"
)
# ViT_B8_linear = partial(
#   ViT_last, 
#   hidden_size=768, 
# )

ViT_L16 = partial(
  VisionTransformer, 
  hidden_size=1024, 
  patches=SimpleNamespace(size=(16, 16)), 
  transformer=dict(
    attention_dropout_rate = 0.0,
    dropout_rate = 0.1,
    mlp_dim = 4096,
    num_heads = 16,
    num_layers = 24,    
  ),
  model_name="ViT-L/16"
)


########################################################################################################################
#  Text Classification
########################################################################################################################

class TextClassifier(nn.Module):
  features: Sequence[int]
  num_classes: int
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):
    x = x.reshape(x.shape[0], -1)
    for feat in self.features:
      x = nn.Dense(feat, dtype=self.dtype)(x)
      x = nn.relu(x)
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    return x


########################################################################################################################
#  Utils
########################################################################################################################

def get_model(args):
  # linear_flag = False
  model_linear = None
  if args.model == 'mlp':
    if args.dataset == 'jigsaw':
      model = MLP(features=[256], num_classes=args.num_classes)  # 
    elif args.dataset == 'adult':
      model = MLP(features=[64], num_classes=args.num_classes)  # 
    else:
      model = MLP(features=[64], num_classes=args.num_classes)
  elif args.model == 'vit':
    model = ViT_S8(num_classes=args.num_classes)
  elif args.model == 'vit-b_8':
    model = ViT_B8(num_classes=args.num_classes)
  elif args.model == 'vit-b_8_lowres':
    model = ViT_B8_lowres(num_classes=args.num_classes)
  elif args.model == 'vit-l_16':
    model = ViT_L16(num_classes=args.num_classes)
  elif args.model == 'mlp_3_layer':
    model = MLP(features=[256, 16], num_classes=args.num_classes)
  elif args.model == 'resnet18_lowres':
    model = ResNet18(num_classes=args.num_classes, lowres=True)
  elif args.model == 'resnet50_lowres':
    model = ResNet50(num_classes=args.num_classes, lowres=True)
  elif args.model == 'simple_cnn_0':
    model = SimpleCNN(num_channels=[16, 64, 256], num_classes=args.num_classes)
  else:
    raise NotImplementedError
  return model
  # if linear_flag:
  # return model, model_linear
  # else:
  #   raise NotImplementedError


def get_num_params(params):
  return int(sum([np.prod(w.shape) for w in tree_flatten(params)[0]]))


def get_apply_fn_test(model):
  def apply_fn_test(params, model_state, x):
    vs = {'params': params, **model_state}
    logits = model.apply(vs, x, train=False, mutable=False)
    return logits
  return apply_fn_test


def get_apply_fn_train(model):
  def apply_fn_train(params, model_state, x):
    vs = {'params': params, **model_state}
    logits, model_state = model.apply(vs, x, mutable=list(model_state.keys()))
    return logits, model_state
  return apply_fn_train