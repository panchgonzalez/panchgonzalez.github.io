---
layout: post
title:  "Pruning Mask RCNN Models with TensorFlow"
date:   2020-05-19 08:00:00 -0500
categories: tensorflow model-optimization
comments: true
tag:
---

TL;DR: This is too much work! Use `tf.keras` and `tensorflow_model_optimization` instead.

Here's the [Github repo](https://github.com/panchgonzalez/tf_object_detection_pruning).

#### Magnitude-based weight pruning

Pruning deep learning models has, for a few years now, been an effective way of inducing
sparsity in the model's various connection matrices. This sparsification of weight
matrices has only a marginal impact on overall accuracy while substantially reducing the
model size. Properly pruned *large-sparse* models have been shown to outperform
*small-dense* models [1].

<p align="center">
<img width="65%" src="{{ site.url }}/img/2020-05-19-pruning/pruning_decay.png" />
</p>

Magnitude-based weight pruning gradually zeros out model weights during training,
achieving a target level of model sparsity in the process. These sparse models are
easier to compress, and with additional work can be leveraged to drive down latency.

In this post I'll go over the process of modifying TensorFlow's Object Detection API to
sparsify models during training using the `tf.contrib.model_pruning` API. This
follows the work of Zhu and Gupta, where sparsity is increased from an initial sparsity
value $$ s_i $$ (usually 0) to a final sparsity value $$ s_f $$ over a span of $$ n $$
pruning steps, starting a training step $$ t_0 $$ with a pruning frequency $$ \Delta t $$:

$$
s_t = s_f+(s_i-s_f)\Big(1-\frac{t-t_0}{n\Delta t}\Big)^3\quad\text{for}\quad t\in\left\{t_0,t_0+\Delta t,...,t_0+n\Delta t\right \}
$$

#### Why TensorFlow 1.x?

While many of TensorFlow's most popular APIs have now moved on to TF 2.0, the
[TensorFlow Object Detection
API](https://github.com/tensorflow/models/tree/master/research/object_detection) is
still stuck on TF 1.x due to its dependence on `tf.contrib.slim`. It does look like
there has been an [active
effort](https://github.com/tensorflow/models/issues/6423#issuecomment-600925072) to
migrate the object detection API to TF 2.0 for a few months now, but in the meantime I
imagine a lot of legacy object detection systems are still using TF 1.x.

#### 0. TensorFlow Model Pruning

**A warning to "modern" TensorFlow users**

Before I dive into the world of deprecation land, I should mention that if you're using
`tf.keras` based models then model pruning is made 100% easier with the new [TensorFlow
Model Optimization API](https://www.tensorflow.org/model_optimization). It is as simple
as

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

model = tf.keras.Sequential([...])

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                      initial_sparsity=0.0, final_sparsity=0.5,
                      begin_step=2000, end_step=4000)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule)
...

model_for_pruning.fit(...)
```
That's it! However, if you're still using `tf.slim` based models (like all of TensorFlow's
legacy Object Detection API) then you'll have to keep going.

**Deprecation land: `tf.contrib.model_pruning`**

Hidden away in the legacy TF 1.15 `tf.contrib` library is a [model pruning API](https://github.com/tensorflow/tensorflow/tree/r1.15/tensorflow/contrib/model_pruning) which was written
in conjunction (I think) with [1]. This API helps inject the necessary TensorFlow ops into
the training graph so the model can be pruned during training.

The first step is to add mask and threshold variables to the layers that you want to undergo
pruning. This can be done by wrapping the weight tensor with the `apply_mask` function

```python
conv = tf.nn.conv2d(images, pruning.apply_mask(weights), stride, padding)
```

This creates a convolutional layer with additional variables mask and threshold as shown below.

<p align="center">
<img width="65%" src="{{ site.url }}/img/2020-05-19-pruning/masked_conv.png" />
</p>

Alternatively, you can use one of the provided TensorFlow layer variants with the
auxiliary variables built-in:

- `layers.masked_conv2d`
- `layers.masked_fully_connected`
- `rnn_cells.MaskedLSTMCell`

The second step is to add ops to the training graph that monitor the distribution of
layer's weight magnitudes and determine the layer threshold. This masks all the
weights below determined threshold achieving the target sparsity for a particular train
step. This can be achieved as follows

```python
tf.app.flags.DEFINE_string(
    'pruning_hparams', '',
    """Comma separated list of pruning-related hyperparameters""")

with tf.graph.as_default():

  # Create global step variable
  global_step = tf.train.get_or_create_global_step()

  # Parse pruning hyperparameters
  pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

  # Create a pruning object using the pruning specification
  p = pruning.Pruning(pruning_hparams, global_step=global_step)

  # Add conditional mask update op. Executing this op will update all
  # the masks in the graph if the current global step is in the range
  # [begin_pruning_step, end_pruning_step] as specified by the pruning spec
  mask_update_op = p.conditional_mask_update_op()

  # Add summaries to keep track of the sparsity in different layers during training
  p.add_pruning_summaries()

  with tf.train.MonitoredTrainingSession(...) as mon_sess:
    # Run the usual training op in the tf session
    mon_sess.run(train_op)

    # Update the masks by running the mask_update_op
    mon_sess.run(mask_update_op)
```

#### 1. TF-slim

As an example, let's say we want to sparsify an InceptionV2 based model.

First I'll need to define the default arg scope for the masked inception model in
`models/research/slim/nets/inception_utils.py` by adding the following function

```python
def masked_inception_arg_scope(weight_decay=0.00004,
                               use_batch_norm=True,
                               batch_norm_decay=0.9997,
                               batch_norm_epsilon=0.001,
                               activation_fn=tf.nn.relu,
                               batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                               batch_norm_scale=False):
  """Defines the default arg scope for masked inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': batch_norm_updates_collections,
      # use fused batch norm if possible.
      'fused': None,
      'scale': batch_norm_scale,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope(
    [model_pruning.masked_conv2d, model_pruning.masked_fully_connected],
    weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [model_pruning.masked_conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

```
For an InceptionV2 model endowed with model pruning create a new
`masked_inception_v2.py` model backbone using the model pruning's
`model_pruning.masked_conv2d` layer. The simplest way to do this is make a copy of
`models/research/slim/nets/inception_v2.py` and replace all instances of `slim.conv2d`
with `model_pruning.masked_conv2d`

```python
"""Contains the definition for inception v2 with masked layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_utils

slim = tf.contrib.slim
model_pruning = tf.contrib.model_pruning
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def masked_inception_v2_base(...):

  ...

  with tf.variable_scope(scope, 'InceptionV2', [inputs]):
    with slim.arg_scope(
        [model_pruning.masked_conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

        ...

        net = model_pruning.masked_conv2d(
            inputs,
            depth(64), [7, 7],
            stride=2,
            weights_initializer=trunc_normal(1.0),
            scope=end_point)
  ...

def masked_inception_v2(...):

  ...

  with tf.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = masked_inception_v2_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      ...

      logits = model_pruning.masked_conv2d(
            net, num_classes, [1, 1], activation_fn=None,
            normalizer_fn=None, scope='Conv2d_1c_1x1')
  ...

masked_inception_v2.default_image_size = 224

```
Finally, add a reference to the masked InceptionV2 arg scope at the end of
the `masked_inception_v2.py` file
```python
masked_inception_v2_arg_scope = inception_utils.masked_inception_arg_scope
```

#### 2. Object Detection API

Let's say we want to train and sparsify an InceptionV2-based Mask R-CNN model. With the
`tf.slim`-based InceptionV2 backbone done all I need to do is:

1. Finish specifying the model architecture using `model_pruning.masked_conv2d` and,
2. Add additional hooks that monitor and prune the weight matrices as train.

**Mask R-CNN with Model Pruning**

Here I need to create a masked version of the Faster RCNN feature extractor using the
model pruning API's `model_pruning.masked_conv2d` layers. Similar to the InceptionV2
backbone above, the easiest way is to copy the existing `FasterRCNNInceptionV2FeatureExtractor`
in `faster_rcnn_inception_v2_feature_extractot.py` and create a masked version, replacing
all instances of `slim.conv2d` with `model_pruning.masked_conv2d`

```python
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import inception_v2
from nets import masked_inception_v2

slim = contrib_slim
model_pruning = tf.contrib.model_pruning

class FasterRCNNMaskedInceptionV2FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Masked Inception V2 feature extractor implementation.

  This variant uses a masked version of InceptionV2 which contains both
  auxiliary mask and threshold variables at each layer which will be used for
  model sparsification during training.
  """
  ...

  def _extract_proposal_features(...):

    ...

    with tf.control_dependencies([shape_assert]):
      with tf.variable_scope('InceptionV2',
                             reuse=self._reuse_weights) as scope:
        with _batch_norm_arg_scope(
          [model_pruning.masked_conv2d, slim.separable_conv2d],
          batch_norm_scale=True, train_batch_norm=self._train_batch_norm):
          _, activations = masked_inception_v2.masked_inception_v2_base(
              preprocessed_inputs,
              final_endpoint='Mixed_4e',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)

    return activations['Mixed_4e'], activations

  ...

  def _extract_box_classifier_features(...):

    ...

    with tf.variable_scope('InceptionV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          [model_pruning.masked_conv2d, slim.max_pool2d, slim.avg_pool2d],
          stride=1,
          padding='SAME',
          data_format=data_format):
        with _batch_norm_arg_scope(
          [model_pruning.masked_conv2d, slim.separable_conv2d],
          batch_norm_scale=True, train_batch_norm=self._train_batch_norm):

          with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = model_pruning.masked_conv2d(
                  net, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_0 = model_pruning.masked_conv2d(
                branch_0, depth(192), [3, 3], stride=2, scope='Conv2d_1a_3x3')
      ...
```

**Model Pruning Hook**

Since the Object Detection API models are trained using the `tf.Estimator` framework,
the best way to monitor and prune the models during training is to write a custom
`ModelPruningHook` that wraps the `model_pruning.Pruning` object and calls the
`Pruning.conditional_mask_update_op`.

After a deep dive through the often scant TensorFlow API documentation and even deeper
dive through the TensorFlow source code, here's the hook that seems to get the job done

```python
class ModelPruningHook(tf.train.SessionRunHook):
  """Updates model pruning masks and thresholds during training."""

  def __init__(self, target_sparsity, start_step, end_step):
    """Initializes a `ModelPruningHook`.

    This hooks updates masks to a specified sparsity over a certain number of
    training steps.

    Args:
      target_sparsity: float between 0 and 1 with desired sparsity
      start_step: int step to start pruning
      end_step: int step to end pruning
    """
    tf.logging.info("Create ModelPruningHook.")
    self.pruning_hparams = self._get_pruning_hparams(
      target_sparsity=target_sparsity,
      start_step=start_step,
      end_step=end_step
    )

  def begin(self):
    """Called once before using the session.
    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    """
    self.global_step_tensor = tf.train.get_global_step()
    self.mask_update_op = self._get_mask_update_op()

  def after_run(self, run_context, run_values):
    """Called after each call to run().
    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.
    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.
    If `session.run()` raises any exceptions then `after_run()` is not called.
    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    run_context.session.run(self.mask_update_op)

  def _get_mask_update_op(self):
    """Fetches model pruning mask update op."""
    graph = tf.get_default_graph()
    with graph.as_default():
      pruning = model_pruning.Pruning(
        self.pruning_hparams,
        global_step=self.global_step_tensor
      )
      mask_update_op = pruning.conditional_mask_update_op()
      pruning.add_pruning_summaries()
      return mask_update_op

  def _get_pruning_hparams(self,
                           target_sparsity=0.5,
                           start_step=0,
                           end_step=-1):
    """Get pruning hyperparameters with updated values.

    Args:
      target_sparsity: float between 0 and 1 with desired sparsity
      start_step: int step to start pruning
      end_step: int step to end pruning
    """
    pruning_hparams = model_pruning.get_pruning_hparams()

    # Set the target sparsity
    pruning_hparams.target_sparsity = target_sparsity

    # Set begin pruning step
    pruning_hparams.begin_pruning_step = start_step
    pruning_hparams.sparsity_function_begin_step = start_step

    # Set final pruning step
    pruning_hparams.end_pruning_step = end_step
    pruning_hparams.sparsity_function_end_step = end_step

    return pruning_hparams
```

I tucked this class into `object_detection/hooks/train_hooks.py`.

**Wrapping up**

Finally, just make sure you're instantiating the model pruning hook and pass it to the
`tf.estimator.EstimatorSpec` creation function

```python
# Instantiate hook
model_pruning_hook = train_hooks.ModelPruningHook(
    target_sparsity=FLAGS.sparsity,
    start_step=FLAGS.pruning_start_step,
    end_step=FLAGS.pruning_end_step
)
hooks = [model_pruning_hook]

# Create train and eval specs
train_spec, eval_specs = model_lib.create_train_and_eval_specs(
    train_input_fn,
    eval_input_fns,
    eval_on_train_input_fn,
    predict_input_fn,
    train_steps,
    eval_on_train_data=False,
    hooks=hooks,
    throttle_secs=FLAGS.throttle_secs)
```

#### 3. Model Pruning Patch

After training a sparsified object detection model, you'll probably want to export the
training graph without the pruning nodes. However, the `strip_pruning_vars` utility
provided by the Model Pruning API doesn't quite work off-the-shelf with the object
detection models.

Essentially, an `InvalidArgumentError` is thrown at  when extracting the masked weights
at `strip_pruning_vars_lib._get_masked_weights()` because we're converting the
`image_tensor` placeholder to a constant without initializing them first. This is solved
by passing a dummy value to input tensor

```python
def _get_masked_weights(input_graph_def):
   """Extracts masked_weights from the graph as a dict of {var_name:ndarray}."""
   input_graph = ops.Graph()
   with input_graph.as_default():
     importer.import_graph_def(input_graph_def, name='')

     with session.Session(graph=input_graph) as sess:
       masked_weights_dict = {}
       for node in input_graph_def.node:
         if 'masked_weight' in node.name:
           masked_weight_val = sess.run(
               sess.graph.get_tensor_by_name(_tensor_name(node.name)),
               feed_dict={"image_tensor:0": np.zeros((1,1,1,1))}) #### Add feed_dict for input placeholder
           logging.info(
               '%s has %d values, %1.2f%% zeros \n', node.name,
               np.size(masked_weight_val),
               100 - float(100 * np.count_nonzero(masked_weight_val)) /
               np.size(masked_weight_val))
           masked_weights_dict.update({node.name: masked_weight_val})
   return masked_weights_dict
```

#### 3. Training

To train, just add the additional model pruning flags to `model_main.py`

- `--sparsity`: target sparsity level
- `--pruning_start_step`: start pruning at this training step
- `--pruning_start_step`: stop pruning at this training step

and train as per usual:

```bash
python ${OBJECT_DETECTION_PATH}/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderrcd ../ \
    --throttle_secs=2100 \
    --sparsity=0.85 \
    --pruning_start_step=100000 \
    --pruning_end_step=200000
```

#### Notes

There are a ton of details that I glossed over, but the main idea is all there. A fully
working example can be found in the following
[repo](https://github.com/panchgonzalez/tf_object_detection_pruning). To prune a
different architecture, say MobileNetV1-based SSD model, just rerun through steps 1-2:

1. Create a masked version of the MobileNetV1 backbone `slim/nets/masked_mobilenet_v1.py`
2. Create a masked version of the MobileNetV1 feature extractor `SSDMaskedMobileNetV1FeatureExtractor`

Or save yourself the trouble and start from a `tf.keras` model and use
`tensorflow_model_optimization` instead.

#### References

1. Michael Zhu and Suyog Gupta, “To prune, or not to prune: exploring the efficacy of pruning for model compression”, *2017 NIPS Workshop on Machine Learning of Phones and other Consumer Devices* ([https://arxiv.org/pdf/1710.01878.pdf](https://arxiv.org/pdf/1710.01878.pdf))
2. [TensorFlow Model Pruning API](https://github.com/tensorflow/tensorflow/tree/r1.15/tensorflow/contrib/model_pruning)
