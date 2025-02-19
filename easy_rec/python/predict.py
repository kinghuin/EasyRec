# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import logging
import os

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from easy_rec.python.inference.predictor import CSVPredictor
from easy_rec.python.inference.predictor import HivePredictor
from easy_rec.python.main import predict
from easy_rec.python.utils import config_util

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

tf.app.flags.DEFINE_string('input_path', None, 'predict data path')
tf.app.flags.DEFINE_string('output_path', None, 'path to save predict result')
tf.app.flags.DEFINE_integer('batch_size', 1024, help='batch size')

# predict by checkpoint
tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'checkpoint to be evaled '
    ' if not specified, use the latest checkpoint in '
    'train_config.model_dir')
tf.app.flags.DEFINE_string('model_dir', None, help='will update the model_dir')

# predict by saved_model
tf.app.flags.DEFINE_string('saved_model_dir', None, help='save model dir')
tf.app.flags.DEFINE_string(
    'reserved_cols', 'ALL_COLUMNS',
    'columns to keep from input table,  they are separated with ,')
tf.app.flags.DEFINE_string(
    'output_cols', 'ALL_COLUMNS',
    'output columns, such as: score float. multiple columns are separated by ,')
tf.app.flags.DEFINE_string('input_sep', ',', 'separator of predict result file')
tf.app.flags.DEFINE_string('output_sep', chr(1),
                           'separator of predict result file')
tf.app.flags.DEFINE_string('selected_cols', '', '')
tf.app.flags.DEFINE_string('fg_json', '', '')
FLAGS = tf.app.flags.FLAGS


def main(argv):

  if FLAGS.saved_model_dir:
    logging.info('Predict by saved_model.')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path, False)
    if pipeline_config.WhichOneof('train_path') == 'hive_train_input':
      predictor = HivePredictor(
          FLAGS.saved_model_dir,
          pipeline_config.data_config,
          fg_json_path=FLAGS.fg_json_path,
          hive_config=pipeline_config.hive_train_input,
          selected_cols=FLAGS.selected_cols,
          output_sep=FLAGS.output_sep)
    else:
      predictor = CSVPredictor(
          FLAGS.saved_model_dir,
          pipeline_config.data_config,
          fg_json_path=FLAGS.fg_json_path,
          selected_cols=FLAGS.selected_cols,
          input_sep=FLAGS.input_sep,
          output_sep=FLAGS.output_sep)

    logging.info('input_path = %s, output_path = %s' %
                 (FLAGS.input_path, FLAGS.output_path))
    if 'TF_CONFIG' in os.environ:
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_num = len(tf_config['cluster']['worker'])
      task_index = tf_config['task']['index']
    else:
      worker_num = 1
      task_index = 0
    predictor.predict_impl(
        FLAGS.input_path,
        FLAGS.output_path,
        reserved_cols=FLAGS.reserved_cols,
        output_cols=FLAGS.output_cols,
        batch_size=FLAGS.batch_size,
        slice_id=task_index,
        slice_num=worker_num)
  else:
    logging.info('Predict by checkpoint_path.')
    assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
    if FLAGS.model_dir:
      pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
      if file_io.file_exists(pipeline_config_path):
        logging.info('update pipeline_config_path to %s' % pipeline_config_path)
      else:
        pipeline_config_path = FLAGS.pipeline_config_path
    else:
      pipeline_config_path = FLAGS.pipeline_config_path

    pred_result = predict(pipeline_config_path, FLAGS.checkpoint_path,
                          FLAGS.input_path)
    if FLAGS.output_path is not None:
      logging.info('will save predict result to %s' % FLAGS.output_path)
      with tf.gfile.GFile(FLAGS.output_path, 'wb') as fout:
        for k in pred_result:
          fout.write(str(k).replace("u'", '"').replace("'", '"') + '\n')


if __name__ == '__main__':
  tf.app.run()
