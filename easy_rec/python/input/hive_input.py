# -*- coding: utf-8 -*-

from easy_rec.python.input.input import Input
from easy_rec.python.utils import odps_util
from easy_rec.python.utils.hive_utils import *

try:
  from pyhive import hive
except ImportError:
  logging.warning('pyhive is not installed.')



class HiveManager(object):

  def __init__(self, host, port, username, info, database='default'):
    self.host = host
    self.port = port
    self.username = username
    self.database = database
    self.info = info

  def __call__(self):
    conn = hive.Connection(
      host=self.host,
      port=self.port,
      username=self.username,
      database=self.database)
    cursor = conn.cursor()
    sql = self.info.gen_sql()
    res = []
    for ep in range(self.info.epoch):
      cursor.execute(sql)
      for result in cursor.fetchall():
        res.append(result)
        if len(res) == self.info.batch_size:
          yield res
          res = []
    pass


class HiveInput(Input):
  """Common IO based interface, could run at local or on data science."""

  def __init__(self,
               data_config,
               feature_config,
               input_path,
               task_index=0,
               task_num=1,
               check_mode=False):
    super(HiveInput, self).__init__(data_config, feature_config, input_path,
                                    task_index, task_num, check_mode)
    if input_path is None:
      return
    self._data_config = data_config
    self._feature_config = feature_config
    self._hive_config = input_path
    self._eval_batch_size = data_config.eval_batch_size
    self._fetch_size = self._hive_config.fetch_size

    self._num_epoch = data_config.num_epochs
    self._num_epoch_record = 1


  def _parse_table(self, *fields):
    fields = list(fields)
    inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
    for x in self._label_fids:
      inputs[self._input_fields[x]] = fields[x]
    return inputs

  def _get_batch_size(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      return self._data_config.batch_size
    else:
      return self._eval_batch_size

  def _build(self, mode, params):

    # get input type
    list_type = [self.get_tf_type(x) for x in self._input_field_types]
    list_type = tuple(list_type)
    list_shapes = [tf.TensorShape([None]) for x in range(0, len(list_type))]
    list_shapes = tuple(list_shapes)

    # read odps tables
    self.this_batch_size = self._get_batch_size(mode)

    _hive_read = HiveUtils(data_config=self._data_config,
                           feature_config=self._feature_config,
                           input_path=self._hive_config,
                           this_batch_size=self.this_batch_size)._hive_read

    dataset = tf.data.Dataset.from_generator(
      _hive_read, output_types=list_type, output_shapes=list_shapes)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(
        self._data_config.shuffle_buffer_size,
        seed=2022,
        reshuffle_each_iteration=True)
      dataset = dataset.repeat(self.num_epochs)
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.map(
      self._parse_table,
      num_parallel_calls=self._data_config.num_parallel_calls)

    # preprocess is necessary to transform data
    # so that they could be feed into FeatureColumns
    dataset = dataset.map(
      map_func=self._preprocess,
      num_parallel_calls=self._data_config.num_parallel_calls)

    dataset = dataset.prefetch(buffer_size=self._prefetch_size)

    if mode != tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.map(lambda x:
                            (self._get_features(x), self._get_labels(x)))
    else:
      dataset = dataset.map(lambda x: (self._get_features(x)))
    return dataset
