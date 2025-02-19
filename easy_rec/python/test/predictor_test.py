# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import json
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from easy_rec.python.inference.predictor import CSVPredictor
from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.utils import config_util
from easy_rec.python.utils import test_utils
from easy_rec.python.utils.test_utils import RunAsSubprocess


class PredictorTest(tf.test.TestCase):

  def setUp(self):
    self.gpus = test_utils.get_available_gpus()
    self.assertTrue(len(self.gpus) > 0, 'no available gpu on this machine')
    logging.info('available gpus %s' % self.gpus)
    test_utils.set_gpu_id(self.gpus[0])
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))
    self._test_path = 'data/test/inference/taobao_infer_data.txt'

  def tearDown(self):
    test_utils.set_gpu_id(None)

  @RunAsSubprocess
  def test_pred_list(self):
    predictor = Predictor('data/test/inference/tb_multitower_export/')
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[2:])
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 100)

  @RunAsSubprocess
  def test_lookup_pred(self):
    predictor = Predictor('data/test/inference/lookup_export')
    lookup_pred_path = 'data/test/inference/lookup_data_test80.csv'
    with open(lookup_pred_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[1:])
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 80)

  @RunAsSubprocess
  def test_pred_dict(self):
    predictor = Predictor('data/test/inference/tb_multitower_export/')
    field_keys = [
        'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
        'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'occupation',
        'new_user_class_level', 'tag_category_list', 'tag_brand_list', 'price'
    ]
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append({f: row[fid + 2] for fid, f in enumerate(field_keys)})
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 100)

  @RunAsSubprocess
  def test_pred_placeholder_named_by_input(self):
    predictor = Predictor(
        'data/test/inference/tb_multitower_placeholder_rename_export/')
    field_keys = [
        'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
        'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'occupation',
        'new_user_class_level', 'tag_category_list', 'tag_brand_list', 'price'
    ]
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        line_input = {}
        for fid, f in enumerate(field_keys):
          if f in ['tag_category_list', 'tag_brand_list']:
            line_input[f] = ['12', '23']
          else:
            line_input[f] = row[fid + 2]
        inputs.append(line_input)
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 100)

  @RunAsSubprocess
  def test_fm_pred_list(self):
    predictor = Predictor('data/test/inference/fm_export/')
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append(row[2:])
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 100)

  @RunAsSubprocess
  def test_fm_pred_dict(self):
    predictor = Predictor('data/test/inference/fm_export/')
    field_keys = [
        'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
        'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'occupation',
        'new_user_class_level', 'tag_category_list', 'tag_brand_list', 'price'
    ]
    with open(self._test_path, 'r') as fin:
      reader = csv.reader(fin)
      inputs = []
      for row in reader:
        inputs.append({f: row[fid + 2] for fid, f in enumerate(field_keys)})
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 100)


class PredictorTestOnDS(tf.test.TestCase):

  def setUp(self):

    self._test_dir = test_utils.get_tmp_dir()
    self._test_output_path = None
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    if self._test_output_path and (os.path.exists(self._test_output_path)):
      shutil.rmtree(self._test_output_path)
    test_utils.set_gpu_id(None)

  @RunAsSubprocess
  def test_local_pred(self):
    test_input_path = 'data/test/inference/taobao_infer_data.txt'
    self._test_output_path = os.path.join(self._test_dir, 'taobao_infer_result')
    save_model_dir = 'data/test/inference/tb_multitower_export/'
    pipeline_config_path = os.path.join(save_model_dir,
                                        'assets/pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, False)
    predictor = CSVPredictor(
        save_model_dir,
        pipeline_config.data_config,
        input_sep=',',
        output_sep=';',
        selected_cols='')

    predictor.predict_impl(
        test_input_path,
        self._test_output_path,
        reserved_cols='ALL_COLUMNS',
        output_cols='ALL_COLUMNS',
        slice_id=0,
        slice_num=1)
    header_truth = 'logits;probs;clk;buy;pid;adgroup_id;cate_id;campaign_id;customer;'\
                   'brand;user_id;cms_segid;cms_group_id;final_gender_code;age_level;pvalue_level;' \
                   'shopping_level;occupation;new_user_class_level;tag_category_list;tag_brand_list;price'

    with open(self._test_output_path + '/slice_0.csv', 'r') as f:
      output_res = f.readlines()
      self.assertTrue(len(output_res) == 101)
      self.assertEqual(output_res[0].strip(), header_truth)

  @RunAsSubprocess
  def test_local_pred_with_part_col(self):
    test_input_path = 'data/test/inference/taobao_infer_data.txt'
    self._test_output_path = os.path.join(self._test_dir, 'taobao_infer_result')
    save_model_dir = 'data/test/inference/tb_multitower_export/'
    pipeline_config_path = os.path.join(save_model_dir,
                                        'assets/pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, False)

    predictor = CSVPredictor(
        save_model_dir,
        pipeline_config.data_config,
        input_sep=',',
        output_sep=';',
        selected_cols='')

    predictor.predict_impl(
        test_input_path,
        self._test_output_path,
        reserved_cols='clk,buy,user_id,adgroup_id',
        output_cols='probs',
        slice_id=0,
        slice_num=1)
    header_truth = 'probs;clk;buy;user_id;adgroup_id'

    with open(self._test_output_path + '/slice_0.csv', 'r') as f:
      output_res = f.readlines()
      self.assertTrue(len(output_res) == 101)
      self.assertEqual(output_res[0].strip(), header_truth)

  @RunAsSubprocess
  def test_local_pred_rtp(self):
    test_input_path = 'data/test/inference/taobao_infer_rtp_data.txt'
    self._test_output_path = os.path.join(self._test_dir,
                                          'taobao_test_feature_result')
    save_model_dir = 'data/test/inference/tb_multitower_rtp_export/'
    pipeline_config_path = os.path.join(save_model_dir,
                                        'assets/pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, False)

    predictor = CSVPredictor(
        save_model_dir,
        pipeline_config.data_config,
        input_sep=';',
        output_sep=';',
        selected_cols='0,3')
    predictor.predict_impl(
        test_input_path,
        self._test_output_path,
        reserved_cols='ALL_COLUMNS',
        output_cols='ALL_COLUMNS',
        slice_id=0,
        slice_num=1)
    header_truth = 'logits;probs;clk;no_used_1;no_used_2;features'
    with open(self._test_output_path + '/slice_0.csv', 'r') as f:
      output_res = f.readlines()
      self.assertTrue(len(output_res) == 101)
      self.assertEqual(output_res[0].strip(), header_truth)

  @RunAsSubprocess
  def test_local_pred_rtp_with_part_col(self):
    test_input_path = 'data/test/inference/taobao_infer_rtp_data.txt'
    self._test_output_path = os.path.join(self._test_dir,
                                          'taobao_test_feature_result')
    save_model_dir = 'data/test/inference/tb_multitower_rtp_export/'
    pipeline_config_path = os.path.join(save_model_dir,
                                        'assets/pipeline.config')
    pipeline_config = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, False)

    predictor = CSVPredictor(
        save_model_dir,
        pipeline_config.data_config,
        input_sep=';',
        output_sep=';',
        selected_cols='0,3')
    predictor.predict_impl(
        test_input_path,
        self._test_output_path,
        reserved_cols='clk,features,no_used_1',
        output_cols='ALL_COLUMNS',
        slice_id=0,
        slice_num=1)
    header_truth = 'logits;probs;clk;features;no_used_1'
    with open(self._test_output_path + '/slice_0.csv', 'r') as f:
      output_res = f.readlines()
      self.assertTrue(len(output_res) == 101)
      self.assertEqual(output_res[0].strip(), header_truth)


class PredictorTestV2(tf.test.TestCase):

  def setUp(self):
    self.gpus = test_utils.get_available_gpus()
    self.assertTrue(len(self.gpus) > 0, 'no available gpu on this machine')
    logging.info('available gpus %s' % self.gpus)
    test_utils.set_gpu_id(self.gpus[0])
    logging.info('Testing %s.%s' % (type(self).__name__, self._testMethodName))

  def tearDown(self):
    test_utils.set_gpu_id(None)

  @RunAsSubprocess
  def test_pred_multi(self):
    predictor = Predictor('data/test/inference/fg_export_multi')
    test_path = 'data/test/rtp/taobao_test_feature.txt'
    with open(test_path, 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        line_toks = line_str.split(';')
        feature = line_toks[-1]
        feature = feature.split('\002')
        inputs.append(feature)
      output_res = predictor.predict(inputs, batch_size=32)
      self.assertTrue(len(output_res) == 10000)
    with open('data/test/rtp/taobao_fg_pred.out', 'r') as fin:
      for line_id, line_str in enumerate(fin):
        line_str = line_str.strip()
        line_pred = json.loads(line_str)
        self.assertTrue(
            np.abs(line_pred['probs'] - output_res[line_id]['probs']) < 5e-6)

  @RunAsSubprocess
  def test_pred_single(self):
    predictor = Predictor('data/test/inference/fg_export_single')
    test_path = 'data/test/rtp/taobao_test_feature.txt'
    with open(test_path, 'r') as fin:
      inputs = []
      for line_str in fin:
        line_str = line_str.strip()
        line_toks = line_str.split(';')
        feature = line_toks[-1]
        inputs.append(feature)
      output_res = predictor.predict(inputs, batch_size=32)
    with open('data/test/rtp/taobao_fg_pred.out', 'r') as fin:
      for line_id, line_str in enumerate(fin):
        line_str = line_str.strip()
        line_pred = json.loads(line_str)
        self.assertTrue(
            np.abs(line_pred['probs'] - output_res[line_id]['probs']) < 5e-5)


if __name__ == '__main__':
  tf.test.main()
