# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np

import symbol_utils

def print_summary(symbol, input_shape):
    arg_name = symbol.list_arguments()
    out_name = symbol.list_outputs()
    arg_shape, out_shape, _ = symbol.infer_shape(data=(1,3,112,112))
    #print({'input' : dict(zip(arg_name, arg_shape)),'output' : dict(zip(out_name, out_shape))})
    in_put =  dict(zip(arg_name, arg_shape))
    out_put = dict(zip(out_name, out_shape))
    print('input: \n')
    for key in in_put:
        print(str(key) + ': ' + str(in_put[key]) + '\n')    
    print('output: \n')
    print(out_put)
    #for key in out_put:
    #print(str(key) + ': ' + str(out_put[key]) + '\n') 

    
def Conv(**kwargs):
    # name = kwargs.get('name')
    # _weight = mx.symbol.Variable(name+'_weight')
    # _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    # body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def mask_linear(data, num_filter, name,**kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter * 1), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                 no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter * 1), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                 no_bias=True, workspace=workspace, name=name + '_conv2')
    mask = Act(data=conv2, act_type='sigmoid', name=name + '_sigmoid')

    return mask


def residual_unit(data, num_filter, stride, bottle_neck, name, **kwargs):
    """Return ResNet Unit symbol for building ResNet
      Parameters
      ----------
      data : str
          Input data
      num_filter : int
          Number of output channels
      bnf : int
          Bottle neck channels factor with regard to num_filter
      stride : tuple
          Stride used in convolution
      dim_match : Boolean
          True means channel number between input and output is the same, otherwise means differ
      name : str
          Base name of the operators
      workspace : int
          Workspace used in convolution operator
      """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)

    # print('in unit3')
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        conv1 = Conv(data=bn1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                     no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                     no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
        conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(stride, stride), pad=(0, 0), no_bias=True,
                     workspace=workspace, name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')

        return bn4


def skip_residual_unit_follow_act(data, num_filter, stride, dim_match, name, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                     no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(stride, stride), pad=(1, 1),
                     no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1,1), pad=(0, 0), no_bias=True,
                     workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = act1
    else:
        conv1sc = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=(stride,stride), no_bias=True,
                           workspace=workspace, name=name + '_conv1sc')
        shortcut = conv1sc
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def skip_residual_unit(data, num_filter, stride, dim_match, name, **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                     no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                     no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(stride, stride), pad=(0, 0), no_bias=True,
                     workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=(stride,stride), no_bias=True,
                           workspace=workspace, name=name + '_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def attention_module_stage1(data, out_channels, **kwargs):
    # 112,112
    attentionA_1_trunk_res2 = skip_residual_unit(data, out_channels, 1, True, "AttentionA_1_trunk_res2_branch1", **kwargs)
    attentionA_1_trunk_res3 = skip_residual_unit(attentionA_1_trunk_res2, out_channels, 1, True, "AttentionA_1_trunk_res3_branch1", **kwargs)

    attentionA_1_mask_down_sample_pool1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    attentionA_1_mask_down_sample_res1_1 = skip_residual_unit(attentionA_1_mask_down_sample_pool1, out_channels, 1, True, "attentionA_1_mask_down_sample_res1_1_branch1", **kwargs)

    #right branch
    attentionA_1_mask_skip_res1 = skip_residual_unit(attentionA_1_mask_down_sample_res1_1, out_channels, 1, True, "attentionA_1_mask_skip_res1_branch1", **kwargs)

    #mask_dowm_sample
    attentionA_1_mask_down_sample_pool2 = mx.sym.Pooling(data=attentionA_1_mask_down_sample_res1_1, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    attentionA_1_mask_down_sample_res2_1 = skip_residual_unit(attentionA_1_mask_down_sample_pool2, out_channels, 1, True, "attentionA_1_mask_down_sample_res2_1_branch1", **kwargs)
    attentionA_1_mask_down_sample_pool3 = mx.sym.Pooling(data=attentionA_1_mask_down_sample_res2_1, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    attentionA_1_mask_down_sample_res_3_1 = skip_residual_unit(attentionA_1_mask_down_sample_pool3, out_channels, 1, True, "attentionA_1_mask_down_sample_res3_1_branch1", **kwargs)

    attentionA_1_mask_down_sample_res_3_2 = skip_residual_unit(attentionA_1_mask_down_sample_res_3_1, out_channels, 1, True, "attentionA_1_mask_down_sample_res3_2_branch1", **kwargs)

    attentionA_1_mask_up_sample_interp_3 = mx.sym.contrib.BilinearResize2D(data=attentionA_1_mask_down_sample_res_3_2, height=28, width=28)

    #mask_skip_res2_branch1
    attentionA_1_mask_skip_res2 = skip_residual_unit(attentionA_1_mask_down_sample_res2_1, out_channels, 1, True, "attentionA_1_mask_skip_res2", **kwargs)
    
    #print_summary(attentionA_1_mask_skip_res2, shape=(1,3,112,112)) 

    attention_A_1_mask_up_sample2 = attentionA_1_mask_up_sample_interp_3 + attentionA_1_mask_skip_res2
    attention_A_1_mask_up_sample_res2_1 = skip_residual_unit(attention_A_1_mask_up_sample2, out_channels, 1, True, "attentionA_1_mask_up_sample_res2_1_branch1", **kwargs)
    attention_A_1_mask_up_sample_interp_2 = mx.sym.contrib.BilinearResize2D(data=attention_A_1_mask_up_sample_res2_1, height=56, width=56)

    attention_A_1_up_sample1 = attentionA_1_mask_skip_res1 + attention_A_1_mask_up_sample_interp_2

    attention_A_1_mask_up_sample_res1_1 = skip_residual_unit(attention_A_1_up_sample1, out_channels, 1, True, "attentionA_1_mask_up_sample_res1_1_branch")


    attention_A_1_mask_up_sample_interp_1 = mx.sym.contrib.BilinearResize2D(data=attention_A_1_mask_up_sample_res1_1, height=112, width=112)

    attention_A_1_mask = mask_linear(attention_A_1_mask_up_sample_interp_1, out_channels, 'attentionA_1_mask', **kwargs)

    attention_A_1_fusion = (1 + attention_A_1_mask) * attentionA_1_trunk_res3

    attentionA_1 = skip_residual_unit(attention_A_1_fusion, out_channels, 1, True, "attentionA_1_branch1")

    return attentionA_1


def attention_module_stage2(data,out_channels,**kwargs):

    attentionB_1_trunk_res2 = skip_residual_unit(data, out_channels, 1, True, 'attentionB_1_trunk_res2_branch1', **kwargs)
    attentionB_1_trunk_res3 = skip_residual_unit(attentionB_1_trunk_res2, out_channels, 1, True, 'attentionB_1_trunk_res3_branch1', **kwargs)

    attentionB_1_mask_down_sample_pool1 = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    attentionB_1_mask_down_sample_res1_1 = skip_residual_unit(attentionB_1_mask_down_sample_pool1, out_channels, 1, True, 'attentionB_1_mask_down_sample_res1_1_branch1', **kwargs)

    attentionB_1_mask_down_sample_pool2 = mx.sym.Pooling(data=attentionB_1_mask_down_sample_res1_1, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='max')

    attentionB_1_mask_down_sample_res2_1 = skip_residual_unit(attentionB_1_mask_down_sample_pool2, out_channels, 1, True, 'attentionB_1_mask_down_sample_res2_1_branch1', **kwargs)
    attentionB_1_mask_down_sample_res2_2 = skip_residual_unit(attentionB_1_mask_down_sample_res2_1, out_channels, 1, True, 'attentionB_1_mask_down_sample_res2_2_branch1', **kwargs)

    attentionB_1_mask_up_sample_interp2 = mx.sym.contrib.BilinearResize2D(data=attentionB_1_mask_down_sample_res2_2, height=28, width=28)

    attentionB_1_mask_skip_res1 = skip_residual_unit(attentionB_1_mask_down_sample_res1_1, out_channels, 1, True, 'attentionB_1_mask_skip_res1_branch1', **kwargs)

    attentionB_1_mask_up_sample1 = attentionB_1_mask_up_sample_interp2 + attentionB_1_mask_skip_res1

    attentionB_1_mask_up_sample_res1_1 = skip_residual_unit(attentionB_1_mask_up_sample1, out_channels, 1, True, 'attentionB_1_mask_up_sample_res1_1_branch1', **kwargs)


    attentionB_1_mask_up_sample_interp_1 = mx.sym.contrib.BilinearResize2D(data=attentionB_1_mask_up_sample_res1_1, height=56, width=56)

    attentionB_1_mask = mask_linear(attentionB_1_mask_up_sample_interp_1, out_channels, 'attentionB_1_mask', **kwargs)

    attentionB_1_fusion = (1 + attentionB_1_mask) * attentionB_1_trunk_res3

    attentionB_1 = skip_residual_unit(attentionB_1_fusion, out_channels, 1, True, 'attentionB_1_branch1', **kwargs)
    return attentionB_1


def attention_module_stage3(data, out_channels,**kwargs):
    # left branch
    attentionC_1_trunk_res2 = skip_residual_unit(data, out_channels, 1, True, 'attentionC_1_trunk_res2_branch1', **kwargs)
    #attentionC_1_trunk_res2 = skip_residual_unit(data, 1024, 1, True, 'attentionC_1_trunk_res2_branch1', **kwargs)

    #attentionC_1_trunk_res3 = skip_residual_unit(attentionC_1_trunk_res2, 1024, 1, True, 'attentionC_1_trunk_res3_branch1', **kwargs)
    attentionC_1_trunk_res3 = skip_residual_unit(attentionC_1_trunk_res2, out_channels, 1, True, 'attentionC_1_trunk_res3_branch1', **kwargs)

    attentionC_1_mask_down_sample_pool1 = mx.sym.Pooling(data=data, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='max')

    attentionC_1_mask_down_sample_res1_1 = skip_residual_unit(attentionC_1_mask_down_sample_pool1, out_channels, 1, True, 'attentionC_1_mask_down_sample_res1_1_branch1', **kwargs)
    #attentionC_1_mask_down_sample_res1_1 = skip_residual_unit(attentionC_1_mask_down_sample_pool1, 1024, 1, True, 'attentionC_1_mask_down_sample_res1_1_branch1', **kwargs)

    attentionC_1_mask_down_sample_res1_2 = skip_residual_unit(attentionC_1_mask_down_sample_res1_1, out_channels, 1, True, 'attentionC_1_mask_down_sample_res1_2_branch1', **kwargs)
    #attentionC_1_mask_down_sample_res1_2 = skip_residual_unit(attentionC_1_mask_down_sample_res1_1, 1024, 1, True, 'attentionC_1_mask_down_sample_res1_2_branch1', **kwargs)

    attentionC_1_mask_up_sample_interp_1 = mx.sym.contrib.BilinearResize2D(data=attentionC_1_mask_down_sample_res1_2, height=28, width=28)

    attentionC_1_mask = mask_linear(attentionC_1_mask_up_sample_interp_1, out_channels, 'attentionC_1_mask', **kwargs)
    #attentionC_1_mask = mask_linear(attentionC_1_mask_up_sample_interp_1, 1024, 'attentionC_1_mask', **kwargs)

    attentionC_1_fusion = (1 + attentionC_1_mask) * attentionC_1_trunk_res3

    attentionC_1 = skip_residual_unit(attentionC_1_fusion, out_channels, 1, True, 'attentionC_1_branch1', **kwargs)
    #attentionC_1 = skip_residual_unit(attentionC_1_fusion, 1024, 1, True, 'attentionC_1_branch1', **kwargs)

    return attentionC_1


def attention_residual_56(num_classes, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_input = kwargs.get('version_input', 1)
    assert version_input >= 0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    act_type = kwargs.get('version_act', 'prelu')
    print(version_input, version_output, version_unit, act_type)
    print('num_classes: %d' %num_classes)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data - 127.5
    data = data * 0.0078125

    # data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if version_input == 0:
        body = Conv(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                    no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = Act(data=body, act_type=act_type, name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    else:
        body = data
        body = Conv(data=body, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = Act(data=body, act_type=act_type, name='relu0')
        # body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    pre_res_1 = skip_residual_unit_follow_act(body, 256, 1, False, 'pre_res_1_branch1', **kwargs)
    attentionA_1_trunk_res1 = skip_residual_unit(pre_res_1, 256, 1, True, 'AttentionA_1_trunk_res1_branch1', **kwargs)
    attentionA_1 = attention_module_stage1(attentionA_1_trunk_res1, 256,**kwargs)
    pre_res_2 = skip_residual_unit_follow_act(attentionA_1, 256, 2, False, 'pre_res_2_branch1', **kwargs)
    attentionB_1_trunk_res1 = skip_residual_unit(pre_res_2, 256, 1, True, 'AttentionB_1_trunk_res1', **kwargs)
    attentionB_1 = attention_module_stage2(attentionB_1_trunk_res1,256, **kwargs)
    pre_res_3 = skip_residual_unit_follow_act(attentionB_1, 256, 2, False, 'pre_res3_branch1', **kwargs)
    #pre_res_3 = skip_residual_unit_follow_act(attentionB_1, 1024, 2, False, 'pre_res3_branch1', **kwargs)
    attentionC_1_trunk_res1 = skip_residual_unit_follow_act(pre_res_3, 256, 1, True, 'AttentionC_1_trunk_res1_branch1', **kwargs)
    #attentionC_1_trunk_res1 = skip_residual_unit_follow_act(pre_res_3, 1024, 1, True, 'AttentionC_1_trunk_res1_branch1', **kwargs)

    attentionC_1 = attention_module_stage3(attentionC_1_trunk_res1, 256, **kwargs)
    post_res_4_1 = skip_residual_unit_follow_act(attentionC_1, 512, 2, False, 'post_res_4_1_branch1', **kwargs)
    #post_res_4_1 = skip_residual_unit_follow_act(attentionC_1, 2048, 2, False, 'post_res_4_1_branch1', **kwargs)
    post_res_4_2 = skip_residual_unit(post_res_4_1, 512, 1, True, 'post_res_4_2_branch1', **kwargs)
    #post_res_4_2 = skip_residual_unit(post_res_4_1, 2048, 1, True, 'post_res_4_2_branch1', **kwargs)
    post_res_4_3 = skip_residual_unit(post_res_4_2, 512, 1, True, 'post_res_4_3_branch1', **kwargs)
    #post_res_4_3 = skip_residual_unit(post_res_4_2, 2048, 1, True, 'post_res_4_3_branch1', **kwargs)
    
    fc1 = symbol_utils.get_fc1(post_res_4_3, num_classes, fc_type)
    print_summary(post_res_4_3, (1,3,112,112))
    #arg_name = post_res_4_3.list_arguments()
    #out_name = post_res_4_3.list_outputs()
    #arg_shape, out_shape, _ = post_res_4_3.infer_shape(data=(1,3,112,112))
    #print({'input' : dict(zip(arg_name, arg_shape)),'output' : dict(zip(out_name, out_shape))})
    return fc1


def get_symbol(num_classes, **kwargs):

    return attention_residual_56(num_classes=num_classes, **kwargs)


if __name__ == '__main__':

    sym = get_symbol(512)
    mx.viz.plot_network(sym)


