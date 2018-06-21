#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
from tf_proj.base.options import get_options
import traceback
import getopt
import sys
import os
from .training import train2, train
from .evaluation import evaluate

flags = tf.app.flags
flags.DEFINE_string('config', "tf_proj/models/wejump/config.json", "")
flags.DEFINE_boolean('test', False, "training or evaluation")
FLAGS = flags.FLAGS

class Usage(Exception):
    pass
        
def print_usage(program_name):
    print("{} [ --config config file ] [ --test ] [ --help ]".format(program_name))

def main(argv=None):
    try:
        if argv is None:
            argv = sys.argv

        if FLAGS.config == "":
            raise Usage("bad value for option: -c")
        
        options = get_options(FLAGS.config)
        os.environ['VISIBLE_CUDA_DEVICES'] = ','.join(options.gpus)
        if FLAGS.test:
            evaluate(options)
        else:
            train2(options)
    except Usage:
        print_usage(argv[0])
        return 0
    except:
        traceback.print_exc()
        return -1
    
