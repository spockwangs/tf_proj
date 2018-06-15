#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
from ..base.options import get_options
import traceback
import getopt
import sys
from .training import train
from .inputs import inputs
from .evaluation import evaluate

class Usage(Exception):
    pass
        
def main(argv=None):
    try:
        config = ""
        is_training = True
        if argv is None:
            argv = sys.argv[1:]
            opts, _ = getopt.getopt(argv, "c:", [ "config=", "help", "test" ])
        for opt, value in opts:
            if opt in ("-c", "--config"):
                config = value
            elif opt in ("--help"):
                raise Usage()
            elif opt in ("--test"):
                is_training = False
            else:
                raise Usage("bad option: {}".format(opt))
        if config == "":
            raise Usage("bad value for option: -c")
        
        options = get_options(config)
        with tf.device("/cpu:0"):
            data_inputs = inputs(is_training, options)
        if is_training:
            train(options, data_inputs)
        else:
            evaluate(options, model, data_inputs)
    except Usage:
        print("{} [ -c | --config config file ] [ --test ] [ --help ]".format(
            sys.argv[0]))
        return 0
    except:
        traceback.print_exc()
        return -1
    