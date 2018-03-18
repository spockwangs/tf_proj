import tensorflow as tf

from .model import Model
from .data_set import DataSet
from .train import Trainer
from ..base.options import get_options
import traceback
import getopt
import sys

class Usage(Exception):
    pass
        
def main(argv=None):
    try:
        config = ""
        if argv is None:
            argv = sys.argv[1:]
            opts, _ = getopt.getopt(argv, "c:", [ "config=", "help" ])
        for opt, value in opts:
            if opt in ("-c", "--config"):
                config = value
            elif opt in ("--help"):
                raise Usage()
            else:
                raise Usage("bad option: {}".format(opt))
        if config == "":
            raise Usage("bad value for option: -c")
        
        options = get_options(config)
        sess = tf.Session()
        model = Model(options)
        model.load(sess)
        data = DataSet(options)
        trainer = Trainer(sess, model, data, options)
        trainer.train()
        test_accuracy = sess.run([model.accuracy], feed_dict={
            model.x: data.test_x,
            model.y: data.test_y,
            model.is_training: False
        })
        print("test_accuracy={}".format(test_accuracy))

    except Usage:
        print("{} [ -c | --config config file ] [ --help ]".format(
            sys.argv[0]))
        return 0
    except:
        traceback.print_exc()
        return -1
    
if __name__ == '__main__':
    sys.exit(main())
