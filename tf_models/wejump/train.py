import tensorflow as tf

from .model import Model
from .data_set import DataSet
from ..base.options import get_options
import traceback
import getopt
import sys

class Usage(Exception):
    pass
        
def main(argv=None):
    try:
        config = ""
        test = False
        if argv is None:
            argv = sys.argv[1:]
            opts, _ = getopt.getopt(argv, "c:", [ "config=", "help", "test" ])
        for opt, value in opts:
            if opt in ("-c", "--config"):
                config = value
            elif opt in ("--help"):
                raise Usage()
            elif opt in ("--test"):
                test = True
            else:
                raise Usage("bad option: {}".format(opt))
        if config == "":
            raise Usage("bad value for option: -c")
        
        options = get_options(config)
        model = Model(options)
        model.load()
        data = DataSet(options)
        if test:
            x, y = data.test_set()
            score = model.score(x, y)
            print("score={}".format(score))
        else:
            model.fit(data)
    except Usage:
        print("{} [ -c | --config config file ] [ --test ] [ --help ]".format(
            sys.argv[0]))
        return 0
    except:
        traceback.print_exc()
        return -1
    
if __name__ == '__main__':
    sys.exit(main())
