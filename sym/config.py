from sym.box import Box
from pprint import pprint

# C is a dict storing all the configuration
C = Box()

# shortcut for C.model
CI = Box()
CM = Box()
CO = Box()


def load_config(fname, verbose=True):
    C.update(C.from_yaml(filename=fname))
    CI.update(C.io)
    CM.update(C.model)
    CO.update(C.optim)
    if verbose:
        pprint(C, indent=4)
