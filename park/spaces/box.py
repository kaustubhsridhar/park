import numpy as np

from park import core, logger
from park.spaces.rng import np_random


class Box(core.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low=None, high=None, struct=None, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
            Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low + np.zeros(shape)
            high = high + np.zeros(shape)

        if dtype is None:  # Autodetect type
            if (high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn("park.spaces.Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))

        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        core.Space.__init__(self, struct, shape, dtype)

    def sample(self):
        return np_random.uniform(low=self.low, high=self.high + (0 if self.dtype.kind == 'f' else 1), size=self.low.shape).astype(self.dtype)

    def contains(self, x):
        cd1 = x.shape == self.shape
        if not cd1: print("shape mismatch {} != {}".format(x.shape, self.shape))
        cd2 = (x >= self.low).all()
        if not cd2: print("low violation {} < {}".format(x, self.low))
        cd3 = (x <= self.high).all()
        if not cd3: print("high violation {} > {}".format(x, self.high))
        return cd1 and cd2 and cd3
