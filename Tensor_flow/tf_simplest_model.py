import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


class DenseNN(tf.Module):
    def __inin__(self, outputs):
        super().__inin__()
        self.outputs = outputs
        self.fl_init = False


a = DenseNN(2)
pass
print(a.outputs)
