from typing import Iterable, TextIO, Callable, Tuple

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe 

from scipy.io import arff

import numpy as np 
from numpy.typing import ArrayLike


@functional_datapipe('read_arff')
class ReadArffDataPipe(IterDataPipe): 
    def __init__(self, dp: Iterable[Tuple[str, TextIO]], featurenames, labelname): 
        self.dp = dp 
        self.featurenames = featurenames 
        self.labelname = labelname

    def __iter__(self): 
        for path, stream in self.dp: 
            data, metadata = arff.loadarff(stream) 
            yield from data

@functional_datapipe('balance') 
class BalancerDataPipe(IterDataPipe): 
    def __init__(self, dp: Iterable[Tuple[ArrayLike, ArrayLike]], balancer):
        self.dp = dp 
        self.balancer = balancer 
        self.buff = []

    def __iter__(self): 
        self.buff = list(self.dp)
        X, y = zip(*self.buff)
        X, y = np.r_['-1', X], np.r_['c', y] 
        X_res, y_res = self.balancer.fit_resample(X, y) 
        yield from zip(X_res, y_res)

    def __len__(self): 
        raise NotImplemented()
