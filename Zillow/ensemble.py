import numpy as np

from neupy.algorithms.ensemble.base import BaseEnsemble


__all__ = ('AveragedNetwork',)


class AveragedNetwork(BaseEnsemble):

    def __init__(self, networks):
        super(AveragedNetwork, self).__init__(networks)
        self.weight = None

    def train(self, input_data, target_data, *args, **kwargs):
        for network in self.networks:
            network.train(input_data, target_data, *args, **kwargs)

    def predict(self, input_data):

        outputs = []
        for i, network in enumerate(self.networks):
            outputs.append(network.predict(input_data))

        return np.mean(outputs, axis=0)
