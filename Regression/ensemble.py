import numpy as np
from sklearn.neural_network import MLPRegressor
from neupy.algorithms.ensemble.base import BaseEnsemble

__all__ = ('AveragedNetwork',)


class AveragedNetwork(BaseEnsemble):

    def __init__(self, networks):
        super(AveragedNetwork, self).__init__(networks)

    def train(self, input_data, target_data, *args, **kwargs):
        for network in self.networks:
            network.train(input_data, target_data, *args, **kwargs)

    def predict(self, input_data):

        outputs = []
        for i, network in enumerate(self.networks):
            outputs.append(network.predict(input_data))

        return np.mean(outputs, axis=0)


class StackedNetwork(BaseEnsemble):

    def __init__(self, networks):
        self.model = MLPRegressor(
            hidden_layer_sizes=(6,),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=1000,
            learning_rate_init=0.01,
            alpha=0.01
        )
        super(StackedNetwork, self).__init__(networks)

    def train(self, input_data, target_data, pre_train=True, *args, **kwargs):
        new_input_data = ()

        for network in self.networks:
            if not pre_train:
                network.train(input_data, target_data, *args, **kwargs)
            output_data = network.predict(input_data)
            new_input_data = new_input_data + (output_data,)

        new_input_data = np.concatenate(new_input_data, axis=1)
        self.model.fit(new_input_data, target_data)

    def predict(self, input_data):
        new_input_data = ()

        for network in self.networks:
            output_data = network.predict(input_data)
            new_input_data = new_input_data + (output_data,)

        new_input_data = np.concatenate(new_input_data, axis=1)
        return self.model.predict(new_input_data)
