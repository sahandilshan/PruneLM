import numpy as np


class Prune:

    def __init__(self, model, percentage):
        self.model = model
        self.percentage = percentage * 0.01
        self.threshold = self.__calculateThreshold()
        # print(self.percentage)

    def __flattenParams(self):
        state_dictionary = self.model.state_dict()
        flatten_array = []
        for params in state_dictionary:
            if 'weight' in params:
                weight = state_dictionary[params].detach().clone()
                weight = weight.cpu()
                weight = weight.flatten()
                flatten_array.append(weight)  # stack each layers in a array
        return np.concatenate(flatten_array)  # Flatten out all layers weights

    def __calculateThreshold(self):
        weights = self.__flattenParams()
        weights = np.absolute(weights)  # converts negative weights to positives
        return np.quantile(weights, self.percentage)  # calcualte the quantile

    def modelPruning(self):
        state_dictionary = self.model.state_dict()  # load the state dicd
        for k in state_dictionary:
            if 'weight' in k:  # skip bias
                w = state_dictionary[k]
                state_dictionary[k] = w * (w.abs() > self.threshold)
        print(f'Threshold: {self.threshold}')
        # print(f'Model pruned from {self.threshold * 100}%')
        return state_dictionary
