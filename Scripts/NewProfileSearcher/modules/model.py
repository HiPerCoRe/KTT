from abc import ABC
from pathlib import Path
from typing import cast

import cupy
from numpy.typing import NDArray

import pickle

# ---- Uncomment code in here if you want to use neural models.
# ---- I don't like having a pytorch dependency (not just for the types).
# ---- TODO: remove cupy dependency as well?

# import torch
# import torch.nn as nn


class Model(ABC):
    """
    An abstraction meant to make work with different kinds of models easier.
    Currently unifies scikit and PyTorch API into a single class
    """

    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError


# class NeuralModelAdapter(Model):
#     def __init__(self, model: nn.Module):
#         self.model = model
#         self.device = next(model.parameters()).device
#
#     def predict(self, X: NDArray) -> NDArray:
#         with torch.no_grad():
#             X_tensor = torch.from_numpy(X).to(self.device)
#
#             predictions: torch.Tensor = self.model(X_tensor)
#             return predictions.cpu().numpy()


class XGBModelAdapter(Model):
    def __init__(self, model):
        self.model = model

    def predict(self, X: NDArray) -> NDArray:
        # TODO: test whether CuPy is not killing the profiler
        if self.model.device == 'cuda':
            return self.model.predict(cupy.array(X))

        return self.model.predict(X)


# neuralModels = { 'DeltaNN': DeltaNN }


def loadModel(filepath: str) -> Model:
    """
    Loads a model, infers the models type from the filename. The filename must be in format
    `[gpu-info]_[algorithm]_[type](_[optional-info]).sav`. (e.g. `2080_all_XGBRegressor_counters.sav`).
    Made to support loading both neural pytorch models and scikit-ones, now only supports scikit ones
    and XGBRegressor.
    """
    try:
        modelType = Path(filepath).stem.split('_')[2]
    except IndexError:
        print("Model's filename is in an incompatible format")
        print('The format: [gpu-info]_[algorithm]_[type](_[optional-info]).sav')
        print('For example, 2080_all_XGBRegressor_counters.sav)')

        exit(-1)

    # if modelType in neuralModels:
    #     state_dictionary = torch.load(filepath, weights_only=True)
    #
    #     input_length = -1
    #     for key in state_dictionary:
    #         if 'input_layer.weight' in key:
    #             input_length = state_dictionary[key].shape[1]
    #             break
    #
    #     model = neuralModels[modelType](input_length)
    #     model.load_state_dict(state_dictionary)
    #
    #     accelerator = torch.accelerator.current_accelerator()
    #     device = accelerator.type if accelerator else 'cpu'
    #
    #     model.to(device)
    #
    #     return NeuralModelAdapter(model)

    model = pickle.load(open(filepath, 'rb'))
    if modelType == 'XGBRegressor':
        return XGBModelAdapter(model)

    return cast(Model, model)
