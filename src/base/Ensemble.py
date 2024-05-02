from typing import Optional
import numpy.random as np_random
from numpy import array as np_array
from numpy import number as np_number

Map          = dict
Name         = str
Distribution = str
Inputs       = Map[str, np_number]

class Ensemble:

    def __init__(
            self, 
            size: int = 1,
            # parameter_names: list[str] = [],
            distributions: Map[Name, tuple[Distribution, Inputs]] = {},
            seed: Optional[int] = None
            ) -> None:
        assert size > 0, "Size must be positive"

        if seed is not None:
            np_random.seed(seed)
        
        self.size = size
        self.parameter_names = distributions.keys()
        self.distributions = distributions
        self.i = 0

        self.parameters = self.__initialize_parameters()
        

    def __initialize_parameters(self) -> dict[str, float]:

        # Create all (name, function) pairs
        # __all__ is a variable inside numpy.random that have all distributions names
        distributions_dict: dict[str, callable] = {
            distribution_name: getattr(np_random, distribution_name) 
            for distribution_name in np_random.__all__ 
            if hasattr(np_random, distribution_name)
        }

        # Add constant (name, function) pair
        distributions_dict["constant"] = lambda constant, size: np_array([constant] * size)

        parameters = {}
        for parameter_name, (distribution_name, distribution_inputs) in self.distributions.items():
            
            # # some_dict.pop(k, default) returns the value associated with key k or default if not found
            # distribution_name = distribution_inputs.pop('distribution', 'uniform')
            
            if distribution_name not in distributions_dict.keys():
                raise ValueError(f"Distribuição desconhecida para o parâmetro {parameter_name}: {distribution_name}")
            

            distribution_function = distributions_dict[distribution_name]
            parameters[parameter_name] = distribution_function(**distribution_inputs, size=self.size)

        return parameters


    def next_param_set(self) -> None:

        self.i += 1

        if self.i == len(self.parameter_names):
            self.i = 0

    
    def get_param_set(self) -> dict[float]:
        param_set = {}
        for parameter_name in self.parameter_names:
            param_set[parameter_name] = self.parameters[parameter_name][self.i]

        return param_set
