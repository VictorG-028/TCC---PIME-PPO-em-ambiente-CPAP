from typing import Generator, Dict, Optional, Type, TypeVar, TypedDict
import numpy as np
import numpy.random as np_random



ParameterName      = str # Parameter name
DistributionName   = str
DistributionInputs = dict[str, np.number]
Distributions      = dict[ParameterName, tuple[DistributionName, DistributionInputs]]

class EnsembleGenerator:

    def __init__(
            self, 
            distributions: Distributions = {},
            seed: Optional[int] = None
            ) -> None:
        """
        [en]
        Ensemble of parameters generator.

        [pt-br]
        Gerador de ensemble de parâmetros.
        """

        self.parameter_names = distributions.keys()
        self.distributions = distributions

        if seed is not None:
            np_random.seed(seed)

        # Create all (name, function) pairs
        # __all__ is a variable inside numpy.random that have all distributions names
        self.distributions_dict: dict[str, callable] = {
            distribution_name: getattr(np_random, distribution_name) 
            for distribution_name in np_random.__all__ 
            if hasattr(np_random, distribution_name)
        }

        def constant_distribution_function(constant: float, size: Optional[int] = None) -> np.ndarray:
            return np.full(size or 1, constant)
        
        self.distributions_dict["constant"] = constant_distribution_function
    
    
    def generate_sample(self) -> dict[float]:
        """
        Gera uma única amostra de parâmetros.

        Returns:
            dict[float]: Um dicionário com os parâmetros gerados.
        """
        sample = {}
        for parameter_name, (distribution_name, distribution_inputs) in self.distributions.items():
            
            if distribution_name not in self.distributions_dict:
                raise ValueError(f"Distribuição desconhecida: {distribution_name}")
            
            if distribution_name == "constant":
                sample[parameter_name] = distribution_inputs["constant"]
                continue
                
            distribution_function = getattr(np_random, distribution_name)
            sample[parameter_name] = distribution_function(**distribution_inputs)
        
        return sample
    

    def generate_ensemble(self, size: int) -> Generator[Dict[str, float], None, None]:
        """
        Gera um ensemble de amostras sob demanda usando um generator.

        Args:
            size (int): Número de amostras no ensemble.

        Yields:
            Dict[str, float]: Uma amostra de parâmetros por iteração.
        """
        for _ in range(size):
            yield self.generate_sample()

