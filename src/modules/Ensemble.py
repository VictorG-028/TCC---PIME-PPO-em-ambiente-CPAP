from typing import Callable, Dict, Literal, NotRequired, Optional, Type, TypeVar, TypedDict
import numpy.random as np_random
from numpy import array as np_array
from numpy import number as np_number
from functools import partial
import inspect

T = TypeVar("T")
Map          = dict
Name         = str # Parameter name
Distribution = str
Inputs       = Map[str, np_number]


# # Manual specification of parameter names for known distributions
# DISTRIBUTIONS_PARAMETERS = {
#     'beta': {'a': np_number, 'b': np_number},
#     'binomial': {'n': np_number, 'p': np_number},
#     'chisquare': {'df': np_number},
#     'dirichlet': {'alpha': np_number},
#     'exponential': {'scale': np_number},
#     'f': {'dfnum': np_number, 'dfden': np_number},
#     'gamma': {'shape': np_number},
#     'geometric': {'p': np_number},
#     'gumbel': {'loc': np_number, 'scale': np_number},
#     'hypergeometric': {'ngood': np_number, 'nbad': np_number, 'nsample': np_number},
#     'laplace': {'loc': np_number, 'scale': np_number},
#     'logistic': {'loc': np_number, 'scale': np_number},
#     'lognormal': {'mean': np_number, 'sigma': np_number},
#     'logseries': {'p': np_number},
#     'multinomial': {'n': np_number, 'pvals': np_number},
#     'multivariate_normal': {'mean': np_number, 'cov': np_number},
#     'negative_binomial': {'n': np_number, 'p': np_number},
#     'noncentral_chisquare': {'df': np_number, 'nonc': np_number},
#     'noncentral_f': {'dfnum': np_number, 'dfden': np_number, 'nonc': np_number},
#     'normal': {'loc': np_number, 'scale': np_number},
#     'pareto': {'a': np_number},
#     'poisson': {'lam': np_number},
#     'power': {'a': np_number},
#     'rayleigh': {'scale': np_number},
#     'standard_cauchy': {},
#     'standard_exponential': {},
#     'standard_gamma': {'shape': np_number},
#     'standard_normal': {},
#     'standard_t': {'df': np_number},
#     'triangular': {'left': np_number, 'mode': np_number, 'right': np_number},
#     'uniform': {'low': np_number, 'high': np_number},
#     'vonmises': {'mu': np_number, 'kappa': np_number},
#     'wald': {'mean': np_number, 'scale': np_number},
#     'weibull': {'a': np_number},
#     'zipf': {'a': np_number}
# }

### Failed attemp to automaticaly generate TypedDict classes based on numpy functions signatures
# # Function to dynamically create TypedDicts for each numpy.random distribution
# def generate_distributions_typedicts() -> dict:

#     def create_typeddict_class(name: str, fields: dict[str, type]) -> any:
#         return TypedDict(name, fields)

#     distributions_dict = {}

#     for distribution_name, params in DISTRIBUTIONS_PARAMETERS.items():
#         typed_dict_class = create_typeddict_class(distribution_name.capitalize() + 'TypedDict', params)
#         distributions_dict[distribution_name] = typed_dict_class
#         # print(f"Added distribution: {distribution_name} with fields: {params}")

#     # Add constant distribution manually since its not form numpy
#     distributions_dict["constant"] = create_typeddict_class("ConstantTypedDict", {"constant": np_number})
#     return distributions_dict

# DistributionsDictTypes = generate_distributions_typedicts()
# print(DistributionsDictTypes.keys())
# print()

# test: DistributionsDictTypes["gamma"] = {
#     "shape": 10,
# }

class BetaTypedDict(TypedDict):
    a: np_number
    b: np_number

class BinomialTypedDict(TypedDict):
    n: np_number
    p: np_number

class ChisquareTypedDict(TypedDict):
    df: np_number

class DirichletTypedDict(TypedDict):
    alpha: np_number

class ExponentialTypedDict(TypedDict):
    scale: np_number

class FTypedDict(TypedDict):
    dfnum: np_number
    dfden: np_number

class GammaTypedDict(TypedDict):
    shape: np_number

class GeometricTypedDict(TypedDict):
    p: np_number

class GumbelTypedDict(TypedDict):
    loc: np_number
    scale: np_number

class HypergeometricTypedDict(TypedDict):
    ngood: np_number
    nbad: np_number
    nsample: np_number

class LaplaceTypedDict(TypedDict):
    loc: np_number
    scale: np_number

class LogisticTypedDict(TypedDict):
    loc: np_number
    scale: np_number

class LognormalTypedDict(TypedDict):
    mean: np_number
    sigma: np_number

class LogseriesTypedDict(TypedDict):
    p: np_number

class MultinomialTypedDict(TypedDict):
    n: np_number
    pvals: np_number

class MultivariateNormalTypedDict(TypedDict):
    mean: np_number
    cov: np_number

class NegativeBinomialTypedDict(TypedDict):
    n: np_number
    p: np_number

class NoncentralChisquareTypedDict(TypedDict):
    df: np_number
    nonc: np_number

class NoncentralFTypedDict(TypedDict):
    dfnum: np_number
    dfden: np_number
    nonc: np_number

class NormalTypedDict(TypedDict):
    loc: np_number
    scale: np_number

class ParetoTypedDict(TypedDict):
    a: np_number

class PoissonTypedDict(TypedDict):
    lam: np_number

class PowerTypedDict(TypedDict):
    a: np_number

class RayleighTypedDict(TypedDict):
    scale: np_number

class StandardCauchyTypedDict(TypedDict):
    pass

class StandardExponentialTypedDict(TypedDict):
    pass

class StandardGammaTypedDict(TypedDict):
    shape: np_number

class StandardNormalTypedDict(TypedDict):
    pass

class StandardTTypedDict(TypedDict):
    df: np_number

class TriangularTypedDict(TypedDict):
    left: np_number
    mode: np_number
    right: np_number

class UniformTypedDict(TypedDict):
    low: np_number
    high: np_number

class VonmisesTypedDict(TypedDict):
    mu: np_number
    kappa: np_number

class WaldTypedDict(TypedDict):
    mean: np_number
    scale: np_number

class WeibullTypedDict(TypedDict):
    a: np_number

class ZipfTypedDict(TypedDict):
    a: np_number

class ConstantTypedDict(TypedDict):
    constant: np_number

# VALID_DISTRIBUTIONS is generated by filtering dir(np_random)
# print(dir(np_random))
VALID_DISTRIBUTIONS_NAMES = [
    'beta', 'binomial', 'chisquare', 'dirichlet', 'exponential', 'f', 'gamma',
    'geometric', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal',
    'logseries', 'multinomial', 'multivariate_normal', 'negative_binomial',
    'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'poisson', 'power',
    'rand', 'randint', 'randn', 'random', 'rayleigh', 'standard_cauchy',
    'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t',
    'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf'
]

VALID_DISTRIBUTIONS_TYPES: Dict[str, Type[T]] = {
    'beta': BetaTypedDict,
    'binomial': BinomialTypedDict,
    'chisquare': ChisquareTypedDict,
    'dirichlet': DirichletTypedDict,
    'exponential': ExponentialTypedDict,
    'f': FTypedDict,
    'gamma': GammaTypedDict,
    'geometric': GeometricTypedDict,
    'gumbel': GumbelTypedDict,
    'hypergeometric': HypergeometricTypedDict,
    'laplace': LaplaceTypedDict,
    'logistic': LogisticTypedDict,
    'lognormal': LognormalTypedDict,
    'logseries': LogseriesTypedDict,
    'multinomial': MultinomialTypedDict,
    'multivariate_normal': MultivariateNormalTypedDict,
    'negative_binomial': NegativeBinomialTypedDict,
    'noncentral_chisquare': NoncentralChisquareTypedDict,
    'noncentral_f': NoncentralFTypedDict,
    'normal': NormalTypedDict,
    'pareto': ParetoTypedDict,
    'poisson': PoissonTypedDict,
    'power': PowerTypedDict,
    'rayleigh': RayleighTypedDict,
    'standard_cauchy': StandardCauchyTypedDict,
    'standard_exponential': StandardExponentialTypedDict,
    'standard_gamma': StandardGammaTypedDict,
    'standard_normal': StandardNormalTypedDict,
    'standard_t': StandardTTypedDict,
    'triangular': TriangularTypedDict,
    'uniform': UniformTypedDict,
    'vonmises': VonmisesTypedDict,
    'wald': WaldTypedDict,
    'weibull': WeibullTypedDict,
    'zipf': ZipfTypedDict,

    # Add your custom distribution types here, following the same structure
    'constant': ConstantTypedDict,  # Example custom distribution
}

################################################################################

class Distribution:
    _VALID_DISTRIBUTIONS = VALID_DISTRIBUTIONS_TYPES

    def __init__(self):
        self._name: Optional[str] = None
        self.params: Optional[Dict[str, T]] = None

    def name(self, name: str) -> "Distribution":
        if not isinstance(name, str) and len(name) == 0:
            raise AttributeError(f"Distribution name must be a non-empty string.")
        
        if name not in VALID_DISTRIBUTIONS_NAMES:
            raise AttributeError(f"Distribution '{name}' not found in VALID_DISTRIBUTIONS.")
        
        self._name = name
        return self
    

    def register_distribution(self, name: str, distribution_class):
        """
        Registers a new distribution type for use with the `name` method.

        Args:
            name (str): The name of the distribution type to register.
            distribution_class (class): The class representing the distribution type.

        Raises:
            TypeError: If `distribution_class` is not a subclass of `Distribution`.
        """

        if not issubclass(distribution_class, Distribution):
            raise TypeError("Registered distribution class must be a subclass of Distribution.")

        self._VALID_DISTRIBUTIONS[name] = distribution_class


    def __getattr__(self, name: str) -> any:
        """
        Provides a way to dynamically access methods for registered distributions.

        Args:
            name (str): The name of the distribution method.

        Returns:
            Any: The method associated with the distribution name, or raises
                 AttributeError if not found.
        """
        print(f"attribute searched = {name}")
        if name in self._VALID_DISTRIBUTIONS:
            # Delegate method calls to the specific distribution type
            return getattr(self._VALID_DISTRIBUTIONS[name](), name)

        raise AttributeError(f"Distribution method '{name}' not found.")

    def __repr__(self):
        return f"Distribution(name={self._name}, params={self.params})"

# Distribution("uniform").low(0).high(10)
# Distribution("gamma").shpae(6)
# # Objective Usage
# ref = Distribution()
# print(ref.name("uniform").low(0).high(1.0))
# print(Distribution().name("uniform").low(0))

# test_variable = Distribution()
# test_variable.name("test_param_name")
# print(test_variable)

################################################################################
class Constant(Distribution):
    def __init__(self, constant: float):
        super().__init__("constant")
        self.constant = constant

class Uniform(Distribution):
    def __init__(self, low: float, high: float):
        super().__init__("uniform")
        self.low = low
        self.high = high

################################################################################
class Ensemble:

    def __init__(
            self, 
            size: int = 1,
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
    

    def get_all_params_set(self) -> list[dict[float]]:
        all_params_set = []
        for i in range(self.size):
            param_set = {}
            for parameter_name in self.parameter_names:
                param_set[parameter_name] = self.parameters[parameter_name][i]
            all_params_set.append(param_set)
        return all_params_set
