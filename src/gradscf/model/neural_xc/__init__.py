"""Neural XC models and their model-owned training/workflows, loaded lazily."""
from importlib import import_module

_EXPORTS = {'Functional': ('.api', 'Functional'),
 'make_functional': ('.api', 'make_functional'),
 'ChannelSpec': ('.config', 'ChannelSpec'),
 'ComponentSpec': ('.config', 'ComponentSpec'),
 'Config': ('.config', 'Config'),
 'NetworkSpec': ('.config', 'NetworkSpec'),
 'COMMON_SEMILOCAL_COMPONENT_SPECS': ('.components', 'COMMON_SEMILOCAL_COMPONENT_SPECS'),
 'FRIENDLY_XC_COMPONENT_ALIASES': ('.components', 'FRIENDLY_XC_COMPONENT_ALIASES'),
 'JAXXCFunctionalInfo': ('.components', 'JAXXCFunctionalInfo'),
 'JAXXCStatus': ('.components', 'JAXXCStatus'),
 'SemilocalEnergyDensityFn': ('.components', 'SemilocalEnergyDensityFn'),
 'SemilocalEnergyDensityModule': ('.components', 'SemilocalEnergyDensityModule'),
 'SemilocalLocalContributionFn': ('.components', 'SemilocalLocalContributionFn'),
 'available_semilocal_component_infos': ('.components', 'available_semilocal_component_infos'),
 'available_semilocal_components': ('.components', 'available_semilocal_components'),
 'make_custom_semilocal_module': ('.components', 'make_custom_semilocal_module'),
 'make_libxc_semilocal_module': ('.components', 'make_libxc_semilocal_module'),
 'resolve_xc_component_name': ('.components', 'resolve_xc_component_name'),
 'semilocal_component_info': ('.components', 'semilocal_component_info'),
 'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE': ('.defaults',
                                              'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE'),
 'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES': ('.defaults',
                                                'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES'),
 'DEFAULT_NEURAL_XC_HF_CHANNEL_MODE': ('.defaults', 'DEFAULT_NEURAL_XC_HF_CHANNEL_MODE'),
 'DEFAULT_NEURAL_XC_HF_INPUT_MODE': ('.defaults', 'DEFAULT_NEURAL_XC_HF_INPUT_MODE'),
 'DEFAULT_NEURAL_XC_RESPONSE_HF_MODE': ('.defaults', 'DEFAULT_NEURAL_XC_RESPONSE_HF_MODE'),
 'DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE': ('.defaults', 'DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE'),
 'DEFAULT_NEURAL_XC_SEMILOCAL_XC': ('.defaults', 'DEFAULT_NEURAL_XC_SEMILOCAL_XC'),
 'DEFAULT_NETWORK_ARCHITECTURE': ('.defaults', 'DEFAULT_NETWORK_ARCHITECTURE'),
 'DEFAULT_NETWORK_HIDDEN_DIMS': ('.defaults', 'DEFAULT_NETWORK_HIDDEN_DIMS'),
 'DEFAULT_INPUT_FEATURE_MODE': ('.defaults', 'DEFAULT_INPUT_FEATURE_MODE'),
 'resolve_coefficient_prior_values': ('.defaults', 'resolve_coefficient_prior_values'),
 'ResidualMixingMLP': ('.networks', 'ResidualMixingMLP'),
 'presets': ('.presets', None),
 'BoundNeuralXCFunctional': ('.factory', 'BoundNeuralXCFunctional'),
 'NeuralXCHybridFunctional': ('.factory', 'NeuralXCHybridFunctional'),
 'NeuralXCMixingMLP': ('.factory', 'NeuralXCMixingMLP'),
 'make_neural_xc_functional': ('.factory', 'make_neural_xc_functional')}
__all__ = ['BoundNeuralXCFunctional',
 'ChannelSpec',
 'COMMON_SEMILOCAL_COMPONENT_SPECS',
 'ComponentSpec',
 'Config',
 'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE',
 'DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES',
 'DEFAULT_NEURAL_XC_HF_CHANNEL_MODE',
 'DEFAULT_NEURAL_XC_HF_INPUT_MODE',
 'DEFAULT_NEURAL_XC_RESPONSE_HF_MODE',
 'DEFAULT_NEURAL_XC_RESPONSE_PT2_MODE',
 'DEFAULT_NEURAL_XC_SEMILOCAL_XC',
 'DEFAULT_NETWORK_ARCHITECTURE',
 'Functional',
 'FRIENDLY_XC_COMPONENT_ALIASES',
 'JAXXCFunctionalInfo',
 'JAXXCStatus',
 'NetworkSpec',
 'DEFAULT_INPUT_FEATURE_MODE',
 'DEFAULT_NETWORK_HIDDEN_DIMS',
 'ResidualMixingMLP',
 'NeuralXCHybridFunctional',
 'NeuralXCMixingMLP',
 'SemilocalEnergyDensityFn',
 'SemilocalEnergyDensityModule',
 'SemilocalLocalContributionFn',
 'available_semilocal_component_infos',
 'available_semilocal_components',
 'make_custom_semilocal_module',
 'make_functional',
 'make_libxc_semilocal_module',
 'make_neural_xc_functional',
 'presets',
 'resolve_coefficient_prior_values',
 'resolve_xc_component_name',
 'semilocal_component_info',
 'training',
 'workflows']

def __getattr__(name):
    if name in ('training', 'workflows'):
        value = import_module('.'+name, __name__)
    elif name in _EXPORTS:
        module, member = _EXPORTS[name]
        value = import_module(module, __name__)
        if member is not None:
            value = getattr(value, member)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
