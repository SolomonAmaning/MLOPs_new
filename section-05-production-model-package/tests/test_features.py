from classification_model.config.core import config
from classification_model.processing.features import TemporalVariableTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    transformer = TemporalVariableTransformer(
        variables=config.model_config.temporal_vars,
        reference_variable=config.model_config.ref_var,
)