from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

price_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
            ),
        ),
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model_config.categorical_vars_with_na_frequent,
            ),
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        # == TEMPORAL VARIABLES ====
        (
            "elapsed_time",
            pp.TemporalVariableTransformer(
                variables=config.model_config.temporal_vars,
                reference_variable=config.model_config.ref_var,
            ),
        
        # ==== VARIABLE TRANSFORMATION =====
        ("log", LogTransformer(variables=config.model_config.numericals_log_vars)),
        (
            "binarizer",
            SklearnTransformerWrapper(
                transformer=Binarizer(threshold=0),
                variables=config.model_config.binarize_vars,
            ),
        ),
        
        # == CATEGORICAL ENCODING
       
        ),
        # encode categorical variables using the target mean
        (
            "categorical_encoder",
            OneHotEncoder(
                encoding_method="binary",
                variables=config.model_config.categorical_vars,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "Logistic",
            LogisticRegression(
                alpha=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
