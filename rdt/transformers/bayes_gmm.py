"""Bayesian gaussian mixture model transformer."""
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

from rdt.transformers.categorical import OneHotEncodingTransformer

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type',
        'transform', 'transform_aux',
        'output_info', 'output_dimensions'
    ]
)


class BayesGMMCopy(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    INPUT_TYPE = 'numerical'
    DETERMINISTIC_TRANSFORM = False
    DETERMINISTIC_REVERSE = False
    COMPOSITION_IS_IDENTITY = False

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, column_name, raw_column_data):
        """Train Bayesian GMM for continuous column."""
        gm = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )

        gm.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = gm.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        ohe = OneHotEncodingTransformer()
        fit_data = pd.DataFrame(raw_column_data, columns=[column_name])

        ohe.fit(fit_data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def _fit(self, data, discrete_columns=()):
        """Fit GMM for continuous columns and One hot encoder for discrete columns.

        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(data.shape[1])]
            data = pd.DataFrame(data, columns=column_names)

        self._column_raw_dtypes = data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in data.columns:
            raw_column_data = data[column_name].to_numpy()
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(column_name, data[column_name])
            else:
                column_transform_info = self._fit_continuous(column_name, raw_column_data)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, raw_column_data):
        gm = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = gm.means_.reshape((1, self._max_clusters))
        stds = np.sqrt(gm.covariances_).reshape((1, self._max_clusters))
        normalized_values = ((raw_column_data - means) / (4 * stds))[:, valid_component_indicator]
        component_probs = gm.predict_proba(raw_column_data)[:, valid_component_indicator]

        selected_component = np.zeros(len(raw_column_data), dtype='int')
        for i in range(len(raw_column_data)):
            component_porb_t = component_probs[i] + 1e-6
            component_porb_t = component_porb_t / component_porb_t.sum()
            selected_component[i] = np.random.choice(np.arange(num_components), p=component_porb_t)

        aranged = np.arange(len(raw_column_data))
        selected_normalized_value = normalized_values[aranged, selected_component].reshape([-1, 1])
        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[np.arange(len(raw_column_data)), selected_component] = 1
        return [selected_normalized_value, selected_component_onehot]

    def _transform_discrete(self, column_transform_info, raw_column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(raw_column_data, columns=[column_transform_info.column_name])
        return [ohe.transform(data).to_numpy()]

    def _transform(self, data):
        """Take raw data and output a matrix data."""
        if not isinstance(data, pd.DataFrame):
            column_names = [str(num) for num in range(data.shape[1])]
            data = pd.DataFrame(data, columns=column_names)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = data[[column_transform_info.column_name]].to_numpy()
            if column_transform_info.column_type == 'continuous':
                column_data_list += self._transform_continuous(column_transform_info, column_data)
            else:
                assert column_transform_info.column_type == 'discrete'
                column_data_list += self._transform_discrete(column_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _reverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = gm.means_.reshape([-1])
        stds = np.sqrt(gm.covariances_).reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _reverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_types()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def _reverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._reverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)

            else:
                assert column_transform_info.column_type == 'discrete'
                recovered_column_data = self._reverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Convert."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
