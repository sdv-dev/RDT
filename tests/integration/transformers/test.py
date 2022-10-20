
from rdt.transformers.base import BaseTransformer
import pandas as pd
import numpy as np

class DummyTransformer(BaseTransformer):

    INPUT_SDTYPE = 'boolean'

    def __init__(self):
        self.output_properties = {
            None: {'sdtype': 'float'},
            'null': {'sdtype': 'float'},
        }

    def _fit(self, data):
        pass

    def _transform(self, data):
        out = pd.DataFrame(dict(zip(
            self.output_columns,
            [
                data.astype(float).fillna(-1),
                data.isna().astype(float)
            ]
        )))

        return out

    def _reverse_transform(self, data):
        output = data[self.output_columns[0]]
        output = output.round().astype(bool).astype(object)
        output.iloc[data[self.output_columns[1]] == 1] = np.nan

        return output

data = pd.DataFrame({'bool': [True, False, True, np.nan]})
transformer = DummyTransformer()

# Run
transformed = transformer.fit_transform(data, 'bool')
print(transformed)
reverse = transformer.reverse_transform(transformed)
