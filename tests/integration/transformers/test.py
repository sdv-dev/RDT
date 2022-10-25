
import pandas as pd

from rdt.hyper_transformer import HyperTransformer
from rdt.transformers.base import BaseTransformer
from rdt.transformers.categorical import FrequencyEncoder
from rdt.transformers.numerical import FloatFormatter

data = pd.DataFrame({
    'integer': [1, 2, 1],
    'categorical': ['a', 'b', 'a']
})


class DoublingTransformer(BaseTransformer):
    INPUT_SDTYPE = 'numerical'

    def _fit(self, data):
        self.output_properties = {
            None: {'sdtype': 'float', 'next_transformer': FloatFormatter()},
            'is_null': {'sdtype': 'float', 'next_transformer': FloatFormatter()}
        }

    def _transform(self, data):
        return data * 2

    def _reverse_transform(self, data):
        return data / 2


class DoublingTransformer2(BaseTransformer):
    INPUT_SDTYPE = 'categorical'

    def _fit(self, data):
        self.output_properties = {
            None: {'sdtype': 'categorical', 'next_transformer': FrequencyEncoder()}
        }

    def _transform(self, data):
        return data * 2

    def _reverse_transform(self, data):
        return pd.Series([i[0] for i in data])


# Run
ht = HyperTransformer()
ht.set_config({
    'sdtypes': {'integer': 'numerical', 'categorical': 'categorical'},
    'transformers': {'integer': DoublingTransformer(), 'categorical': DoublingTransformer2()}
})
ht.fit(data)
transformed = ht.transform(data)
print(transformed)
reverse_transformed = ht.reverse_transform(transformed)
print(reverse_transformed)
