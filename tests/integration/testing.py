import re

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer, get_demo
from rdt.errors import ConfigNotSetError, InvalidConfigError, InvalidDataError, NotFittedError
from rdt.transformers import (
    AnonymizedFaker, BaseTransformer, BinaryEncoder, ClusterBasedNormalizer, FloatFormatter,
    FrequencyEncoder, LabelEncoder, OneHotEncoder, RegexGenerator, UnixTimestampEncoder,
    get_default_transformer, get_default_transformers)

data = pd.DataFrame({
    'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
    'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
})

# Run
ht = HyperTransformer()
ht.detect_initial_config(data)
ht.update_sdtypes({
    'categorical': 'pii',
    'names': 'pii'
})
ht.update_transformers({
    'names': AnonymizedFaker(),
    'categorical': AnonymizedFaker()
})
ht.fit(data)
print(ht.field_transformers['names'].random_states['fit'])
transformed = ht.transform(data)
print(ht.reverse_transform(transformed))

print(ht.field_transformers['names'].random_states['fit'])
transformed = ht.transform(data)
print(ht.reverse_transform(transformed))

ht.reset_randomization()
print(ht.field_transformers['names'].random_states['fit'])
transformed = ht.transform(data)
print(ht.reverse_transform(transformed))

