
from rdt.transformers.base import BaseTransformer
import pandas as pd
import numpy as np

from rdt.transformers.text import RegexGenerator
data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'username': ['a', 'b', 'c', 'd', 'e']
})

instance = RegexGenerator()
transformed = instance.fit_transform(data, 'id')
reverse_transform = instance.reverse_transform(transformed)