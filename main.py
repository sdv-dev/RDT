from rdt.hyper_transformer import HyperTransformer

if __name__ == "__main__":
    meta_file = 'demo/Airbnb_demo_meta.json'
    ht = HyperTransformer(meta_file)
    tl = ['NumberTransformer', 'DTTransformer', 'CatTransformer']
    transformed = ht.hyper_fit_transform(transformer_list=tl)
    print(transformed)
    res = ht.hyper_reverse_transform(tables=transformed)
    print(res)

# from dataprep.transformers.DTTransformer import *
# from dataprep.utils import *
# col, col_meta = get_col_info('users', 'timestamp_first_active',
#                              'demo/Airbnb_demo_meta.json')
# transformer = DTTransformer()
# print(col)
# transformed_data = transformer.fit_transform(col, col_meta)
# print(type(transformed_data['timestamp_first_active']))
# reversed = transformer.reverse_transform(transformed_data['timestamp_first_active'], col_meta)
# print(reversed)
