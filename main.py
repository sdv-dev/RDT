from transformer.hyper_transformer import *

if __name__ == "__main__":
    meta_file = 'data/Airbnb_demo_meta.json'
    ht = HyperTransformer(meta_file)
    tl = ['DTTransformer', 'NumberTransformer', 'CatTransformer']
    transformed = ht.hyper_fit_transform(transformer_list=tl)
    print(transformed)
    res = ht.hyper_reverse_transform(tables=transformed)
    print(res)

# from transformer.transformers.CatTransformer import *
# from transformer.utils import *
# col, col_meta = get_col_info('users', 'gender', 'demo/Airbnb_demo_meta.json')
# transformer = CatTransformer()
# transformed_data = transformer.fit_transform(col, col_meta)
# reversed = transformer.reverse_transform(transformed_data['gender'], col_meta)