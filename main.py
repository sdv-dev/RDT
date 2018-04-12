from transformer.hyper_transformer import *

if __name__ == "__main__":
    meta_file = 'data/Airbnb_demo_meta.json'
    ht = HyperTransformer(meta_file)
    tl = ['DTTransformer', 'NumberTransformer']
    transformed = ht.hyper_fit_transform(transformer_list=tl)
    print(transformed)
    res = ht.hyper_reverse_transform(tables=transformed)
    print(res)