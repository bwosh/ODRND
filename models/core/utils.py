def get_single_element(x:list):
    if len(x)!=1:
        raise Exception("Validation of exactly one element failed.")
    return x[0]