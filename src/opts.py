import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # GENERAL
    parser.add_argument("--cache_path", type=str, default="../cache/")

    # DATASET
    parser.add_argument("--train_ds_name", type=str, default="TRAIN2017")
    parser.add_argument("--train_ds_path", type=str, default="../cache/instances_train2017.json")

    parser.add_argument("--val_ds_name", type=str, default="VAL2017")
    parser.add_argument("--val_ds_path", type=str, default="../cache/instances_val2017.json")

    parser.add_argument("--supercategories", type=str, default="person,vehicle,animal")

    # Model
    parser.add_argument("--model", type=str, default="ssdlitemn2")
    parser.add_argument("--model_width", type=float, default=0.0625) # set to minimal 
    

    # Test flags
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--pred", action="store_true")
    parser.add_argument("--model_flops_path", type=str, default='../cache/model_flops.h5')

    # Training
    parser.add_argument("--load_model", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default='../cache/model.h5')
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()
    return args