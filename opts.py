import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # DATASET
    parser.add_argument("--train_ds_name", type=str, default="VAL2017")
    parser.add_argument("--train_ds_path", type=str, default="./cache/instances_val2017.json")

    parser.add_argument("--val_ds_name", type=str, default="VAL2017")
    parser.add_argument("--val_ds_path", type=str, default="./cache/instances_val2017.json")

    parser.add_argument("--supercategories", type=str, default="person,vehicle,animal")


    # Test flags
    parser.add_argument("--print_model_summary", action="store_true")
    parser.add_argument("--run_check_flops", action="store_true")
    parser.add_argument("--run_test_code", action="store_true")
    parser.add_argument("--check_preds", action="store_true")
    parser.add_argument("--zero_mask_loss", action="store_true")

    # Training
    parser.add_argument("--load_model", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default='./assets/model.h5')
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=0)

    args = parser.parse_args()
    return args