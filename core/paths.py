import os

# make paths always relative to this location of paths.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE_UP = os.path.dirname(PROJECT_ROOT)

# mini dataset
mini_dataset = perfect_teeth_dataset_dir = os.path.join(PROJECT_ROOT, "minimal_test_data/perfect_teeth_dataset")

# training and validation paths
main_dataset_dir = os.path.join(ONE_UP,"data/dataset_2_n300")
train_dir = os.path.join(main_dataset_dir, "train")
val_dir = os.path.join(main_dataset_dir, "val")

# test paths
test_dir = os.path.join(main_dataset_dir, "test")

base_output_path = os.path.join(ONE_UP, "outputs/output")

def get_model_path(run):
    model_path = os.path.join(base_output_path, f"unet_braces_d2_{run}.pth")
    return model_path


best_model = 'unet_braces_d2_30-1-ep104.pth'

def get_best_models_path():
    model_path = os.path.join(PROJECT_ROOT, 'trained_models', best_model)
    if os.path.exists(model_path):
        return model_path
    model_path = os.path.join(PROJECT_ROOT, best_model)
    if os.path.exists(model_path):
        return model_path
    raise FileNotFoundError(f"The Model {best_model} was not found at folder 'trained_models' or at Project Root.")