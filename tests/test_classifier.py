import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.Classifier as test
import unittest


class TestClassifier(unittest.TestCase):

    def test_init_correct_usage(self):
        None

    def test_init_wrong_module_type(self):
        None

    def test_init_wrong_classes_type(self):
        None

    def test_init_wrong_layer_config_type(self):
        None

    def test_init_wrong_loss_function_type(self):
        None

    def test_init_wrong_optimizer_type(self):
        None

    def test_init_verbosity_2(self):
        None

    def test_init_wrong_verbosity_type(self):
        None

    def test_init_wrong_enable_cuda_type(self):
        None

    def test_init_wrong_cuda_device_type(self):
        None

    def test_init_wrong_cuda_device_number(self):
        None

    def test_train_correct_usage(self):
        None

    def test_train_no_dataset(self):
        None

    def test_train_wrong_dataset_type(self):
        None

    def test_train_wrong_batch_size_type(self):
        None

    def test_train_wrong_epochs_type(self):
        None

    def test_train_negative_epochs(self):
        None

    def test_train_wrong_num_workers_type(self):
        None

    def test_train_negative_num_workers(self):
        None

    def test_train_pin_memory_wrong_type(self):
        None

    def test_evaluate_wrong_dataset_type(self):
        None

    def test_evaluate_wrong_batch_size_type(self):
        None

    def test_evaluate_wrong_num_workers_type(self):
        None

    def test_evaluate_def_negative_num_workers(self):
        None

    def test_evaluate_pin_memory_wrong_type(self):
        None

    def test_save_correct_path(self):
        None

    def test_save_invalid_path(self):
        None

    def test_set_verbosity_2(self):
        None

    def test_set_verbosity_wrong_type(self):
        None

    def test_set_device_enable_cuda_wrong_type(self):
        None

    def test_set_trainable_layers_layers_wrong_type(self):
        None

    def test_set_trainable_layers_correct_usage(self):
        None

    def test_from_file_invalid_path(self):
        None

    def test_from_file_wrong_file_format(self):
        None

    def test_from_file_correct_usage(self):
        None


if __name__ == '__main__':
    unittest.main()
