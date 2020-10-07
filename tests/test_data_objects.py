import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.data_objects as test
import unittest


wrong_dicom_file_path = os.path.join(parent_dir, "325261597578315993471860132776680.dcm.gz")
correct_dicom_file_path = os.path.join(parent_dir, "examples", "dicom", "CT",
                                       "325261597578315993471860132776680.dcm.gz")


class TestDicomObject(unittest.TestCase):

    def test_from_file_correct_file(self):
        None

    def test_from_file_no_file(self):
        None

    def test_from_file_wrong_file_format(self):
        None

    def test_from_file_wrong_label_type(self):
        None

    def test_from_file_correct_usage(self):
        None

    def test_get_value_of_existing_field(self):
        None

    def test_get_value_of_non_existing_field(self):
        None

    def test_meets_constraints_correct_usage(self):
        None

    def test_meets_constraints_wrong_param_type(self):
        None


if __name__ == '__main__':
    unittest.main()
