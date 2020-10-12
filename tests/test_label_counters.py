import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai as gem
import unittest

class TestLabelCounter(unittest.TestCase):

    def test__str__(self):
        ctr = gem.LabelCounter()

    def test_update_wrong_param_type(self):
        None

    def test_update_correct_param(self):
        None

    def test_update_nexted_correct_param(self):
        None

    def test_update_on_dimcom(self):
        None

    def test_update_on_gemicai_dicom_object(self):
        None


if __name__ == '__main__':
    unittest.main()
