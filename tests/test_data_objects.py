import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.data_objects as test
import unittest


wrong_dicom_file_path = os.path.join(parent_dir, "325261597578315993471860132776680.dcm.gz")
correct_dicom_file_path = os.path.join(parent_dir, "examples", "dicom", "CT",
                                       "325261597578315993471860132776680.dcm.gz")
gemset_path = os.path.join(parent_dir, "examples", "gzip", "CT", "000001.gemset")


class TestDicomObject(unittest.TestCase):

    def test_from_file_correct_usage(self):
        self.assertIsInstance(test.DicomObject.from_file(correct_dicom_file_path, ['Modality']), test.DicomObject)

    def test_from_file_no_file(self):
        with self.assertRaises(FileNotFoundError):
            test.DicomObject.from_file(wrong_dicom_file_path, ['Modality'])

    def test_from_file_wrong_file_format(self):
        with self.assertRaises(TypeError):
            test.DicomObject.from_file(gemset_path, ['Modality'])

    def test_from_file_wrong_label_type(self):
        with self.assertRaises(TypeError):
            test.DicomObject.from_file(correct_dicom_file_path, {'Modality'})

    def test_get_value_of_existing_field(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        self.assertNotEqual(obj.get_value_of('Modality'), None)

    def test_get_value_of_non_existing_field(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        self.assertEqual(obj.get_value_of('@31'), None)

    def test_get_value_of_wrong_value_type(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        with self.assertRaises(TypeError):
            obj.get_value_of(list())

    def test_meets_constraints_meet_criteria(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        value = obj.get_value_of('Modality')
        self.assertEqual(obj.meets_constraints({'Modality': value}), True)

    def test_meets_constraints_does_not_meet_criteria(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        self.assertEqual(obj.meets_constraints({'Modality': '@31'}), False)

    def test_meets_constraints_wrong_param_type(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        with self.assertRaises(TypeError):
            obj.meets_constraints(('Modality', '@31'))


if __name__ == '__main__':
    unittest.main()
