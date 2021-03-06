import os
import gemicai.data_objects as test
import unittest


wrong_dicom_file_path = os.path.join("..", "325261597578315993471860132776680.dcm.gz")
correct_dicom_file_path = os.path.join("..", "examples", "dicom", "CT",
                                       "325261597578315993471860132776680.dcm.gz")
gemset_path = os.path.join("..", "examples", "gemset", "CT", "000001.gemset")


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

    def test_from_file_wrong_tensor_size_type(self):
        with self.assertRaises(TypeError):
            test.DicomObject.from_file(correct_dicom_file_path, ['Modality'], tensor_size="z")

    def test_from_file_different_tensor_sizes(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'], tensor_size=None)
        self.assertIsInstance(obj, test.DicomObject)
        self.assertEqual(obj.tensor.shape, test.torch.Size([512, 512]))
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'], tensor_size=(244, 200))
        self.assertIsInstance(obj, test.DicomObject)
        self.assertEqual(obj.tensor.shape, test.torch.Size([244, 200]))

    def test_get_value_of_existing_field(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        self.assertNotEqual(obj.get('Modality'), None)

    def test_get_value_of_non_existing_field(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        self.assertEqual(obj.get('@31'), None)

    def test_get_value_of_wrong_value_type(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        with self.assertRaises(TypeError):
            obj.get(list())

    def test_meets_constraints_meet_criteria(self):
        obj = test.DicomObject.from_file(correct_dicom_file_path, ['Modality'])
        value = obj.get('Modality')
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
