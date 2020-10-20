import pydicom
import gemicai as gem

dx = gem.load_dicom('examples/dicom/DX/dx1.dcm.gz')
mg = gem.load_dicom('/mnt/SharedStor/tutorials/Mammography/325262712664941599250632305555836.dcm.gz')
assert isinstance(dx, pydicom.Dataset)
ai = gem.Gemicai('/mnt/SharedStor/trees')
print(ai.classify(dx))
print(ai.classify(mg))
