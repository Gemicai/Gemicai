import pydicom
import gemicai as gem

ds = gem.load_dicom('examples/dicom/DX/dx1.dcm.gz')
assert isinstance(ds, pydicom.Dataset)
ai = gem.Gemicai()
print(ai.classify(ds))
