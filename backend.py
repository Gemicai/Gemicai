import pydicom
import gemicai as gem

dx = gem.load_dicom('examples/dicom/DX/526135013287415993965300423842921.dcm.gz')
assert isinstance(dx, pydicom.Dataset)
ai = gem.GemicaiZGT('C:\\Users\\niekh\\Desktop\\trees\\trees')
print(ai.classify(dx))
