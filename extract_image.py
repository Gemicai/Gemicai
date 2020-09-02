import gzip
import pydicom as dicom
import os
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import sys
import string

image_formats = ["png", "jpg"]

# if needed we can export it later
fields_of_interest = ['Rows', 'StudyDate', 'SeriesTime', 'ContentTime', 'StudyInstanceUID', 'SeriesInstanceUID','SOPInstanceUID',
'Modality', 'SeriesDate', 'AccessionNumber', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription', 'InstanceNumber', 
 'PatientOrientation', 'ImageLaterality', 'ImageComments', 'SeriesNumber', 'PatientName']

def load_dicom(filename):
   if (filename.endswith('.dcm')):
      ds = dicom.dcmread(filename)
   else:
      with gzip.open(filename) as fd:
         ds = dicom.dcmread(fd, force=True)
   ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
   return ds

def print_help():
   print("Proper program usage: ")
   print("extract_image.py {input_location} {output_location}.{png/jpg}")

# ------------------------ END OF THE DECLARATIONS ------------------------

args = []
for arg in sys.argv:
   args.append(str(arg))


# process command line params
try:	
   if len(args) != 3 or not (args[2].split('.')[1] in image_formats): 
      raise Exception("Improper program usage");  
except:
   print_help()


# load a dcm file
try:
   ds = load_dicom(os.path.normpath(args[1]))
except:
   print("Please priovide a valid .dcm or .dcm.gz file")


# try to fetch the relevant data (so maybe we can process it later)
dcm_data = []
for field in fields_of_interest:
   try:
      dcm_data.append(getattr(ds, field))
   except:
      dcm_data.append("NULL")

# a colormap and a normalization instance
cmap = plt.cm.gray
norm = plt.Normalize(vmin=ds.pixel_array.min(), vmax=ds.pixel_array.max())

# map the normalized data to colors
image = cmap(norm(ds.pixel_array))

# save the image
plt.imsave(os.path.normpath(args[2]), image, cmap='gray')
   
#if we have to handle dcm_data we can od it using this code I have found
"""

# create data frame to keep records of the images
df = pd.DataFrame(data=map_list, columns = ['filename','StudyTime', 
                                            'SeriesTime', 
                                            'ContentTime', 
                                            'StudyInstanceUID', 
                                            'SeriesInstanceUID',
                                            'SOPInstanceUID',
                                            'Modality', 
                                            'SeriesDate', 
                                            'AccessionNumber', 
                                            'BodyPartExamined', 
                                            'StudyDescription', 
                                            'SeriesDescription', 
                                            'InstanceNumber', 
                                            'PatientOrientation', 
                                            'ImageLaterality', 
                                            'ImageComments', 
                                            'SeriesNumber', 
                                            'PatientName'])

# save dataframe for later usage
df.to_pickle('30Day_Mortality_hipfracture_dicom_info.pkl')

print (datetime.now() - startTime )

"""



