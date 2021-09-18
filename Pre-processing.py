import numpy as np
import matplotlib.pyplot as plt
import cv2




import pydicom
filename= 'C:\\Users\\aksha\\Desktop\\1-4.dcm'
#filename= 'C:\\Users\\aksha\\Desktop\manifest-1617905855234\\Breast-Cancer-Screening-DBT\DBT-P00023\\01-01-2000-DBT-S04378-MAMMO SCREENING DIGITAL BILATERAL-20650\\19710.000000-NA-51654\\1-1.dcm'
ds = pydicom.dcmread(filename)

print()
print(f"File path........: {filename}")
print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print(f"Patient's Name...: {display_name}")
print(f"Patient ID.......: {ds.PatientID}")
print(f"Modality.........: {ds.Modality}")
print(f"Study Date.......: {ds.StudyDate}")
print(f"Image size.......: {ds.Rows} x {ds.Columns}")
#print(f"Pixel Spacing....: {ds.PixelSpacing}")

# use .get() if not sure the item exists, and want a default value if missing
#print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

# plot the image using matplotlib
array= ds.pixel_array
print(array.shape)
plt.imshow(array[0], cmap=plt.cm.gray)

plt.show()

