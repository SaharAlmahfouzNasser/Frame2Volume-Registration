# Frame2Volume-Registration
Fusing intra-operative 2D transrectal ultrasound (TRUS) images with pre-operative 3D magnetic resonance
(MR) volume to guide prostate biopsy can significantly improve the outcome. However, such a multi-
modal 2D/3D registration problem is very challenging due to several significant hurdles such as difference
in dimensions/resolution, large modal appearance difference, large deformations, and heavy computational
load.
In general, registering a 2D frame to a 3D volume requires information from the tracking system that
can be either electromagnetic or optical tracking system. The electromagnetic tracking is responsive to any
metal tool which may exist in the field. However, the issue with optical tracker lies in the need for a clear
field of view between the sensors and the markers.
FVR network predicts six parameters, three parameters for shifting in x,y, and z directions and the rest for rotation in all three directions. To solve the
problem of dimensions mismatch, first, they extract feature maps from the 2D image and the 3D volume.
Then, they concatenate these maps and give them as input to the encoder to predict the six degrees of
freedom of the rigid registration. See figure 1.
![FVR](https://user-images.githubusercontent.com/52508554/199548140-94fd122e-9a76-4674-8042-5f0112f841c3.png)

# References:
Hengtao Guo et al. “End-to-end ultrasound frame to volume registration”. In: International Conference
on Medical Image Computing and Computer-Assisted Intervention. Springer. 2021, pp. 56–65.
