* Codes created for 3d slicer console - 



**Step 4:** 4\_create\_hilumanchor.py (step 4

**Step 8:** 8\_Load\_sequence\_3DSlicer.py



* All other codes work outside 3dslicer in a python environment. 
* **Step 1** in the pipeline is done in 3DSlicer using nninteractive and manual correction (segmentation tools)
* **Step 3** is also done completely in 3Dslicer using the General registration(Elastix) module. Steps - 


&#x20;     Upload the segmentation of left lung and surface meshed model of collapsed

&#x20;     Convert both into 2 separate binary labelmaps (Rightclick onto the segmentation -> export visible segments to binary label map).

&#x20;     Using the Module "Cast Skalar Volume", convert the two binary label maps into two volumes.

&#x20;     Apply Image registrations using the "General Registration (Elastix)" Module. (Fixed Volume: inflated lung volume, Moving Volume: collapsed lung volume, with default parameters). This gives a transform (vector field).

&#x20;     Apply the vector field to the volumetrized surface model of the collapsed lung, to receive a surface model of the inflated lung with same corresponding nodes and number of faces on the surfaces.



* **Step 5** is included in the code of **step 6** 
* **Step 6** produces the results but to visualize it in a sequence format **step 7** and **step 8** are done

