
### File Description
+ **dataprep.py**:  
assistant file, which helps to choose 400 images from every class in training data, and 100 images from every class in testing data
+ **CI1-DL-Food11.ipynb**(deprecated):  
notebook based on colab, so far contains the procedure of training by inception_v3(existed in tf) using our own dataset, testing one image we provide, and save the model we built in local computer. The accuracy of cross-validated training data is 85.4%.
+ **DL_Food_11.ipynb**(main file)：
completed procedure from data acquisition(from google drive) to model saving/loading, but still exists serveral bugs(eg. Sam's model can be saved but new vgg cannot) 

### Access to Dataset
+ images preview  
https://drive.google.com/open?id=1IBktmsI_-nWHq2SXigzHDrwnrlx-kpl5
+ provided for downloading directly from google drive  
https://drive.google.com/open?id=1f6_eDFiKv93XQbOIlnv7fAPBITtrEHZC

### to do
+ fix bugs of loading model
+ tune parameters
