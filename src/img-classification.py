'''
Created on Sep 6, 2019

@author: mark
'''
# import all required Python packages:
import skimage.io as io
import numpy as np
import os, shutil
   
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib

pn=os.path.abspath(__file__)
pn=pn.split("src")[0]
path=os.path.join(pn,'data')

# set up your directories with the MODIS data
rootdir = os.path.join(path,'raster')
# path to your training data
path_pix = os.path.join(path,'samples')
# path to your model
path_model = os.path.join(path,'models')
# path to your classification results
path_class = os.path.join(path,'class')

 # declare a new function
def training():
    # path to your MODIS TIFF  
    raster = rootdir + "modis_stack_ndvi.tif"
    # path to your corresponding pixel samples (training data) 
    samples = path_pix + "samples_modis.tif"  

    # read in MODIS raster
    img_ds = io.imread(raster)
    # convert to 16bit numpy array 
    img = np.array(img_ds, dtype='int16')

    # do the same with your sample pixels 
    roi_ds = io.imread(samples)   
    roi = np.array(roi_ds, dtype='int8')  
    
    # read in your labels
    labels = np.unique(roi[roi > 0]) 
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

    # compose your X,Y data (dataset - training data)     
    X = img[roi > 0, :] 
    Y = roi[roi > 0]     

    # assign class weights (class 1 has the weight 3, etc.)
    weights = {1:3, 2:2, 3:2, 4:2}

    # build your Random Forest Classifier 
    # for more information: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    rf = RandomForestClassifier(class_weight = weights, n_estimators = 100, criterion = 'gini', max_depth = 4, 
                                min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', 
                                bootstrap = True, oob_score = True, n_jobs = 1, random_state = None, verbose = True)  

    # alternatively you may try out a Gradient Boosting Classifier 
    # It is much less RAM consuming and considers weak training data      
    """ 
    rf = GradientBoostingClassifier(n_estimators = 300, min_samples_leaf = 1, min_samples_split = 4, max_depth = 4,    
                                    max_features = 'auto', learning_rate = 0.8, subsample = 1, random_state = None,         
                                    warm_start = True)
    """

    # now fit your training data with the original dataset
    rf = rf.fit(X,Y)

    # export your Random Forest / Gradient Boosting Model     
    model = path_model + "model.pkl"
    joblib.dump(rf, model)

#And now write your classification script:
def classification():
    # Read worldfile of original dataset
    tfw_old = str(raster.split(".tif")[0]) + ".tfw"     

    # Read Data    
    img_ds = io.imread(raster)   
    img = np.array(img_ds, dtype='int16')    

    # call your random forest model
    rf = path_model + "model.pkl"          
    clf = joblib.load(rf)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction = clf.predict(img_as_array) 
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)  

    # now export your classificaiton
    classification = path_class  + "classification.tif" 
    io.imsave(classification, class_prediction)    

    # Assign Worldfile to classified image    
    tfw_new = classification.split(".tif")[0] + ".tfw"   
    shutil.copy(tfw_old, tfw_new)
    
training()
classification()