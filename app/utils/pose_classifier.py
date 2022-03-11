import numpy as np
from joblib import load

class PoseClassifier(object):
  """
    Returns the class(exercise) and sub-class(start/end) from the frame features

    Arguments:
      features : image pixel array
  """
  def __init__(self):
    # load models
    self.clf_clas = load('../resources/models/cascade/xgb_clf.model')
    self.clf_subclas = load('../resources/models/cascade/knn_clf.model')
    self.clas_encoder = load('../resources/models/cascade/clas_encoder.model')
    self.subclas_encoder = load('../resources/models/cascade/subclas_encoder.model')
  
  def classify(self, features):
    """   returns the identified pose class, for given frame features   """
    pred_clas = self.clf_clas.predict(features.reshape(1, -1))
    clas_name = self.clas_encoder.inverse_transform(pred_clas)[0]
    features = np.hstack((features, pred_clas))
    pred_subclas = self.clf_subclas.predict(features.reshape(1, -1))
    subclas_name = self.subclas_encoder.inverse_transform(pred_subclas)[0]

    final_pred = 'random-random'
    if subclas_name.startswith(clas_name):
      # if both classifiers agree to a common exercise type then they are probably correct
      final_pred = subclas_name
    return final_pred
