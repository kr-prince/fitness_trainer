import cv2
import numpy as np
import mediapipe as mp

np.random.seed(123)


class Point(object):
  """
    Class to denote a joint in 2-D space 

    Arguments:
      x, y : x-axis and y-axis co-ordinates
      visibility : visibility score of the joint
  """
  def __init__(self, x=0.0, y=0.0, visibility=0.0):
    self.x = x
    self.y = y
    self.visibility = visibility
  
  def __add__(self, other):
    return Point(self.x+other.x, self.y+other.y, (self.visibility+other.visibility)*0.5)
  
  def __sub__(self, other):
    return Point(self.x-other.x, self.y-other.y, (self.visibility+other.visibility)*0.5)
  
  def __mul__(self, num):
    if (type(num) is int) or (type(num) is float):
      return Point(self.x*num, self.y*num, self.visibility)
    raise AttributeError("Multiplication allowed only with scalars")
  
  def __repr__(self):
    return "x=%f  y=%f  visibility=%f" %(self.x, self.y, self.visibility)



class FrameProcessor(object):
  """
    Class containing utility methods for processing a frame and returning its 
    featurized form
  """
  def __init__(self):
    self.mp_pose = mp.solutions.pose
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.point_names = ['NOSE','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_HIP','RIGHT_HIP',
      'LEFT_ELBOW','RIGHT_ELBOW' ,'LEFT_WRIST','RIGHT_WRIST','LEFT_KNEE','RIGHT_KNEE',
      'LEFT_ANKLE','RIGHT_ANKLE','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX']
  
  def _get_frame_landmarks(self, frame):
    # returns the landmarks if pose is detected in the given image frame array, else None
    with self.mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.3,min_tracking_confidence=0.4) as pose:
      results = pose.process(frame)
    return results.pose_landmarks

  def _get_points_coordinates(self, pose_landmarks, frame_shape):
    # returns the co-ordinates of the points after translational normalization
    h, w = frame_shape[:2]
    lhip, rhip = (pose_landmarks.landmark[self.mp_pose.PoseLandmark['LEFT_HIP']],
                  pose_landmarks.landmark[self.mp_pose.PoseLandmark['RIGHT_HIP']])
    
    # find the mid-hip coordinates 
    mid_hips = Point((lhip.x+rhip.x)*0.5, (lhip.y+rhip.y)*0.5, (lhip.visibility+rhip.visibility)*0.5)
    points = list()
    for name in self.point_names:
      point = pose_landmarks.landmark[self.mp_pose.PoseLandmark[name]]
      points.append(Point((point.x-mid_hips.x)*w, (point.y-mid_hips.y)*h, point.visibility))
    
    return points
  
  def _get_distance(self, p1, p2):
    # returns euclidean dist between two points
    p = p1 - p2
    return np.sqrt(p.x**2 + p.y**2)

  def _get_angle(self, p1, p2, p3):
    # returns anti-clockwise angle made by point 2 with point 1 and 3
    ab, bc = p1-p2, p3-p2
    dot_prod = (ab.x*bc.x)+(ab.y*bc.y)
    mod_prod = np.sqrt((ab.x**2+ab.y**2)*(bc.x**2+bc.y**2))
    angle = np.rad2deg(np.arccos(dot_prod/mod_prod))   # in degrees
    det = ab.x*bc.y - ab.y*bc.x   # determinant for correct quadrant
    angle = 360-angle if det<0 else angle
    return angle

  def get_frame_features(self, frame):
    """   returns the featurized form of the given frame(image array)   """
    pose_landmarks = self._get_frame_landmarks(frame)
    features = None
    self.coordinates = None
    
    if pose_landmarks is not None:
      # get all the required body points
      (nose, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, 
      left_wrist, right_wrist, left_knee, right_knee, left_ankle, right_ankle, 
      left_foot_idx, right_foot_idx) = self._get_points_coordinates(pose_landmarks, frame.shape)
      
      assert (nose.visibility>0.4), "Face not in frame"
      assert (left_shoulder.visibility>0.4 and right_shoulder.visibility>0.4), "Shoulders not in frame"
      assert (left_hip.visibility>0.4 and right_hip.visibility>0.4), "Body not in frame"
      assert (left_foot_idx.visibility>0.4 and right_foot_idx.visibility>0.4), "Feet not in frame"
      
      # calculate the torso length which will be used to normalise the distances
      neck = (left_shoulder + right_shoulder)*0.5
      torso_len = (self._get_distance(neck, left_hip) + self._get_distance(neck, right_hip))*0.5
      
      # find body core - mid-point of line joining mid-shoulder and mid-hips
      mid_hips = (left_hip + right_hip)*0.5
      core = (neck + mid_hips)*0.5
      
      # calculate distance features
      dist_feats = np.array([
        # distance of limbs from body core
            self._get_distance(core, nose),
            # self._get_distance(core, left_shoulder), self._get_distance(core, right_shoulder),
            self._get_distance(core, left_elbow), self._get_distance(core, right_elbow),
            self._get_distance(core, left_wrist), self._get_distance(core, right_wrist),
            # self._get_distance(core, left_hip), self._get_distance(core, right_hip),
            self._get_distance(core, left_knee), self._get_distance(core, right_knee),
            self._get_distance(core, left_ankle), self._get_distance(core, right_ankle),
        
        # 2-joints distances
            self._get_distance(left_shoulder, left_wrist), self._get_distance(right_shoulder, right_wrist),
            self._get_distance(left_hip, left_elbow), self._get_distance(right_hip, right_elbow),
            self._get_distance(left_shoulder, left_knee), self._get_distance(right_shoulder, right_knee),
            self._get_distance(left_hip, left_ankle), self._get_distance(right_hip, right_ankle),
            self._get_distance(left_knee, left_foot_idx), self._get_distance(right_knee, right_foot_idx),
        
        # cross joint distances
            self._get_distance(left_wrist, right_wrist), self._get_distance(left_elbow, right_elbow),
            self._get_distance(left_shoulder, right_shoulder), self._get_distance(left_hip, right_hip),
            self._get_distance(left_knee, right_knee)  #, self._get_distance(left_ankle, right_ankle)
        ])
      # normalise dist features
      dist_feats /= torso_len
      
      # calculate angle features
      ground = Point(core.x, frame.shape[0]-1, 0.9)
      angle_feats = np.array([
        # angles made by neck with both elbows, angles made by hips with both knees,
        # spine angle, and body with respect to ground
          self._get_angle(left_elbow, neck, right_elbow), self._get_angle(left_knee, mid_hips, right_knee),
          self._get_angle(nose, neck, mid_hips), self._get_angle(nose, core, ground)
      ])
      angle_feats /= 360.0
      
      visibility_feats = np.array([
        # visibility features of left and right profiles(upper and lower body)
            (left_shoulder.visibility + left_hip.visibility)*0.5,
            (left_hip.visibility + left_knee.visibility)*0.5,
            (right_shoulder.visibility + right_hip.visibility)*0.5,
            (right_hip.visibility + right_knee.visibility)*0.5
        ])
      
      features = np.hstack((dist_feats, angle_feats, visibility_feats))

      # Save the coordinates for pose checking later
      self.coordinates = (nose, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow,
            left_wrist, right_wrist, left_knee, right_knee, left_ankle, right_ankle, left_foot_idx, 
            right_foot_idx, neck, torso_len, mid_hips, core, ground)
    return features

  def pose_corrector(self, pose_clas):
    """
      This function checks the Pose for various key-points specific to it and recommends
      changes to the user. It uses the stored co-ordinates of just previously featurized frame.
    """
    feedback = list()
    if self.coordinates is not None:
      (nose, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow,
          left_wrist, right_wrist, left_knee, right_knee, left_ankle, right_ankle, left_foot_idx, 
          right_foot_idx, neck, torso_len, mid_hips, core, ground) = self.coordinates

      if pose_clas.startswith('jumping_jacks'):
        feet_width = self._get_distance(left_ankle, right_ankle)
        shoulder_width = self._get_distance(left_shoulder, right_shoulder)
        body_angle = self._get_angle(neck, core, mid_hips)
        body_angle = min(body_angle, 360.0-body_angle)
        if pose_clas=='jumping_jacks-start':
          # feet should be around (2*shoulder width) apart, hands stretched straight
          left_hand_angle = self._get_angle(left_shoulder, left_elbow, left_wrist)
          left_hand_angle = min(left_hand_angle, 360-left_hand_angle)
          right_hand_angle = self._get_angle(right_shoulder, right_elbow, right_wrist)
          right_hand_angle = min(right_hand_angle, 360-right_hand_angle)
          
          if(feet_width <= shoulder_width*0.85):
            feedback.append("Your feet are too close when starting Jumping Jacks.")
          if(feet_width > 2*shoulder_width):
            feedback.append("Your feet are too wide when starting Jumping Jacks.")
          if((left_hand_angle < 120 and left_hand_angle > 180) or (right_hand_angle < 120 and right_hand_angle > 180)):
            feedback.append("Keep your arms straight while doing Jumping Jacks.")

        if pose_clas=='jumping_jacks-end':
          # Keep hands above head and legs (2-3 shoulder length) wide apart
          if not(nose.y > left_wrist.y and nose.y > right_wrist.y):
            feedback.append("Keep your arms above your head when ending Jumping Jacks.")
          if(feet_width <= 1.5*shoulder_width):
            feedback.append("Your feet are too close when ending Jumping Jacks.")
          if(feet_width > 2.75*shoulder_width):
            feedback.append("Your feet are too wide when ending Jumping Jacks.")

      elif pose_clas.startswith('crunches'):
        if pose_clas=='crunches-start':
          body_angle = self._get_angle(neck, core, mid_hips)
          body_angle = min(body_angle, 360.0-body_angle)
          # Keep your body relaxed on ground
          if body_angle > 190 or body_angle < 170:
            feedback.append("Lie down in a relaxed way while starting crunches.")
        if pose_clas=='crunches-end':
          body_angle = self._get_angle(nose, core, mid_hips)
          body_angle = min(body_angle, 360.0-body_angle)
          # Keep your head slightly raised from ground
          if body_angle < 120 or body_angle > 160:
            feedback.append("Raise your head slightly from neck while ending crunches.")

      elif pose_clas.startswith('lunges'):
        body_angle = self._get_angle(neck, core, mid_hips)
        body_angle = min(body_angle, 360.0-body_angle)
        # core straight 
        if body_angle > 190 or body_angle < 170:
          feedback.append("Keep your core straight while doing lunges.")
        if pose_clas=='lunges-start':
          pass
        if pose_clas=='lunges-end':
          # knees should not cross toes
          left_leg_angle = self._get_angle(left_ankle, left_knee, left_hip)
          left_leg_angle = min(left_leg_angle, 360.0-left_leg_angle)
          right_leg_angle = self._get_angle(right_ankle, right_knee, right_hip)
          right_leg_angle = min(right_leg_angle, 360.0-right_leg_angle)
          if ((left_leg_angle > 100 or left_leg_angle < 80) or (right_leg_angle > 100 or right_leg_angle < 80)):
            feedback.append("While doing lunges your knees should be at right angles.")
      
      elif pose_clas.startswith('planks'):
        # check if the whole body is in straight line - 
        # neck, core, mid-hips :: mid-hips, mid-knees, mid-ankle
        upper_body_angle = self._get_angle(neck, core, mid_hips)
        upper_body_angle = min(upper_body_angle, 360.0-upper_body_angle)
        mid_knees = (left_knee + right_knee)*0.5
        mid_ankle = (left_ankle + right_ankle)*0.5
        lower_body_angle = self._get_angle(mid_hips, mid_knees, mid_ankle)
        lower_body_angle = min(lower_body_angle, 360.0-lower_body_angle)
        if upper_body_angle < 165 and upper_body_angle > 190:
          feedback.append("Straighten your upper body while doing planks.")
        if lower_body_angle < 165 and lower_body_angle > 190:
          feedback.append("Straighten your lower body while doing planks.")

      elif pose_clas.startswith('squats'):
        if pose_clas=='squats-start':
          pass
        if pose_clas=='squats-end':
          # knees should not cross toes
          if (right_knee.x-right_hip.x >= 0):
            # facing right
            dist = right_foot_idx.x - right_knee.x
          else:
            dist = left_knee.x - left_foot_idx.x
          if dist <= 0:
            feedback.append("While doing squats your knees should not cross your toes.")
      
      else:
        pass

    return " ".join(feedback) if len(feedback) else None
  
  def lightcheck(self, frame):
    """   
      Function to check light conditions given an image frame
    """
    thres, bright_thres, dark_thres = 0.3, 225, 30
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = np.size(gray)
    dark_pixels = np.sum(gray <= dark_thres)
    bright_pixels = np.sum(gray >= bright_thres)
    
    light_status = None
    if dark_pixels/total_pixels > thres:
      light_status = "Please come to a lighted area."
    elif bright_pixels/total_pixels > thres:
      light_status = "Your screen is overexposed. Please adjust."
    else:
      pass
    return light_status



if __name__ == '__main__':
  import cv2
  import time
  from frame_extractor import FrameExtractor

  fproc = FrameProcessor()
  # fex = FrameExtractor(time_interval=50)
  # user_frames = fex.get_frames('../../resources/videos/user_video_webcam.mp4')
  # for frame in user_frames:
  #   start_time = time.time()
  #   frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  #   feats = fproc.get_frame_features(frame)
  #   if feats is not None:
  #     print(feats.shape)
  #   else:
  #     print("No features")
  #   print("Time taken: %f" %(time.time()-start_time))
  #   print('-'*50)

