import os
import cv2

class FrameExtractor(object):
  """
    Class to extract frames from a given video at every specific intervals

    Arguments:
      time_interval : time interval of frame capturing in ms (default 100ms)
      debug : print info if debug is true
  """
  def __init__(self, time_interval=100, debug=False):
    self.time_interval = time_interval
    self.debug = debug
  
  def get_frames(self, file_path):
    """
      Returns the frames from video file

      Arguments:
        file_path : path to video file
    """
    assert os.path.isfile(file_path), "File not found: %s"%file_path
    try:
      vidObj = cv2.VideoCapture(file_path)
      frames, frame_count = list(), 0
      # start reading video stream
      status, image = vidObj.read()
      while status:
        frame_count += 1
        frames.append(image)
        # set the video time to the corret time as per time_interval
        vidObj.set(cv2.CAP_PROP_POS_MSEC, frame_count*self.time_interval)
        status, image = vidObj.read()
      else:
        if self.debug:
          print("%d frames extracted." %frame_count)

    except Exception as ex:
      frames = None
      print("Error: %s" %str(ex))

    return frames


