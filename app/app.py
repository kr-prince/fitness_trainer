import cv2
import time
import streamlit as st

from main_engine import MainEngine


# Setup the various UI components
RUN_STATUS = st.checkbox('Start', value=True)
USER_STREAM = st.empty()
USER_FEEDBACK = st.empty()
REF_VID = st.sidebar.empty()
SETS_COUNTS = st.sidebar.empty()
USER_FEEDBACK_TEMPLATE = '<span style="font-family:sans-serif; color:Red; font-size: 2.0rem;">%s</span>'
SETS_COUNTS_TEMPLATE = '<span style="font-family:sans-serif; font-size: 1.2rem;">%s</span>'
FINAL_REPORT_TEMPLATE = '<span style="font-family:sans-serif; color:Red; font-size: 2.0rem;">\
  <ul style="list-style-type:circle;">%s</ul></span>'

output = st.session_state.get('output', None)
if (output is not None) and (not RUN_STATUS):
  # If the app is not in running state, we show the exercises data for the user so far
  sets_counts = output['sets_counts']
  pose_counts = output['pose_counts']
  exercise_data_text = ''
  for k,v in sets_counts.items():
    # First display list of completed sets of each exercise
    exercise_data_text += '<li>'+k.replace('_',' ').title()+' - '+str(v)+'</li>'
  non_ex_counts = pose_counts.get('random', 0)
  ex_counts = sum([v for k,v in pose_counts.items() if k != 'random'])
  # Exercise intensity is the ratio of exercise related poses with total(including random) poses
  k,v = 'Intensity',round(ex_counts/(ex_counts + non_ex_counts),1)
  exercise_data_text += '<li>'+k.replace('_',' ').title()+' - '+str(v)+'</li>'
  USER_STREAM.write(FINAL_REPORT_TEMPLATE %exercise_data_text, unsafe_allow_html=True)

def main():
  second_start = time.time()
  me = MainEngine()

  camera = cv2.VideoCapture(0)
  while camera.isOpened() and RUN_STATUS:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)   # Horizontally flip the frame for mirror image
    USER_STREAM.image(frame, channels='BGR')
    
    time_now = time.time()
    if (time_now-second_start)*1000 >= 200 and RUN_STATUS:
      me.push((time_now, frame))
      # cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_RGB2BGR)
      # cv2.imwrite("./pics/pic_%d.png"%i, frame)
      second_start = time.time()
    
    output = me.get_output()
    if output is not None:
      if 'add' in output:
        pose = output['add']
        REF_VID.image('../resources/videos/%s.gif'%pose, caption='Wahoo Fitness')
        sets_counts = output['sets_counts']
        sets_counts = ['%s : %d' %(key.replace('_',' ').title(), val) for key, val in sets_counts.items()]
        sets_counts_text = '<br>'.join(sets_counts)
        SETS_COUNTS.write(SETS_COUNTS_TEMPLATE %sets_counts_text, unsafe_allow_html=True)
      if 'correction' in output:
        correction = output['correction']
        USER_FEEDBACK.write(USER_FEEDBACK_TEMPLATE %correction, unsafe_allow_html=True)
      if 'error' in output:
        error = output['error']
        USER_FEEDBACK.write(USER_FEEDBACK_TEMPLATE %error, unsafe_allow_html=True)
      
      st.session_state['output'] = output
  
  camera.release()
  cv2.destroyAllWindows()

  # me.speech.stop()

  output = me.get_output()
  while output is not None:
    # process any remaining outputs to make the queue empty
    st.session_state['output'] = output
  
  # me.stop()


if __name__ == '__main__' and (output is not None):
  # This prevents streamlit from re-runnning the whole app in case of click or change event
  main()