from collections import Counter

from utils.worker import Worker
from utils.frame_processor import FrameProcessor
from utils.pose_classifier import PoseClassifier
from utils.speech_engine import SpeechEngine


class MainEngine(Worker):
	"""
		This is the main driver class running all the utilities and sending updates to UI
	"""
	def __init__(self):
		# setup utilities
		self.fproc = FrameProcessor()
		self.pose_clf = PoseClassifier()
		self.speech = SpeechEngine()
		self.clf_window = []
		self.last_lightcheck = None
		self.pose_counts_hist = {}
		self.run()

	def name(self):
		return str(self.__class__.__name__)

	def push(self, time_frame):
		"""   push a frame to the main engine for processing   """
		self.inputs.put(time_frame)
	
	def main(self):
		count = 0
		ongoing_pose = None
		# sets_counts keeps a count of all the completed exercise, whereas pose_counts keeps 
		# count of all types of classified pose. These is later used for analysis between exercise
		# and non-exercise(random) poses
		pose_counts, sets_counts = {}, {}

		while True:
			inp = self.inputs.get()
			if (type(inp) is int) and (inp == 0):
				# this is the kill signal
				break
			
			else:
				time_stamp, frame = inp
				try:
					# get features from frame
					feats = self.fproc.get_frame_features(frame)
					if feats is not None:
						pose_clas = self.pose_clf.classify(feats)
						# to avoid jitter we do smoothening using moving window
						self.clf_window.append(pose_clas)
						self.clf_window = self.clf_window[-3:]
						pose_clas = Counter(self.clf_window).most_common(1)[0][0]
						# Get recommendation based on pose correction check
						correction = self.fproc.pose_corrector(pose_clas)

						exc_clas = pose_clas.split('-')[0]
						pose_counts[exc_clas] = pose_counts.get(exc_clas, 0)+1
						# print(exercise_counts)
						if exc_clas != 'random':
							if pose_clas.endswith('-end') and (ongoing_pose is not None):
								# Handles Jumping Jacks, Squats, Crunches, Lunges
								if (ongoing_pose==pose_clas.replace('end', 'start')):
									sets_counts[exc_clas] = sets_counts.get(exc_clas, 0)+1
									self.outputs.put({'add': exc_clas, 'correction': correction, 'sets_counts': sets_counts,
																			'pose_counts': pose_counts})
							if pose_clas.endswith('planks'):
								# Handles planks. Planks are measured in milliseconds
								sets_counts[exc_clas] = sets_counts.get(exc_clas, 0)+0.2
								self.outputs.put({'add': 'planks', 'correction': correction, 'sets_counts': sets_counts,
																		'pose_counts': pose_counts})
							ongoing_pose = pose_clas
						
					# we also do a lightcheck after every 2 secs
					if (self.last_lightcheck is None) or (time_stamp-self.last_lightcheck > 2):
						status = self.fproc.lightcheck(frame)
						self.last_lightcheck = time_stamp
						assert status is None, status
				except Exception as ex:
					error_text = str(ex)
					self.outputs.put({'error':error_text, 'pose_counts': pose_counts, 'sets_counts': sets_counts})
					self.speech.say(error_text)
				finally:
					# print("Speech Put : %d" %ocount)
					count += 1
					# print("Out Count: %d"%count)
					pass

