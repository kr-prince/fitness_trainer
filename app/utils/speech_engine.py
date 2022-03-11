import time, os
import hashlib
from gtts import gTTS
from playsound import playsound


from .worker import Worker


class SpeechEngine(Worker):
	"""
		This will handle text to speech related stuffs
	"""
	def __init__(self):
		# Setup the audio repository
		self.speech_dir = '../resources/speech/'
		if not os.path.isdir(self.speech_dir):
			os.mkdir(self.speech_dir)
		
		# cache the previously converted and stored speech files for faster processing.
		# Also the time_stamp for each speech is saved so that we do not bombard the user
		# with any single speech continuously.  
		self.speech_history = {file: time.time()-10 for file in os.listdir(self.speech_dir) \
													if os.path.isfile(os.path.join(self.speech_dir, file))}
		self.run()

	def name(self):
		return str(self.__class__.__name__)

	def say(self, text):
		self.inputs.put((time.time(), text))
	
	def main(self):
		while True:
			inp = self.inputs.get()
			if (type(inp) is int) and (inp == 0):
				# this is the kill signal
				break
			else:
				time_stamp, text = inp

			try:
				# convert to hash code and save the processed speech audio
				hash_code = hashlib.md5(text.encode()).hexdigest()
				file_name = hash_code+'.mp3'
				time_hist = self.speech_history.get(file_name)
				if time_hist is None:
					# process the speech only once and then cache it
					speech_obj = gTTS(text=text, lang='en', slow=False)
					speech_obj.save(os.path.join(self.speech_dir, file_name))
					self.speech_history[file_name] = time.time()
					playsound(os.path.join(self.speech_dir, file_name), block=False)
				else:
					# if it was played before then ensure it is not repeated in next 10 secs
					if time_stamp-time_hist > 10:
						self.speech_history[file_name] = time.time()
						playsound(os.path.join(self.speech_dir, file_name), block=False)
				# self.outputs.put("success")
			except Exception as ex:
				print("Error in SpeechEngine : %s" %str(ex))
				# self.outputs.put("error")
			finally:
				# print("Speech Put : %d" %ocount)
				# ocount += 1
				pass
