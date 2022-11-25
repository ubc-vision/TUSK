from datetime import datetime

class tensorboard_scheduler():
	def __init__(self, config):
		self.eval_interval = config.summary_interval
		self.save_interval = config.save_interval
		self.valid_interval = config.valid_interval
		self.stop_time = config.stop_time

		self.start_time = datetime.now()
		self.eval_counter = 0
		self.save_counter = 0
		self.valid_counter = 0
    	
	def schedule(self):

		delta_secs = (datetime.now() - self.start_time).total_seconds()
		delta_mins = delta_secs/60

		if self.stop_time!=-1 and delta_mins>self.stop_time:
			return False, False, False

		# if delta_mins<min(self.eval_interval,self.save_interval,self.valid_interval):
		# 	return False, False, False

		if delta_mins > self.eval_counter*self.eval_interval:
			eval_flag = True
			self.eval_counter = int(delta_mins/self.eval_interval) + 1
		else: 
			eval_flag =False

		if delta_mins > self.save_counter*self.save_interval:
			save_flag = True
			self.save_counter = int(delta_mins/self.save_interval)  + 1
		else:
			save_flag = False

		if delta_mins > self.valid_counter*self.valid_interval:
			valid_flag = True
			self.valid_counter = int(delta_mins/self.valid_interval)  + 1
		else:
			valid_flag = False

		return eval_flag, save_flag, valid_flag
	def get_delta_time(self):
		delta_secs = (datetime.now() - self.start_time).total_seconds()
		return delta_secs

