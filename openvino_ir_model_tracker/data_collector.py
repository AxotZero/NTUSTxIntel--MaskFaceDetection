import pandas as pd


class DataCollector():
	def __init__(self, save_path):
		self.data_len = 0
		self.save_path = save_path
		
		self.data = {'timestamp': [], 'mask':[], 'gender':[], 'age':[]}
		pd.DataFrame(self.data).to_csv(self.save_path, mode='w', index=None)


	def add(self, tracking_object):

		self.data['timestamp'].append(tracking_object.classified_timestamp)
		self.data['mask'].append(tracking_object.mask)
		self.data['gender'].append(tracking_object.gender)
		self.data['age'].append(tracking_object.age)
		self.data_len += 1

		if self.data_len >= 5:
			self.save()

	def save(self):
		df = pd.DataFrame(self.data)
		df.to_csv(self.save_path, mode='a', header=None, index=None)

		self.data = {'timestamp': [], 'mask':[], 'gender':[], 'age':[]}

		self.data_len = 0








