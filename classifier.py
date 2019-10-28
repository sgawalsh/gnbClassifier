import os, pickle, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

dims = [60, 40]

class modelGNB:
	def __init__(self, modelName, ts = .3):
		x_train, self.x_test, y_train, self.y_test = getTrainingData(ts)
		x_train = x_train.reshape(len(y_train), dims[0] * dims[1]) # flatten image
		
		self.gnb = GaussianNB()
		self.gnb.fit(x_train, y_train)
		
		with open(os.getcwd() + "/models/" + modelName + ".pkl", "wb") as f:
			pickle.dump(self.gnb, f)

def classify(model, input): # classification function
	return model.predict(input.reshape(input.shape[0], dims[0] * dims[1]))

def evaluateAcc(y_test, preds): # evaluation function
	print("Accuracy: ", metrics.accuracy_score(y_test, preds))

def getTrainingData(ts):
	imgsA = np.load(os.getcwd() + "/for_candidate/class_a.npy")
	imgsB = np.load(os.getcwd() + "/for_candidate/class_b.npy")
	
	data = np.append(imgsA, imgsB, axis = 0)
	labels = np.append(np.zeros(imgsA.shape[0]), np.full(imgsB.shape[0], 1)) # a's labelled as 0, b's as 1
	return train_test_split(data, labels, test_size = ts)
	
if __name__ == "__main__":
	model = modelGNB("test")
	preds = classify(model.gnb, model.x_test)
	evaluateAcc(model.y_test, preds)