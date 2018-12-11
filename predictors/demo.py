import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut
import decisionTree as dt
import naiveBayes as nb
from sklearn.metrics import roc_curve, auc

# Load the Pokemon data
fileName = r'.\Pokemon_Cleaned.tsv'
columnTypes = {"Name": str, "Category": str, "Type 1": str, "Type 2": str, 
               "Ability 1": str, "Ability 2": str, "Ability 3": str, "Group": str}
data = pd.read_csv(fileName, header=0, sep='\t', dtype=columnTypes)
print("Random samples data:")
print(data.sample(5))

# Declare global variables
Target = "Group"
Labels = data[Target].unique()
LabelColors = {"Ordinary": "b", "Legendary": "r", "Ultra Beast": "g"}
DropColumns = ["Pokedex", "Name", "Generation", "Category"]
ReducedData = data.drop(DropColumns, axis=1)
Training, Test = None, None

# Load the data points to Training and Test variables
with open("training.txt", "r") as file:
    indeces = list(map(lambda x: int(x), file.readline().strip().split(" ")))
    Test = data.drop(index=indeces)
    Training = data.drop(index=Test.index)

print()

#############################################################################################################
#############################################################################################################
#############################################################################################################

if __name__ == "__main__":
	import time
	timeStart = time.time()
    
	# Use Decision Tree to train on the training set and predict on the test set
	dtree = dt.DecisionTree(Target, maxDepth=3)
	dtree.train(Training, quiet=True)
	dtPred = dtree.classify(Test.drop([Target], axis=1))

	# Confusion Matrix, Precision, Recall, F Scores, Misclassification Error
	dtMatrix = ut.buildConfusionMatrix(dtPred["Prediction"], Test[Target], Labels)
	dtPrecisions, dtRecalls = ut.getPrecisionsAndRecalls(dtMatrix, Labels)
	dtFScores = ut.computeFScores(dtPrecisions, dtRecalls)
	print("Decision Tree Error: {0:.2f}%".format(ut.computeError(dtPred["Prediction"], Test["Group"]) * 100))
	print("Decision Tree Avg F-score: {0:.2f}%".format(dtFScores[0]))

	# Decision Tree ROC
	dtRocFig = plt.figure("Decision Tree ROC")
	plots = []
	for label in Labels:
	    c = LabelColors[label]
	    fpr, tpr, rocThresholds = roc_curve(y_true=Test[Target], y_score=[v[1] for v in dtPred.values], pos_label=label)
	    rocAuc = auc(fpr, tpr)
	    plot, = plt.plot(fpr, tpr, c, label='{0} AUC = {1:.2f}'.format(label, rocAuc))
	    plots.append(plot)
	  
	dtRocPlot, = plt.plot([0,1],[0,1],'r--', label="Random")
	plots.append(dtRocPlot)
	plt.title('Decision Tree Receiver Operating Characteristic')
	plt.xlim([0, 1.05])
	plt.ylim([0, 1.05])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.legend(handles=plots, loc='best')
	dtRocFig.show()

	# Decision Tree PR curve
	p, r = ut.precisionRecallCurve(Test[Target], dtPred["Probability"])
	dtPrFig = ut.plotPrecisionRecallCurve(p, r, LabelColors, "Decision Tree PR Curve")
	dtPrFig.show()


#############################################################################################################
#############################################################################################################
#############################################################################################################


	# Use Naive Bayes to train on the training set and predict on the test data set
	nBayes = nb.NaiveBayes(Target, Labels)
	nBayes.train(Training, quiet=True)
	nbPred = nBayes.classify(Test.drop([Target], axis=1), quiet=True)

	nbMatrix = ut.buildConfusionMatrix(nbPred["Prediction"], Test[Target], Labels)
	nbPrecisions, nbRecalls = ut.getPrecisionsAndRecalls(nbMatrix, Labels)
	nbFScores = ut.computeFScores(nbPrecisions, nbRecalls)
	print("Naive Bayes Error: {0:.2f}%".format(ut.computeError(nbPred["Prediction"], Test["Group"]) * 100))
	print("Naive Bayes Avg F-score: {0:.2f}%".format(nbFScores[0]))

	# Naive Bayes ROC
	# For each label, treat other labels as "negative"
	del plots[:]
	nbRocFig = plt.figure("Naive Bayes ROC")
	for label in Labels:
	    c = LabelColors[label]
	    fpr, tpr, thresholds = roc_curve(y_true=Test[Target], y_score=[v[1] for v in nbPred.values], pos_label=label)
	    rocAuc = auc(fpr, tpr)
	    plot, = plt.plot(fpr, tpr, c, label='{0} AUC = {1:.2f}'.format(label, rocAuc))
	    plots.append(plot)
	    
	nbRocPlot, = plt.plot([0,1],[0,1],'r--', label="Random")
	plots.append(nbRocPlot)
	plt.title('Naive Bayes Receiver Operating Characteristic')
	plt.xlim([0, 1.05])
	plt.ylim([0, 1.05])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.legend(handles=plots, loc='best')
	nbRocFig.show()

	# Naive Bayes PR curve
	p, r = ut.precisionRecallCurve(Test[Target], nbPred["Probability"])
	nbPrFig = ut.plotPrecisionRecallCurve(p, r, LabelColors, "Naive Bayes PR Curve")
	nbPrFig.show()

	elapsedTime = time.time() - timeStart
	print("Elapsed Time: {0:.2f} seconds".format(elapsedTime))

	input("***** Press enter to end the program *****")