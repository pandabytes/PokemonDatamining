import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import DecisionTree, NaiveBayes
from sklearn.metrics import roc_curve, auc
from utils import plots as pl, samplings as sp, evaluations as ev

# Load the Pokemon data
filePath = os.path.join("..", "data", "Pokemon_Cleaned.tsv")
columnTypes = {"Name": str, "Category": str, "Type 1": str, "Type 2": str, 
               "Ability 1": str, "Ability 2": str, "Ability 3": str, "Group": str}
data = pd.read_csv(filePath, header=0, sep='\t', dtype=columnTypes)
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
sampleFilePath = os.path.join("..", "data", "sample.txt")
with open(sampleFilePath, "r") as file:
    indeces = list(map(lambda x: int(x), file.readline().strip().split(" ")))
    Test = ReducedData.drop(index=indeces)
    Training = ReducedData.drop(index=Test.index)

print()

# ############################################################################################################
# ############################################################################################################
# ############################################################################################################

if __name__ == "__main__":
	timeStart = time.time()
    
	# Use Decision Tree to train on the training set and predict on the test set
	dtree = DecisionTree(Target, maxDepth=3)
	dtree.train(Training, quiet=True)
	dtPred = dtree.classify(Test.drop([Target], axis=1))

	# Confusion Matrix, Precision, Recall, F Scores, Misclassification Error
	dtMatrix = ev.buildConfusionMatrix(dtPred["Prediction"], Test[Target], Labels)
	dtPrecisions, dtRecalls = ev.getPrecisionsAndRecalls(dtMatrix, Labels)
	dtFScores = ev.computeFScores(dtPrecisions, dtRecalls)
	print("Decision Tree Error: {0:.2f}%".format(ev.computeError(dtPred["Prediction"], Test[Target]) * 100))
	print("Decision Tree Avg F-score: {0:.2f}".format(dtFScores[0]))

	# # Decision Tree ROC
	dtRoc = pl.plotRocCurve(Test[Target], dtPred["Probability"], LabelColors, "Decision Tree")
	dtRoc.show()

	# Decision Tree PR curve
	p, r = pl.precisionRecallCurve(Test[Target], dtPred["Probability"])
	dtPrFig = pl.plotPrecisionRecallCurve(p, r, LabelColors, "Decision Tree PR Curve")
	dtPrFig.show()


# # #############################################################################################################
# # #############################################################################################################
# # #############################################################################################################


	# Use Naive Bayes to train on the training set and predict on the test data set
	nBayes = NaiveBayes(Target, Labels)
	nBayes.train(Training, quiet=True)
	nbPred = nBayes.classify(Test.drop([Target], axis=1), quiet=True)

	nbMatrix = ev.buildConfusionMatrix(nbPred["Prediction"], Test[Target], Labels)
	nbPrecisions, nbRecalls = ev.getPrecisionsAndRecalls(nbMatrix, Labels)
	nbFScores = ev.computeFScores(nbPrecisions, nbRecalls)
	print("Naive Bayes Error: {0:.2f}%".format(ev.computeError(nbPred["Prediction"], Test["Group"]) * 100))
	print("Naive Bayes Avg F-score: {0:.2f}".format(nbFScores[0]))

	# Naive Bayes ROC
	nbRoc = pl.plotRocCurve(Test[Target], nbPred["Probability"], LabelColors, "Naive Bayes")
	nbRoc.show()

	# Naive Bayes PR curve
	p, r = pl.precisionRecallCurve(Test[Target], nbPred["Probability"])
	nbPrFig = pl.plotPrecisionRecallCurve(p, r, LabelColors, "Naive Bayes PR Curve")
	nbPrFig.show()

	elapsedTime = time.time() - timeStart
	print("Elapsed Time: {0:.2f} seconds".format(elapsedTime))

	input("***** Press enter to end the program *****")