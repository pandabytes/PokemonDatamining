import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def precisionRecallCurve(actuals, predictionScores):
    ''' Get the precision and recall values to plot the Precision-Recall Curve.
        This function treat each label as "positive" and others as "negative".
        In other words, this function "normalizes" the the prediction labels to be
        binary label. 

        Note: This is different from the getPrecisionsAndRecalls() function above where the
        labels are transformed to binary labels.
    '''
    # Use the prediction score as the threshold
    thresholds = set(predictionScores)
    precisions = {}
    recalls = {}

    for label in actuals:
        labelPrecisions = []
        labelRecalls = []
        binarizedLabels = []

        # Tranfrom the labels to binary form: 0 and 1
        for i, j in zip(actuals, predictionScores):
            binarizedLabel = int(i == label)
            binarizedLabels.append((binarizedLabel, j))

        # Compute precision and recall for each threshold
        for t in thresholds:
            positive = 0
            truePositive = 0
            truePositiveFalsePositive = 0

            for b in binarizedLabels:
                if (b[0] == 1):
                    positive += 1
                if (b[0] == 1 and b[1] >= t):
                    truePositive += 1
                if (b[1] >= t):
                    truePositiveFalsePositive += 1

            precision = truePositive / truePositiveFalsePositive
            recall = truePositive / positive
            labelPrecisions.append(precision)
            labelRecalls.append(recall)

        precisions[label] = labelPrecisions
        recalls[label] = labelRecalls

    return precisions, recalls

def plotPrecisionRecallCurve(precisions, recalls, labelColors, plotTitle):
    ''' Plot the Precision-Recall Curve for each label '''
    if (len(precisions) != len(recalls)):
        raise ValueError("Length of precisions and recalls must match")
    if (precisions.keys() != recalls.keys()):
        raise ValueError("Keys in precisions and recalls must match")

    fig = plt.figure(plotTitle)
    plots = []
    for label in precisions.keys():
        labelPrecisions = precisions[label]
        labelRecalls = recalls[label]

        # Group precision and recall in tuple and sort by recall in ascending order
        labelPr = [(1, 0)]
        for i, j in zip(labelPrecisions, labelRecalls):
            labelPr.append((i, j))
        labelPr.sort(key=lambda x: x[1])
        
        # Plot the PR curve
        plot, = plt.plot([i[1] for i in labelPr], [i[0] for i in labelPr], c=labelColors[label], label=label)
        plots.append(plot)

    plt.title(plotTitle)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(handles=plots, loc="best")
    return fig

def plotRocCurve(actuals, predictionScores, labelColors, plotTitle):
    ''' Plot the Receiver Operating Characteristics curve '''
    fig = plt.figure(plotTitle + " Receiver Operating Characteristic")
    plots = []
    for label, color in labelColors.items():
        fpr, tpr, thresholds = roc_curve(y_true=actuals, y_score=predictionScores, pos_label=label)
        aucScore = auc(fpr, tpr)
        plot = plt.plot(fpr, tpr, color, label='{0} AUC = {1:.2f}'.format(label, aucScore))[0]
        plots.append(plot)

    randomnessLine = plt.plot([0,1], [0,1], "r--", label="Random")[0]
    plots.append(randomnessLine)

    plt.title(plotTitle + " Receiver Operating Characteristic")
    plt.legend(handles=plots, loc="best")
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    return fig
