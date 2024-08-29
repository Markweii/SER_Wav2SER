import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def parse_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    labels = []
    # 逐行读取文件内容，直到遇到 "======" 停止
    for line in lines:
        if '====' in line:
            break
        if '=>' in line:
            label, index = line.split('=>')
            labels.append(label.strip().strip("'"))
    
    return labels



def plotc(fpath):
    CSVFILE = fpath + '/predictions.csv'
    LABELFILE = fpath + '/save/label_encoder.txt'
    test_df = pd.read_csv(CSVFILE)

    predictedValue = test_df['prediction'].values
    actualValue = test_df['true_value'].values

    # Parse labels from the label encoder file
    labels = parse_labels(LABELFILE)

    cmt = confusion_matrix(actualValue, predictedValue)

     # Classification report
    report = classification_report(actualValue, predictedValue, target_names=labels, output_dict=True)
    # Accuracy
    
    print("WA accuracy:", accuracy_score(actualValue, predictedValue))
    # Weighted Accuracy
    weighted_acc = np.sum([report[label]['precision'] * report[label]['support'] for label in labels]) / np.sum([report[label]['support'] for label in labels])
    print("UA accuracy:", weighted_acc)

    # Classification report
    print(classification_report(actualValue, predictedValue, target_names=labels))

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cmt, display_labels=labels)
    disp.plot()
    plt.savefig(fpath+'/confusionmatrix.jpg')

if __name__ == "__main__":
    plotc("C:/Users/503503/Desktop/speechbrain/custom/wav2vec2/results/train_with_wav2vec2/1999")