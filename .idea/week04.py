from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas

# got some help from the book


# read in data
def read_in_iris():
    iris = datasets.load_iris()

    # scaler = StandardScaler()
    # scaler.fit(iris.data)
    # scaled_data = scaler.transform(iris.data)
    # bin data
    binned_data = np.ndarray(shape=(len(iris.data), 4), dtype=pandas._libs.interval.Interval)
    # bins = []
    for i in range(0, len(iris.data)-1):
        binned_data[i] = pandas.cut(iris.data[i], 2)
    # Show the data (the attributes of each instance)
    # print(binned_data)

    # Show the target values (in numeric format) of each instance
    # print(iris.target)

    # Show the actual target names that correspond to each number
    # print(iris.target_names)
    headers = ["sepal length", "sepal width", "petal length", "petal width"]
    data_train, data_test, target_train, target_test = train_test_split(binned_data, iris.target, test_size=0.30,
                                                                        random_state=42)
    return data_train, data_test, target_train, target_test, headers


def calc_entropy(labels, base=2):
  tuple = np.unique(labels, return_counts=True)
  value = tuple[0]
  counts = tuple[1]
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()


# set up for making a tree using nodes
class Node:
    def __init__(self, data, next_down):
        self.data = data
        # list of next nodes down the tree
        self.next_down = next_down


class Tree_Model:
    def __init__(self, data, target, headers):
        self.data_train = data
        self.target_train = target
        self.headers = headers

    def findPath(self, tree, start, end, pathSoFar):
        print(type(tree))
        # print(type(start))
        # print(type(pathSoFar))
        pathSoFar = pathSoFar + tree[start]
        if start == end:
            return pathSoFar
        if start not in tree:
            return None
        for node in tree[start]:
            if node not in pathSoFar:
                newpath = findPath(tree, node, end, pathSoFar)
                return newpath
        return None

    def predict(self, data_test, target_test):
        prediction = []
        # print("length of data", len(self.data))
        for i in range(0, len(data_test)):
            prediction.append(self.predictOne(data_test[i]))

        # print("prediction length", len(prediction))
        return prediction

class tree_classifier:

    def calc_info_gain(self, data, classes, feature):
        gain = 0

        nData = len(data)
        # List the values that feature can take
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])
            featureCounts = np.zeros(len(values))
            entropy = np.zeros(len(values))
            valueIndex = 0

        # Find where those values appear in data[feature] and the corresponding class
            for value in values:
                dataIndex = 0

        newClasses = []
        for datapoint in data:
            if datapoint[feature] == value:
                featureCounts[valueIndex] += 1
        newClasses.append(classes[dataIndex])
        dataIndex += 1
        # Get the values in newClasses
        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass) == 0:
                classValues.append(aclass)
        classCounts = np.zeros(len(classValues))
        classIndex = 0
        for classValue in classValues:
            for aclass in newClasses:
                if aclass == classValue:
                    classCounts[classIndex] += 1
        classIndex += 1
        for classIndex in range(len(classValues)):
            entropy[valueIndex] += calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
            valueIndex += 1
            return gain

    def fit(self, data, classes, featureNames):
        # Various initialisations suppressed
        frequency = []
        for i in range(0, len(data)):
            frequency.append(np.unique(classes, return_counts=True))
        nData = len(data)
        nFeatures = len(featureNames)
        default = classes[np.argmax(frequency[i][0])]
        if nData == 0 or nFeatures == 0:
            # Have reached an empty branch
            return default
        elif classes.count(classes[0]) == nData:
            # Only 1 class remains
            return classes[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                g = self.calc_info_gain(data,classes,feature)
                gain[feature] = calc_entropy(data[-1]) - g
                bestFeature = np.argmax(gain)
                tree = {featureNames[bestFeature]:{}}
            # Find the possible feature values
            for value in values:
                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature]==value:
                        if bestFeature==0:
                            datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature==nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            datapoint.extend(datapoint[bestFeature+1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature+1:])
                            newData.append(datapoint)
                            newClasses.append(classes[index])
                            index += 1
        # Now recurse to the next level
        subtree = make_tree(newData,newClasses,newNames)
        # And on returning, add the subtree on to the tree
        tree[featureNames[bestFeature]][value] = subtree
        return tree

# actual beginning of doing stuff
data_train, data_test, target_train, target_test, headers = read_in_iris()

classifier = tree_classifier()
model = classifier.fit(np.column_stack((data_train, target_train)), headers, np.unique(target_train))
target_predicted =  model.predict(data_test, target_test)

# find accuracy
i = 0
correct = 0
for stuff in target_predicted:
    if target_predicted[i] == target_test[i]:
        correct += 1
    i += 1

accuracy = correct / len(target_predicted)
accuracy *= 100

# print accuracy
print("accuracy of decision tree is {:.2f}%".format(accuracy))


