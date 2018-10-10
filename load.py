from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


##################################################
# CLASSES & FUNCTION
##################################################
# just some code I found to help
# MOST FREQUENT
def mostFrequent(arr, n):
    # Sort the array
    arr.sort()

    # find the max frequency using
    # linear traversal
    max_count = 1
    res = arr[0]
    curr_count = 1

    for i in range(1, n):
        if arr[i] == arr[i - 1]:
            curr_count += 1

        else:
            if curr_count > max_count:
                max_count = curr_count
                res = arr[i - 1]

            curr_count = 1

    # If last element is most frequent
    if curr_count > max_count:
        max_count = curr_count
        res = arr[n - 1]

    return res


# KNN MODEL
class KNNModel:
    def __init__(self, data, target, k):
        self.data_train = data
        self.target_train = target
        self.k = k

    def predictOne(self, datum):
        pairs = {}
        distances = []
        # print("data train length", len(self.data_train))
        i = 0
        for thing in self.data_train:
            # saving time by not taking the sqrt
            distance = (thing[0] - datum[0])**2 + (thing[1] - datum[1])**2 + \
                       (thing[2] - datum[2])**2 + (thing[3] - datum[3])**2
            pairs[distance] = self.target_train[i]
            distances.append(distance)
            i += 1

        nNeighbors = []
        j = 0
        while j < self.k:
            # print("shortest distance", min(distances))
            nNeighbors.append(pairs[min(distances)])
            distances.remove(min(distances))
            j += 1

        # print("\n")
        return mostFrequent(nNeighbors, len(nNeighbors))

    def predict(self, data_test, target_test):
        prediction = []
        # print("length of data", len(self.data))
        for i in range(0, len(data_test)):
            prediction.append(self.predictOne(data_test[i]))

        # print("prediction length", len(prediction))
        return prediction


# KNN CLASSIFIER
class KNNClassifier:

    def fit(self, data, targets, k):
        return KNNModel(data, targets, k)


iris = datasets.load_iris()

scaler = StandardScaler()
scaler.fit(iris.data)
scaled_data = scaler.transform(iris.data)

# # Show the data (the attributes of each instance)
# print(iris.data)
#
# # Show the target values (in numeric format) of each instance
# print(iris.target)
#
# # Show the actual target names that correspond to each number
# print(iris.target_names)


data_train, data_test, target_train, target_test = train_test_split(scaled_data, iris.target, test_size=0.30, random_state=42)

# print("training data", da)

##################################################
# GAUSSIAN STUFF
##################################################
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

target_predicted = model.predict(data_test)

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
print("accuracy of Gaussian is {:.2f}%".format(accuracy))

##################################################
# KNN STUFF
##################################################
classifier = KNNClassifier()
model = classifier.fit(data_train, target_train, 5)
target_predicted_knn = model.predict(data_test, target_test)
# print(len(target_test))

# find accuracy of KNN
correct = 0
# print("target_predicted_knn", len(target_predicted_knn))
for i in range(0, len(target_predicted_knn)):
    if target_predicted_knn[i] == target_test[i]:
        correct += 1

accuracy_knn = correct / len(target_predicted_knn)
accuracy_knn *= 100

#print accuracy
print("accuracy of KNN is {:.2f}%".format(accuracy_knn))


###############################
# Random stuff
###############################
x = [1,2,3,4,5,6,7,8,9,10,11,12]
kf = KFold(12, n_folds=3)

for train_index, test_index in kf:
    print(train_index, test_index)