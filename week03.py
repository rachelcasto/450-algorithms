
from sklearn import datasets, preprocessing, model_selection
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas


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


def Gaussian_Predict(car_data, car_targets):
    num_folds = 10
    subset_size = int(len(car_data) / num_folds)
    accuracy = 0.0
    classifier = GaussianNB()
    predictions = []
    kf = model_selection.KFold(n_splits=num_folds, shuffle=False)
    for train_index, test_index in kf.split(car_data):
        # data = car_data[train_index]
        # targets = car_targets[train_index]
        for j in train_index:
            predictions.append(
                classifier(car_data[train_index[j]], car_data[test_index[j]], car_data[train_index[j]], car_data[test_index[j]]))

        for i in test_index:
            target_predicted = kf.predict(car_data[i])
            # find accuracy
            # i = 0
            correct = 0
            if target_predicted[i] == car_targets[i]:
                correct += 1
            i += 1

        my_accuracy = correct / len(target_predicted)
        accuracy += my_accuracy

    accuracy = (accuracy * 100) / len(test_index)
    # k_fold_validation(0, subset_size)
    return accuracy


def readInCar():
    car_data = r"C:\Users\casto\Documents\BYUI\Fall 2018\CS 450\carData.csv"
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    dataset = pandas.read_csv(car_data, names=headers, index_col=False)
    df = pandas.DataFrame(dataset)
    # df = dataset
    df = df.apply(LabelEncoder().fit_transform)
    # scaled_data = pandas.get_dummies(df)
    # scaled_data = ((df - df.min()) / (df.max() - df.min()))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(df)
    # scaled_data = pandas.DataFrame(np_scaled)

    # scaler = StandardScaler()
    # scaler.fit(dataFrame)
    # scaled_data = scaler.transform(dataFrame)
    col = len(df.columns)
    just_data = df.iloc[:, 1:]
    # just_data = just_data.values
    targets = df.iloc[:, 0]
    # targets = targets.values
    return just_data, targets


def readInAutism():
    data = r"C:\Users\casto\Documents\BYUI\Fall 2018\CS 450\autismAdultData.csv"
    headers = ["Age", "Gender", "Ethnicity", "Born with jaundice", "Family member with PDD",
               "Who is completing the test", "Country of residence", "Screening Method Type",
               "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q19", "Screening Score"]
    # data = r"C:\Users\casto\Documents\BYUI\Fall 2018\CS 450\carData.csv"
    dataset = pandas.read_csv(data, names=headers, index_col=False, na_values="?")
    dataset = dataset.dropna(how="any")
    df = pandas.DataFrame(dataset)
    # df = dataset
    df = df.apply(LabelEncoder().fit_transform)
    # scaled_data = pandas.get_dummies(df)
    # scaled_data = ((df - df.min()) / (df.max() - df.min()))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(df)
    # scaled_data = pandas.DataFrame(np_scaled)

    # scaler = StandardScaler()
    # scaler.fit(dataFrame)
    # scaled_data = scaler.transform(dataFrame)
    col = len(df.columns)
    just_data = df.iloc[:, 1:]
    # just_data = just_data.values
    targets = df.iloc[:, 0]
    # targets = targets.values
    return just_data, targets

def readInMPG():
    data = r"C:\Users\casto\Documents\BYUI\Fall 2018\CS 450\auto-mpg.data"
    headers = ["MPG", "cylinders", "displacement", "horsepower", "weight", "acceleration",
               "model year", "origin", "car name"]
    dataset = pandas.read_csv(data, names=headers, index_col=False, na_values="?", delimiter="\t")
    dataset = dataset.dropna(how="any")
    df = pandas.DataFrame(dataset)
    # df = dataset
    df = df.apply(LabelEncoder().fit_transform)
    # scaled_data = pandas.get_dummies(df)
    # scaled_data = ((df - df.min()) / (df.max() - df.min()))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # np_scaled = min_max_scaler.fit_transform(df)
    # scaled_data = pandas.DataFrame(np_scaled)

    # scaler = StandardScaler()
    # scaler.fit(dataFrame)
    # scaled_data = scaler.transform(dataFrame)
    col = len(df.columns)
    just_data = df.iloc[:, 1:]
    # just_data = just_data.values
    targets = df.iloc[:, 0]
    # targets = targets.values
    return just_data, targets


def k_fold_validation(i, subset_size):
    # testing_this_round = car_data[i * subset_size:][:subset_size]
    # training_this_round = car_data[:i * subset_size] + car_data[(i + 1) * subset_size:]
    # target_testing_this_round = car_targets[i * subset_size:][:subset_size]
    # target_training_this_round = car_targets[:i * subset_size] + car_targets[(i + 1) * subset_size:]
    classifier = GaussianNB()
    # model = classifier.fit(training_this_round, target_training_this_round)
    #
    # target_predicted = model.predict(target_testing_this_round)

    # train, test = KFold(len(car_targets), n_folds=10, shuffle=True, random_state=42)
    # clf = classifier
    # target_predicted = cross_val_score(clf, car_data, car_targets, cv=k_fold, n_jobs=1)


car_data, car_targets = readInCar()
accuracy = Gaussian_Predict(car_data, car_targets)
print("total accuracy of the car data is {:.2f}%".format(accuracy))

autism_data, autism_targets = readInAutism()
accuracy = Gaussian_Predict(autism_data, autism_targets)
print("total accuracy of the autism data is {:.2f}%".format(accuracy))

mpg_data, mpg_targets = readInMPG()
accuracy = Gaussian_Predict(mpg_data, mpg_targets)
print("total accuracy of the mpg data is {:.2f}%".format(accuracy))

# Show the data (the attributes of each instance)
# print(car_data)

# Show the target values (in numeric format) of each instance
# print(car_targets)

# Show the actual target names that correspond to each number
# print(iris.target_names)




# data_train, data_test, target_train, target_test = train_test_split(car_data, car_targets, test_size=0.30, random_state=42)

# print("training data", da)

##################################################
# GAUSSIAN STUFF
##################################################





##################################################
# KNN STUFF
##################################################
# classifier = KNNClassifier()
# model = classifier.fit(data_train, target_train, 1)
# target_predicted_knn = model.predict(data_test, target_test)
# # print(len(target_test))
#
# # find accuracy of KNN
# correct = 0
# # print("target_predicted_knn", len(target_predicted_knn))
# for i in range(0, len(target_predicted_knn)):
#     if target_predicted_knn[i] == target_test[i]:
#         correct += 1
#
# accuracy_knn = correct / len(target_predicted_knn)
# accuracy_knn *= 100
#
# #print accuracy
# print("accuracy of KNN is {:.2f}%".format(accuracy_knn))