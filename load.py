from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

# # Show the data (the attributes of each instance)
# print(iris.data)
#
# # Show the target values (in numeric format) of each instance
# print(iris.target)
#
# # Show the actual target names that correspond to each number
# print(iris.target_names)

data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.30, random_state=42)

classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

target_predicted = model.predict(data_test)

#find accuracy
i = 0
correct = 0
for stuff in target_predicted:
    if target_predicted[i] == target_test[i]:
        correct += 1
    i += 1

accuracy = correct / len(target_predicted)
accuracy *= 100

#print accuracy
#print(accuracy)
print("accuracy is {:.2f}%".format(accuracy))


#model
class HardCodedModel:
    def predict(self, data):
        # target = data[:]
        # target[:] = 0
        # print(len(target))
        prediction = []
        for i in range(0, len(data)):
            prediction.append(0)
        return prediction


#own classifier
class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel()


classifier = HardCodedClassifier()
model = classifier.fit(data_train, target_train)
target_predicted_hardcoded = model.predict(data_test)
#print(len(target_test))

#find accuracy of hard coded
correct = 0
for i in range(0, len(target_predicted_hardcoded)):
    if target_predicted_hardcoded[i] == target_test[i]:
        correct += 1

accuracy_hardcoded = correct / len(target_predicted_hardcoded)
accuracy_hardcoded *= 100

#print accuracy
print("accuracy of hard coded is {:.2f}%".format(accuracy_hardcoded))