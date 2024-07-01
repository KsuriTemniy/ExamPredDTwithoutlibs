import csv
import random

train_students = []
test_students = []


with open('student_exam_data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        train_students.append(row)
    train_students = train_students[1:].copy()

with open('input.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        test_students.append(row)
    test_students = test_students[1:].copy()

normalization = [{'Да': 0.2, 'Нет': 0.1},
                 {'Хорошее': 0.05, 'Нормальное': 0.02, 'Плохое': 0.01},
                 {'0': 0.01,'1': 0.04, '2-3': 0.05, '4+': 0.07},
                 {'Низкая': 0.8, 'Средняя': 1.3, 'Высокая': 1.7},
                 {'Последний час': 0.5,'Последняя ночь': 0.9,'За несколько дней': 2.0, 'За неделю': 5.0},
                 {'Нет': 0, 'Да': 1}]
train_data = []
test_data = []
for row in train_students:
    temp_str = [0]*9
    temp_str[:3] = map(lambda x: int(x)*0.1, [row[0],row[1],row[2]])
    for i in range(3, 9):
        temp_str[i] = normalization[i-3][row[i]]
    train_data.append(temp_str)
for row in test_students:
    temp_str = [0]*9
    temp_str[:3] = map(lambda x: int(x)*0.1, [row[0],row[1],row[2]])
    for i in range(3, 8):
        temp_str[i] = normalization[i-3][row[i]]
    test_data.append(temp_str)

random.shuffle(train_data)
TRAIN_SIZE = 250


X_train, X_test, y_train, y_test = ([row[:-1] for row in train_data[:TRAIN_SIZE]].copy(),
                                    [row[:-1] for row in train_data[TRAIN_SIZE:]].copy(),
                                    [row[-1] for row in train_data[:TRAIN_SIZE]].copy(),
                                    [row[-1] for row in train_data[TRAIN_SIZE:]].copy())

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_labels = len(set(y))

        if depth >= self.max_depth or n_labels == 1:
            return LeafNode(y)

        best_split = self._best_split(X, y)

        if best_split is None:
            return LeafNode(y)

        left_indices, right_indices, split_feature, split_value = best_split
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]

        left_subtree = self._grow_tree(left_X, left_y, depth + 1)
        right_subtree = self._grow_tree(right_X, right_y, depth + 1)

        return DecisionNode(split_feature, split_value, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None

        for feature_idx in range(len(X[0])):
            feature_values = set([row[feature_idx] for row in X])

            for value in feature_values:
                left_indices = [i for i, x in enumerate(X) if x[feature_idx] <= value]
                right_indices = [i for i, x in enumerate(X) if x[feature_idx] > value]

                gini = self._gini_index(y, left_indices, right_indices)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (left_indices, right_indices, feature_idx, value)

        return best_split

    def _gini_index(self, y, left_indices, right_indices):
        n_left = len(left_indices)
        n_right = len(right_indices)
        n_total = n_left + n_right

        left_classes = [y[i] for i in left_indices]
        right_classes = [y[i] for i in right_indices]

        p_left = sum(1 for c in left_classes if c != 0) / n_left if n_left != 0 else 0
        p_right = sum(1 for c in right_classes if c != 0) / n_right if n_right != 0 else 0

        gini_left = 1 - p_left ** 2 - (1 - p_left) ** 2
        gini_right = 1 - p_right ** 2 - (1 - p_right) ** 2

        gini_index = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

        return gini_index

    def predict(self, X):
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, node):
        if isinstance(node, LeafNode):
            return node.predicted_class
        if x[node.feature] <= node.value:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)


class LeafNode:
    def __init__(self, y):
        self.predicted_class = max(set(y), key=y.count)


class DecisionNode:
    def __init__(self, feature, value, left, right):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            bootstrap_X = [X[i] for i in indices]
            bootstrap_y = [y[i] for i in indices]
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return [max(set(x), key=x.count) for x in zip(*predictions)]

random_forest = RandomForest(n_estimators=10, max_depth=5)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)
validate_predictions = random_forest.predict(test_data)
accuracy = sum(1 for p, y_true in zip(predictions, y_test) if p == y_true) / len(y_test)
print("Accuracy:", accuracy)
validate_predictions = ['Да' if pred == 1 else 'Нет' for pred in validate_predictions]
with open('preds.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Контрольная 1', 'Контрольная 2', 'Контрольная 3', 'Сон накануне', 'Настроение', 'Энергетиков накануне', 'Посещаемость занятий', 'Время подготовки', 'Прогноз'])
    for student, prediction in zip(test_students, validate_predictions):
        csvwriter.writerow(student + [prediction])



        


