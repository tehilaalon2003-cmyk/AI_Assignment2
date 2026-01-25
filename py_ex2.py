from collections import Counter
from io_utils import read_tsv_dataset, write_output
from naive_bayes import train_naive_bayes, predict_naive_bayes
from metrics import accuracy


# Load train and test datasets from files
train = read_tsv_dataset("train.txt")
test = read_tsv_dataset("test.txt")

# Feature names are all columns except the last one (class)
feature_names = train.header[:-1]

# Train Naive Bayes on the training set
nb_model = train_naive_bayes(train.X, train.y, feature_names)

# Predict Naive Bayes labels for the test set
nb_predictions = [
    predict_naive_bayes(nb_model, sample)
    for sample in test.X
]

# Calculate Naive Bayes accuracy (test is labeled)
nb_acc = accuracy(test.y, nb_predictions)

# Temporary Decision Tree baseline: predict the majority class from train
majority_class = Counter(train.y).most_common(1)[0][0]
dt_predictions = [majority_class] * len(test.X)

# Calculate Decision Tree (baseline) accuracy
dt_acc = accuracy(test.y, dt_predictions)

# Write final results to output.txt in the required format
write_output("output.txt", dt_predictions, nb_predictions, dt_acc, nb_acc)

print("output.txt created successfully")
print("DT accuracy:", dt_acc)
print("NB accuracy:", nb_acc)
