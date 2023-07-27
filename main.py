import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from augmentation import orthogonal_distortion_smote

target_job = 14
filepath = "OVRClassifier_" + str(target_job)

df = pd.read_csv("ITPersonality.csv")

# remove the discrete variables in feature

df = df.drop(columns=["大学専攻"])

# normalization on age, degree, school_level, experience, gender

max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
avg_std_scaler = lambda x: (x - np.mean(x)) / np.std(x)

sex_scaler = lambda x: (x - 1.5) * (-2.0)

df["年齢"] = df[["年齢"]].apply(max_min_scaler)
df["性別"] = df[["性別"]].apply(sex_scaler)
df["学歴"] = df[["学歴"]].apply(max_min_scaler)
df["出身校ランキング"] = df[["出身校ランキング"]].apply(max_min_scaler)
df["IT経験総年数"] = df[["IT経験総年数"]].apply(max_min_scaler)

column_names = df.columns.tolist()

column_names_part1 = column_names[15:18]
column_names_part2 = column_names[-2:]
column_names_part3 = column_names[18:-2]

final_column_names = ["label"] + column_names_part1 + column_names_part2 + column_names_part3
print(final_column_names)

positive_num = 0
negative_num = 0
normalized_samples = []

# normalize self-evaluation

for index, row in df.iterrows():
    info = row.tolist()
    target = info[0:15]
    age_sex_degree = info[15:18]
    school_experience = info[-2:]
    self_evaluation = info[18:-2]

    normalized_vector = age_sex_degree + school_experience

    self_evaluation = np.array(self_evaluation, dtype=float)

    avg = np.mean(self_evaluation)
    std = np.std(self_evaluation)

    for i in range(len(self_evaluation)):
        self_evaluation[i] = (self_evaluation[i] - avg) / std

    label = target[target_job-1]

    if label == 1.0:
        positive_num += 1
    else:
        negative_num += 1

    normalized_samples.append([label] + normalized_vector + self_evaluation.tolist())

if positive_num > 1 and negative_num > 1:
    smoted_dataset = orthogonal_distortion_smote(normalized_samples)
    samples_per_class = int(len(smoted_dataset) / 2)
    labels = []
    for i in range(samples_per_class):
        labels.append(1)
    for j in range(samples_per_class):
        labels.append(0)
    # remove NaN
    filtered_dataset = []
    filtered_labels = []
    for index in range(len(smoted_dataset)):
        NaNcheck = np.argwhere(np.isnan(smoted_dataset[index]))
        if len(NaNcheck) == 0:
            filtered_dataset.append(smoted_dataset[index])
            filtered_labels.append(labels[index])
    clf = LogisticRegression(random_state=0).fit(filtered_dataset, filtered_labels)
    print(clf.predict(filtered_dataset))
    print(clf.predict_proba(filtered_dataset))
    # output the smoted dataset
    output_buffer = []
    for n in range(len(filtered_labels)):
        output_buffer.append([filtered_labels[n]] + filtered_dataset[n])
    output_df = pd.DataFrame(output_buffer, columns=final_column_names)
    output_df.to_csv("./smoted/augmented_" + str(target_job)+".csv", encoding="utf-8-sig")
else:
    print("too few positive samples")