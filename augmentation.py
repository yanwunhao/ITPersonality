import sys
import random

import numpy as np

alpha = 0.1


def gram_schmidt(vectors):
    num_vectors = len(vectors)
    ortho_vectors = np.zeros_like(vectors)

    for i in range(num_vectors):
        ortho_vectors[i] = vectors[i]
        for j in range(i):
            projection = np.dot(vectors[i], ortho_vectors[j]) / np.dot(ortho_vectors[j], ortho_vectors[j])
            ortho_vectors[i] -= projection * ortho_vectors[j]

        # Normalize the vector to make it unit length (optional)
        ortho_vectors[i] /= np.linalg.norm(ortho_vectors[i])

    return ortho_vectors


def orthogonal_distortion_smote(data):
    positive_samples = []
    negative_samples = []
    for sample in data:
        sample_np = np.array(sample, dtype=float)
        NaNcheck = np.argwhere(np.isnan(sample_np))
        if len(NaNcheck) != 0:
            continue
        if sample[0] == 1.0:
            positive_samples.append(sample[1:])
        else:
            negative_samples.append(sample[1:])

    if len(positive_samples) < len(negative_samples):
        minority = positive_samples
        majority = negative_samples
        augmentation_label = 1
    else:
        minority = negative_samples
        majority = positive_samples
        augmentation_label = 0

    if len(minority) > 1:
        majority_np = np.array(majority, dtype=float)
        majority_center = np.average(majority_np, axis=0)

        while len(minority) < len(majority):
            prototype_index = random.sample(range(0, len(minority)), 2)
            prototype_a = minority[prototype_index[0]]
            prototype_b = minority[prototype_index[1]]
            prototype_a = np.array(prototype_a)
            prototype_b = np.array(prototype_b)
            new_sample = (prototype_a + prototype_b) / 2.0

            # orthogonal distortion
            distance = new_sample - majority_center

            available_orthogonal_distortions = gram_schmidt(np.array([distance]))
            available_orthogonal_distortions.reshape((1,31))

            if available_orthogonal_distortions.shape[0] > 0:
                # distortion_index = random.sample(range(0, available_orthogonal_distortions.shape[0]), 1)
                distortion = available_orthogonal_distortions[0]
                new_sample_with_distortion = new_sample + alpha * distortion


            minority.append(new_sample_with_distortion.tolist())
        if augmentation_label == 1:
            positive_samples = minority
            negative_samples = majority
        else:
            positive_samples = majority
            negative_samples = minority
        smoted_dataset = positive_samples + negative_samples
        return smoted_dataset
    else:
        sys.exit(1)
