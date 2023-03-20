import numpy as np

data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes'],
])


def candidate_elimination(data):
    specific_hypothesis = data[0][:-1]
    general_hypothesis = [['?' for i in range(
        len(specific_hypothesis))] for j in range(len(specific_hypothesis))]

    for i in range(len(data)):
        if data[i][-1] == "Yes":
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] != data[i][j]:
                    specific_hypothesis[j] = '?'
                    general_hypothesis[j][j] = '?'
        else:
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] != data[i][j]:
                    general_hypothesis[j][j] = specific_hypothesis[j]
                else:
                    general_hypothesis[j][j] = '?'

    indices = [i for i, val in enumerate(
        general_hypothesis) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_hypothesis.remove(['?', '?', '?', '?', '?', '?'])

    return specific_hypothesis, general_hypothesis


s, g = candidate_elimination(data)
print(s, g)
