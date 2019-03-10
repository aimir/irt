# IRT

## Item Response Theory in Python

Currently contains simple code, using a 4-parameter model, and allowing for partial credit.

The parameter estimation is done using MMLE with parameter regulation, and the underlying optimization uses `scipy.optimize`

estimate_thetas receives an input array, where each line represents the scores of a single person in each question, and returns the estimated theta parameters per person and the model parameters per question.
