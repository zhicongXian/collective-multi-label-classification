# Collective Multi-Label Classification

## Description

The basic idea of the algorithm is to utilize the principle of maximum information entropy, ensuring that the resulting 
distribution satisfies constraints, such as the expected correlation among the labels and the correlation between 
labels and features. A paper introducing this algorithm can be found at this [link](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1184&context=cs_faculty_pubs)

This constrained optimization problem can be solved using the Lagrange multiplier method. The problem is then transformed into an unconstrained optimization method by optimizing the Lagrange multipliers.

The optimal distribution that satisfies the data expectation constraints belongs to the Gibbs family distribution.

## Implementation

This is an improvement of the original implementation in [github](https://github.com/LouQiongdan/Collective-Multi-Label-Classifier).

This code enhances the training speed of the algorithm through code refactoring and the application of the scipy
optimization library.
