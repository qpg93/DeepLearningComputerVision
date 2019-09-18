# Decision Tree

## Concepts

* Entropy: incertainty of stochastic variable Y
* Conditional Entropy: incertainty of stochastic variable Y under the condition X
* Information Gain: information G brought to system by information F

## Main algorithms

### 1. Histroy

* ID3: Quinlan 1979
  * I: Iterative 循环 迭代
  * D: Dichotomiser 二分器
  * 3: 3rd generation of inductive learning 归纳学习
* C4.5: Quinlan 1993
* CART: Breiman 1984

### 2. Pros & Cons

#### 2.1 Pros

1. Easy to understand
2. Classification + Regression

#### 2.2 Cons

1. Discrete
2. Prone to overfit
3. NP-Complete (Greedy algorithm: local optimum)
4. Usually easy to choose features having more attributes

#### 2.3 Solutions

1. Prune # in leaf node
2. C4.5