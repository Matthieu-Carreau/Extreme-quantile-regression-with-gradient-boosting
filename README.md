# Extreme-quantile-regression-with-gradient-boosting

This repository contains our report and code for a project made with Naël Farhan for the course Statistical Learning With Extreme Values (https://helios2.mi.parisdescartes.fr/~asabouri/extremes-cours-MVA/extreme-course-MVA.html) of the MVA master.

The project is based on the following article:

Velthoen, J., Dombry, C., Cai, JJ. et al. Gradient boosting for extreme quantile regression. Extremes
(2023).

It presents a method for estimating the conditional quantiles at extreme levels of a random variable $Y \in \mathbb{R}$ given a vector of covariates $X \in \mathbb{R}^n$.
The method relies on the estimation of parameters of a Generalized Pareto Distribution that model the tail of the distribution of $Y$ conditionally to $X$.

Our report contains a summary of the article as well as results of our experiments with our implementation of the algorithms.
