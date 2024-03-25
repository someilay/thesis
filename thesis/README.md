# Structure

* $\textbf{Introduction.}$ This chapter contains a brief intro to the topic with an explanation of why this research can be useful. Also, this section describes the structure of the thesis.
---
* $\textbf{Literature review.}$ Here will be a review of previous research. Maybe it will be nice to include an orientation difference problem. 
    * Add reference to Sliding on Manifolds: Geometric Attitude Control with Quaternions
    * Add references on comparison SO(3), SE(3) errors, closed loops control
    * Closed chains constraints on Kenguru
---
* $\textbf{Methodology.}$ This section contains a deep explanation of how I achieved the results. It explains the Udwadia-Kalaba approach, and how it can be applied to the problem. Maybe it would contain proof of stability.
    * Describe a class of physical systems (rigid body, KKT, open and closed loop, derivation)
    * Transition to Udwadia-Kalaba approach (article Верещагина)
    * Baumgarte stabiliztion, Why is it needed?
    * What is rigid-body constraint? SE(3), SO(3), S(3). Define differentional equation. Proof or reference.
    * Priotization for several constraints
---
* $\textbf{Implementation.}$ In this chapter the applied frameworks are demonstrated. This chapter describes how the 
suggested method can be implemented by using Pinocchio, Mujoco, 
CvxPy and e.t.c. The optimization hints can be included.

---
* $\textbf{Evaluation and Discussion.}$ The different plots and 
tables is placed here. These figures may contain error convergence, computation speed, or something else.
---
* $\textbf{Conclusion.}$ This section contains a discussion about the achieved results. It may contain further work suggestions.
