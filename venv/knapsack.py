# -*- coding: utf-8 -*-
"""
    Solves a random knapsack instance
"""

import cvxpy as cy
import numpy as np
import time

np.set_printoptions(precision=3)   # print only a few decimals

np.random.seed(1)  # fix the random number generator

n=10   # number of items

x = cy.Variable(n,boolean=True)   # continues variable
# x=cy.Bool(n)  # binary variable
# x=cy.Int(n)  # integer variable

a = np.random.rand(n)  # random weights
c = np.random.rand(n)  # random costs

b = 0.25*n   # b is the capacity of the knapsack


### setting the constraints
constraints = [x>=0]

constraints += [a*x<=b]

### Form objective.
obj = cy.Maximize(c*x)

### Define the problem.
prob = cy.Problem(obj, constraints)

### Solve the problem

tic = time.time()  # start timing

# prob.solve()  # Solves the problem with the standard solver. Use this for convex quadratic problems. So not for ILPs

prob.solve(solver=cy.GLPK_MI)  # uses GLPK instead of the standard solver.

#prob.solve(solver=cy.GLPK_MI)  # uses GLPK_MI instead of the standard solver. Use this for (mixed) ILPs.

toc = time.time()  # stop timing
elapsed = toc-tic


print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal variables\n", x.value)
print("elapsed time:", "%.6f" % elapsed  )
