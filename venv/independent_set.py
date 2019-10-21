# # -*- coding: utf-8 -*-
#
# """
#     Find a maximum independent set of vertices in a graph.
# """
#
import cvxpy as cy
#import numpy as np


n=5
vertices = range(n)
# edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(2,7),(3,8),(4,9),(1,6)]    # a cycle on 5 vertices
edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(1,6),(2,7),(3,8),(4,9), (5,7), (7,9),(9,6), (6,8), (8,5) ]

#x=cy.Variable(n,k)  # continues variable
x=cy.Variable(n,boolean=True)  # binary variable
#x=cy.Int(n)  # integer variable


# setting the constraints
constraints = [x>=0]

for e in edges:
	constraints += [x[e[0]] + x[e[1]] <= 1]


# Form objective.
obj = cy.Maximize( sum(x[v] for v in vertices) )

# Form and solve problem.
prob = cy.Problem(obj, constraints)

#prob.solve()  # standard solver.
#prob.solve(solver=cy.GLPK)  # uses GLPK instead of the standard solver
prob.solve(solver=cy.GLPK_MI)  # uses GLPK_MI instead of the standard solver

print ("status:", prob.status)
print ("optimal value = ", prob.value)

print ("optimal var = \n", x.value)
print ('\n')

for v in vertices:
	if x[v].value==1:
		print( "Vertex %s is in the independent set." %(v+1))


#
# -*- coding: utf-8 -*-

"""
    Find a maximum independent set of vertices in a graph.
"""

import cvxpy as cy

# import numpy as np


n = 10
vertices = range(n)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8),
		 (8, 5)]

# x=cy.Variable(n,k)  # continues variable
x = cy.Variable(n,boolean=True)  # binary variable
# x=cy.Int(n)  # integer variable


# setting the constraints
constraints = [x >= 0]

for e in edges:
	constraints += [x[e[0]] + x[e[1]] <= 1]

# Form objective.
obj = cy.Maximize(sum(x[v] for v in vertices))

# Form and solve problem.
prob = cy.Problem(obj, constraints)

# prob.solve()  # standard solver.
# prob.solve(solver=cy.GLPK)  # uses GLPK instead of the standard solver
prob.solve(solver=cy.GLPK_MI)  # uses GLPK_MI instead of the standard solver

print("status:", prob.status)
print("optimal value = ", prob.value)
print("optimal var = \n", x.value)
print('\n')

for v in vertices:
	if x[v].value == 1:
		print("Vertex %s is in the independent set." % (v + 1))


