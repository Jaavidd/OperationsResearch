

import cvxpy as cy

# import numpy as np


n = 10
colors=3
vertices = range(n)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (7, 9), (9, 6), (6, 8),
		 (8, 5)]

x=cy.Variable(n,integer=True)
# continues variable
x = cy.Variable((colors,n),boolean=True)  # binary variable
# x=cy.Int(n)  # integer variable


# setting the constraints
constraints = [x >= 0]
#
# for e in edges:
# 	constraints += [x[e[0]] - x[e[1]] >= 1]

for e in vertices:
    constraints+= [cy.sum(x[:,e])==1]

for e in edges:
    for i in range(colors):
        constraints+= [x[i,e[0]]+x[i,e[1]]<= 1 ]


# Form objective.
# obj = cy.Maximize(sum(x[v] for v in vertices))
obj=cy.Maximize(0)
# Form and solve problem.
prob = cy.Problem(obj, constraints)

# prob.solve()  # standard solver.
# prob.solve(solver=cy.GLPK)  # uses GLPK instead of the standard solver
prob.solve(solver=cy.GLPK_MI)  # uses GLPK_MI instead of the standard solver

print("status:", prob.status)
print("optimal value = ", prob.value)
print("optimal var = \n", x.value)
print('\n')

# for v in vertices:
# 	if x[v].value == 1:
# 		print("Vertex %s is in the independent set." % (v + 1))

for v in vertices:
    for c in range(colors):
        if x[c,v].value==1:
            print("Vertex %s is in the independent set." % (v + 1))

