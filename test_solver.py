from pyosolver import PYOSolver


solver = PYOSolver("J:\\PioSOLVER", "PioSOLVER2-edge")
solver.load_tree("J:\\PioSOLVER\\saves\\hu_preflop.cfr")
print(solver.show_node("r:0"))
print(solver.show_children_actions("r:0"))
#print(solver.show_range("IP","r:0"))