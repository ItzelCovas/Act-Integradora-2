from graphviz import Digraph

g = Digraph()
g.node("A")
g.node("B")
g.edge("A", "B")
g.render("test_graph", format="png", cleanup=True)
print("Renderización OK → test_graph.png creado")