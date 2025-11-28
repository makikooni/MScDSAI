import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Graphs (solution)""")
    return


@app.cell
def _():
    import networkx as nx, numpy, matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 600
    return numpy, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 1
    Modify ```edge_list.txt```,```adjacency_list.txt``` to obtain the given graph
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 1. Adjacency list:""")
    return


@app.cell
def _(nx):
    _fh = open('adjacency_list.txt', 'rb')
    G = nx.read_adjlist(_fh)
    return (G,)


@app.cell
def _(G, nx, plt):
    nx.draw(G,with_labels = True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 2. Edge list:""")
    return


@app.cell
def _(nx):
    _fh = open('edge_list.txt', 'rb')
    H = nx.read_edgelist(_fh)
    _fh.close()
    return (H,)


@app.cell
def _(H, nx, plt):
    nx.draw(H,with_labels = True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### 3. Adjacency Matrix""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now create a 2D numpy.array that defines the adjacency matrix""")
    return


@app.cell
def _(numpy):
    # your code here
    A = numpy.array([[0, 1, 0, 1, 0], [1, 0, 1, 1, 0], [0, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 0, 1, 1, 0]])
    return (A,)


@app.cell
def _(A, nx, plt):
    G_1 = nx.from_numpy_array(A)
    nx.draw(G_1, with_labels=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""-------""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""-----""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 2: Degree centrality
    For the following adjacency matrix, calculate the degree centrality for every node and then plot the graph with nodes coloured according to centrality value.
    """
    )
    return


@app.cell
def _():
    adj = [[0, 1, 0, 0, 0, 0, 0, 1, 0],
               [1, 0, 1, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 1, 0, 1, 0, 0, 1],
               [0, 0, 1, 0, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 1, 0, 0, 0],
               [0, 0, 1, 1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 1, 1],
               [1, 1, 0, 0, 0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0, 0, 1, 1, 0]
               ]
    return (adj,)


@app.cell
def _(adj, numpy, nx):
    G_2 = nx.from_numpy_array(numpy.array(adj))
    pos = nx.spring_layout(G_2)
    return G_2, pos


@app.cell
def _(G_2, nx, plt, pos):
    nx.draw(G_2, with_labels=True, pos=pos)
    plt.show()
    return


@app.cell
def _(adj):
    degrees = []
    for _i in adj:
        degrees.append(sum(_i))
    return (degrees,)


@app.cell
def _(degrees):
    degrees
    return


@app.cell
def _(G_2, degrees, nx, plt, pos):
    nx.draw(G_2, cmap=plt.get_cmap('YlOrRd'), pos=pos, node_color=degrees, with_labels=True, font_color='white')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 3: Closeness centrality
    Now calculate closeness centrality for each node and colour plot the nodes accordingly
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(G_2, closeness, nx, plt, pos):
    nx.draw(G_2, cmap=plt.get_cmap('YlOrRd'), pos=pos, node_color=closeness, with_labels=True, font_color='white')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 4: Betweenness centrality
    Now calculate betweenness centrality for each node and colour plot the nodes accordingly
    """
    )
    return


@app.cell
def _(numpy, nx):
    betweenness = []

    def betweennessAll(adj):
        allNodes = []
        for _i in range(len(adj)):
            allNodes.append(betweennessNode(_i, adj))
        return allNodes

    def betweennessNode(i, adj):
        G = nx.from_numpy_array(numpy.array(adj))
        total = 0
        for j in range(len(adj)):
            for k in range(j):
                allPaths = [p for p in nx.all_shortest_paths(G, source=k, target=j)]
                allPaths_i = []
                for path in allPaths:
                    if i in path and path[0] != i and (path[len(path) - 1] != i):
                        allPaths_i.append(path)
                if len(allPaths) > 0:
                    total = total + len(allPaths_i) / len(allPaths)
        return total
    return (betweennessAll,)


@app.cell
def _(adj, betweennessAll):
    betweennessAll(adj)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's check our result using networkx""")
    return


@app.cell
def _(adj, numpy, nx):
    G_3 = nx.from_numpy_array(numpy.array(adj))
    nx.betweenness_centrality(G_3, normalized=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 5
    Using numpy arrays, run 10 iteratons using equation (i) and find the eigenvector centrality for each node of the graph
    """
    )
    return


@app.cell
def _(numpy):
    A_1 = [[0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 1, 0]]
    x = [1, 1, 1, 1, 1, 1, 1, 1]
    A_1 = numpy.array(A_1)
    x = numpy.array(x)
    for _i in range(10):
        x = A_1.dot(x)
        x = x / max(x)
    print(x)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
