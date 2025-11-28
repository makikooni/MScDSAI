import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Graphs""")
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


@app.cell(disabled=True)
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


@app.cell(disabled=True)
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
def _():
    # your code here
    A = []
    return (A,)


@app.cell(disabled=True)
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


@app.cell(disabled=True)
def _(G_2, nx, plt, pos):
    nx.draw(G_2, with_labels=True, pos=pos)
    plt.show()
    return


@app.cell
def _():
    #Your code here
    degrees = []

    return (degrees,)


@app.cell(disabled=True)
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
    #Your code here
    closeness=[]
    return (closeness,)


@app.cell(disabled=True)
def _(G_2, closeness, nx, plt, pos):
    nx.draw(G_2, cmap=plt.get_cmap('YlOrRd'), pos=pos, node_color=closeness, with_labels=True, font_color='white')
    plt.show()
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
def _():
    #Your code here
    betweenness = []
    return (betweenness,)


@app.cell(disabled=True)
def _(G_2, betweenness, nx, plt, pos):
    nx.draw(G_2, cmap=plt.get_cmap('YlOrRd'), pos=pos, node_color=betweenness, with_labels=True, font_color='white')
    plt.show()
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
def _():
    #Your code here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
