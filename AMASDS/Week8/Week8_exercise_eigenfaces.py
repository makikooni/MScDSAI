import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise week 9: Eigenfaces

    Application of PCA to data compression and facial recognition
    """)
    return


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot as plt
    import numpy as np
    from pathlib import Path
    return Path, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download the dataset

    Download the LFWcrop Face Dataset greyscale dataset from:
    https://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip (49 MB)

    Extract the files to the same directory as the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _(Path, np, plt):
    def get_files():
        path = Path('lfwcrop_grey/faces/')
        items = sorted(path.glob('*'))  # Get all items
        files = [i for i in items if i.is_file()]  # Filter for files
        return np.array(files)

    def test_train(num_train, num_test):
        num_images = num_train + num_test

        files = get_files()
        indices_files = np.arange(len(files), dtype=int)  # Indices of all files

        rng = np.random.default_rng(seed=0)
        selection = rng.choice(indices_files, size=num_images, replace=False, shuffle=True)

        # Split into train and test files
        files_train = files[selection[:num_train]]
        files_test = files[selection[num_train:]]

        return files_train, files_test

    def read_images(files):
        return np.array([plt.imread(f) for f in files], dtype=np.float64)

    def get_names(files):
        return [f.name[:f.name.find('0')-1].replace("_", " ") for f in files]

    def show_images(images, titles=None, n_row=4, n_col=4, h=64, w=64):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            if titles is not None:
                plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())
    return get_names, read_images, show_images, test_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and test data
    """)
    return


@app.cell
def _(get_names, read_images, show_images, test_train):
    # Number of train and test images
    num_train = 1000
    num_test = 16

    files_train, files_test = test_train(num_train, num_test)

    # Training data
    images_train = read_images(files_train)
    names_train = get_names(files_train)

    # Test data
    images_test = read_images(files_test)
    names_test = get_names(files_test)

    # Visualise some of the training images
    show_images(images_train, names_train, n_row=4, n_col=4)

    # Convert images to column vectors
    data_train = images_train.reshape(num_train, -1)
    data_test = images_test.reshape(num_train, -1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Implement PCA

    Implement principal component analysis, using one of the two approaches:
    - eigendecomposition or
    - singular value decomposition

    The Numpy package provides the necessary functions:
    - `np.linalg.eig()` https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    - `np.linalg.svd()` https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

    Follow the instructions in the slides.

    Apply PCA to the train set to obtain the eigenvectors (eigenfaces), the eigenvalues and the mean image (used to centre the dataset)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Visualise the Eigenfaces

    Use the function `show_images` to plot the eigenvectors (principal axes) computed with PCA as images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Explained variance

    How many eigenvectors explain at least (A) 50 %, (B) 70 %, (C) 90 % of the variance?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task: Compress and reconstruct

    With the test set, evaluate the reconstruction quality for different numbers of principal components.

    Use the principal components that explain (A) 50 %, (B) 70 %, (C) 90 % of the variance to
    1. Project the test data onto the principal axes
    2. Reconstruct the test images from the projected test data (from step 1)
    3. Visualise the reconstructed test images (from step 2)

    For this, you need to use the same means from the training data used in PCA. The means need to be subtracted from the test data for projection and added back for reconstruction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: Facial recognition

    We can use the eigenfaces as predictors. For example, to find the training example that is most similar to a new image.

    For this, use a new image that is not in the train or test set. You may upload an image or select one from the dataset folder. The image needs to be greyscale and 64 × 64 pixels. To turn the image into a Numpy array, you can use the helper function `read_images()`, which takes as input a list of files.

    Compare the projection onto the eigenvectors of the new image to the eigenfaces in terms of similarity.

    Are there any instance-based classifiers that you can use for this purpose?
    """)
    return


if __name__ == "__main__":
    app.run()
