import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import sklearn as sk
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    print(data.data.shape)  # Affiche les dimensions des données (569 échantillons, 30 caractéristiques)
    print(data.target.shape)  # Affiche les dimensions des cibles (569 échantillons)
    return data, load_breast_cancer


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    data.keys()
    return


@app.cell
def _(data):
    print(data['target_names'])
    print(data['feature_names'])
    print((data['data']).shape)
    return


@app.cell
def _(load_breast_cancer):

    df = load_breast_cancer(as_frame=True)
    print(df.frame.head())  # Affiche les 5 premières lignes du DataFrame
    return (df,)


@app.cell
def _(df):
    df_cancer = df.frame
    df_cancer.head()
    return (df_cancer,)


@app.cell
def _(df_cancer):
    df_cancer.tail()
    return


@app.cell
def _(df_cancer):
    df_cancer.describe()
    return


@app.cell
def _(df_cancer):
    grouped_stats = df_cancer.groupby('target').describe()
    grouped_stats
    return


@app.cell
def _(df_cancer):
    _df = df_cancer # just in this cell 
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],['target']))
    # not neccecary cell 

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    https://medium.com/@kaushiksimran827/breast-cancer-classification-a-simple-guide-with-scikit-learn-and-support-vector-machine-svm-47a790412edf

    """
    )
    return


@app.cell
def _(df_cancer, sns):
    sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
    return


@app.cell
def _(df_cancer):
    print(df_cancer.dtypes)
    return


@app.cell
def _():
    return


@app.cell
def _(df_cancer, plt, sns):

    # df_cancer est ton DataFrame
    sns.countplot(x='target', data=df_cancer, label="Count")

    plt.xlabel("Target")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Répartition des classes dans le dataset Breast Cancer")
    plt.legend()
    plt.show() 
    return


if __name__ == "__main__":
    app.run()
