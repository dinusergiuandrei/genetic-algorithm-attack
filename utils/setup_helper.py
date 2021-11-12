def get_import_targets():
    packages = {
        'os': 'os',
        'sys': 'sys',
        'math':'math',
        'numpy': 'np',
        'pandas': 'pd',
        'seaborn': 'sns',
        'matplotlib.pyplot': 'plt',
        'itertools':'itertools'
    }
    functions = {
        'pprint': 'pprint',
        'scipy.stats': ['pearsonr', 'wilcoxon', 'pearsonr'],
        'scipy.stats.mstats': 'winsorize',
        'IPython.core.interactiveshell': 'InteractiveShell',
        'pathlib': 'Path',
    }
    function_aliases = {}

    variables = {
        'title_size': 20,
        'label_size': 18,
        'legend_size': 18,
        'wide_plot_shape': (15, 5),
        'square_plot_shape': (10, 10)
    }

    return packages, functions, function_aliases, variables


def import_all(g):
    import importlib

    packages, functions, function_aliases, variables = get_import_targets()

    for package, alias in packages.items():
        g[alias] = importlib.import_module(package)

    for function_path, function_name in functions.items():
        if isinstance(function_name, str):
            function_name = [function_name]
        for f in function_name:
            alias = function_aliases.get(f, f)
            g[alias] = getattr(importlib.import_module(function_path), f)

    for variable, value in variables.items():
        g[variable] = value


def setup(g):
    import_all(g)
    g['InteractiveShell'].ast_node_interactivity = 'all'
    g['plt'].style.use('ggplot')
    g['pd'].set_option('display.max_columns', None)
    g['plt'].rcParams['figure.figsize'] = (15, 5)
    g['plt'].rcParams['axes.titlesize'] = 24
    g['plt'].rcParams['axes.labelsize'] = 22
    g['plt'].rcParams['ytick.labelsize'] = 20
    g['plt'].rcParams['xtick.labelsize'] = 20
