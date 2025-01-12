# Lontras

<img src="https://raw.githubusercontent.com/luxedo/lontras/refs/heads/main/docs/_static/lontra.png" height=400 alt="Lontras Logo"/>

[![PyPI - Version](https://img.shields.io/pypi/v/lontras.svg)](https://pypi.org/project/lontras)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lontras.svg)](https://pypi.org/project/lontras)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/luxedo/lontras/publish.yml)
![Codecov](https://img.shields.io/codecov/c/github/luxedo/lontras)
![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/luxedo/lontras)

---

> ### ⚠️ This library is under development and has not been released yet

> We love `pandas` and its siblings! They're the industry-standard tools for powerful data
> manipulation. However, for smaller projects or when minimizing dependencies is paramount,
> `lontras` offers a lightweight, pure-Python alternative built on simple dictionaries. Designed
> for ease of use and direct integration, lontras encourages you to copy and paste its core
> components into your code. It's the perfect stand mixer for baking smaller data tasks when
> bringing out the full industrial equipment of `pandas` is too much.
>
> ### TLDR;
>
> - Small-scale `DataFrame` operations? `lontras`!
> - Fast `DataFrame` loading, no dependencies? `lontras`!
> - Embed `DataFrames` in [MicroPython](https://micropython.org/)? `lontras`!
> - Use `DataFrames` in the browser ([PyScript](https://pyscript.net/))? `lontras`!
> - High [Pandas](https://pandas.pydata.org/) compatibility? `lontras`!

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Core Functionalities](#core-functionalities)
- [License](#license)

## Installation

If you prefer to install lontras using pip for easy management and updates, you can do so with the
following command:

```console
pip install lontras
# or
uv pip install lontras
```

This will download and install lontras from the Python Package Index (PyPI).

For lightweight projects where you want to avoid external dependencies, you can simply copy the
source file [src/lontras/lontras.py](src/lontras/lontras.py) into your project directory. This
allows you to directly use the library functions from your code without any installation.

## Documentation

Check out the API Docs at: ...

## Core Functionalities:

Lontras prioritizes simplicity and minimal dependencies. It leverages Python's native dictionaries
(via [UserDict](https://docs.python.org/3/library/collections.html#collections.UserDict)) to offer
core `DataFrame` and `Series` functionalities without external libraries.

### Data Structures:

- `Series`: A one-dimensional array-like structure.
- `DataFrame`: A two-dimensional labeled data structure.

### Accessing Data:

- `loc` and `iloc`: Access data by label or by index.
- Label-based access: Access data using standard dictionary-like syntax (e.g., series['label']).
- Positional access (slicing): Use slices for location-based access (e.g., series[1:3]).
- Attribute-based access: For convenient access to all values for a given key, use attribute-based access using **getattr** that dynamically retrieves data based on provided keys or a list of keys.

### Modifying Data:

- Setting values: Modify existing values or add new entries using `loc` or `iloc` assignment (e.g., series.loc['label'] = value).
- Deleting values: Remove entries using del series['label'].
- Concatenation: Combine Series or DataFrames vertically or horizontally.

### Transforming Data:

- Mapping and applying functions: Apply functions element-wise using map or along axes/indices using apply.
- Sorting: Sort indexes and values using provided sorting functions.
- Basic operations: Use standard Python operators (+, -, \*, /, //, %, \*\*, comparisons) for element-wise operations.

### Data Aggregation and Combination:

- groupby: Group data based on a column and perform operations within each group (similar to pandas groupby).
- Join/Merge: Merge two DataFrames based on specific columns (similar to pandas join/merge operations).
- reduce: Apply a function cumulatively to the elements. Basic reduction functions like max, min, argmax, sum, etc.
- Leverages Python's built-in statistics module for basic statistical calculations.

### Limitations & Trade-offs: (Same as before)

- Specialized Data Handling: Lontras focuses on core functionalities and doesn't include specialized functions for datetime, strings, or categorical data. However, users can achieve similar behavior through apply and map functions.
- Statistical Functions: Limited set of statistical functions. Lontras relies primarily on Python's built-in statistics module.
- Data Import/Export: Supports limited import/export formats. External libraries might be necessary for complex file handling.
- Missing Data Handling: Doesn't include dedicated functions for handling missing data. Users can implement their own logic using conditional statements or filtering.
- Multilevel Indexing: Lacks built-in support for multilevel indexing. However, tuple indexes can be used to achieve similar hierarchical structures.
- Plotting: Currently doesn't include plotting functionalities. External plotting libraries are recommended for visualization.

## License

`lontras` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
