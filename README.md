# cheml
Chemical engineering ML
## Overview

**cheml** is a toolkit for building, training, and deploying machine learning models tailored for chemical engineering applications. It provides utilities for data preprocessing, model selection, evaluation, and integration with chemical process data.

## Features

- Data loaders and preprocessors for chemical datasets
- Ready-to-use ML model templates (regression, classification)
- Model evaluation and visualization tools
- Support for scikit-learn, PyTorch, and TensorFlow
- Example workflows for common chemical engineering problems

## Installation

```bash
git clone https://github.com/mv-per/cheml.git
cd cheml
```

# Environment Setup

CheML uses conda for development. Please refer to [conda documentation](https://www.anaconda.com/blog/getting-started-with-conda-environments) for conda installation. You can also use a conda alternative, such as the [mamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) package, that is a C++ implementation of conda.

With conda setup, you can generate the environment by

```bash
chmod +x ./scripts/create_dev_env.sh
./scripts/create_dev_env.sh
```

then you can activate the environment by invoking:

```
conda activate cheml
```

The package uses [pre-commit](https://pre-commit.com/) to lint and check typing on the files. Please initiate pre-commit on your machine by invoking:

```
pre-commit install
```

With the activated environment, one can build the package using the following command (that uses the [invoke package](https://www.pyinvoke.org/)):
```
inv build
```

## Example Applications

- Predicting reaction yields
- Process optimization
- Property estimation

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

MIT License
