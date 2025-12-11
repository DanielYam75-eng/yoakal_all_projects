# RE-model
## Introduction
Historically, at least 33% of the expenditure originates in contracts between the MOD and different vendors for services and goods. These expenditures show up in the IIT table mostly as RE invoices, together with ZY invoices (which represent price changes due to linkage to an index) and ZF invoices (which represent fines due to delayed delivery). In addition to their relative size, this type of expenditure tends to be volatile and highly dependent on real-world context, as opposed to expenditures related directly to salaried workers, which tend to follow a stable overarching trend linked only to the size of the workforce.

On the positive side, the MM, FM and FI modules in the MODEL contain rich documented context that we believe may help forecast these expenditures relatively accurately, provided we can build a probabilistic model that connects this context to the actual expenditures.

Our main object of concern is the PO, which represents an actual contract between the MOD and a vendor. As such, it has many useful attributes, such as the procurement organization (the specific MOD subunit that created the contract). Invoices-which we aim to predict-are submitted *against* POs. For example, you might have a PO with ID 4444 and a total value of 10,000 NIS. In the first month, an invoice of 1,000 NIS is submitted; in the second month, an invoice of 2,000 NIS is submitted. At that point, the total expense is 3,000 NIS, and the PO balance is 7,000 NIS. Our task is to predict the 3,000 NIS expense (for a given time frame).

Another complexity arises from the fact that new POs are created continuously over time, so at inference time we do not know the full set of POs for which we should predict expenses within the required time frame.

## Methods
Due to the complexity of the task, the repository contains two independent machine-learning tools: the -Augmenter,- which uses a Naive Bayes generative model to simulate future POs, and the -Predictor,- which uses XGBoost Regressor mode to forecast the monthly expense of each PO individually. Details follow.

### Preprocessing
All tables have a multi-index composed of three columns: the document ID (string), the document item (string), and the fund year (integer). The tables are eventually transformed into a format where each row corresponds to a PO plus its age, and various features are engineered. The full set of features available for training and inference is:

- **integer_features**: `age`, `N`
- **floating_features**: `po_net_value`, `cumulative_portion`
- **categorical_features**: `po_type`, `fingroup`, `huka`, `procurement_organization`, `expenditure_type`, `quarter`

### The Augmenter
We specify a set of basic PO attributes from which all features used by the Predictor can be derived. From the Predictor-s perspective, these attributes fully specify the PO. We generate POs as follows: given the total PO net value in some month and fund year, we sample individual PO net values from a lognormal distribution fitted to the data. This distribution is appropriate for financial data, and it fits actual data well. After that, we use a Naive Bayes model that predicts PO net value from the attributes, but in generation mode we invert it and sample attributes given a PO net value. We repeat this process until the full required set of POs is generated.

### The Predictor
The basic unit is a PO together with its age (in months). We aim to predict the amount of invoices submitted in that specific month. We build a table whose columns are the features plus the target, which is the amount of invoices divided by the PO net value. We fit a random forest regressor and use it to predict the normalized invoice amount for each PO at each age.

### Sampling
The dataset is very large, and training on all data often results in crashes. Therefore, we use a sampling fraction. Even a small fraction such as `1e-3` still provides enough data for meaningful training.

### Smoothing
Monthly invoice sequences for a PO are intermittent and difficult to predict. Instead of specialized methods like Croston-s, we smooth the labels using a window size `smoothing_window`. We group PO ages into consecutive windows and replace each value with the group-s average.

## User Guide
### Synopsis
```
usage: re-forecast [-h] -o OUTPUT_PATH [-c CONFIG] [-m MODEL] [--fine] [--debug] [--time] [--monthly]

The training/inference tool for RE forecast
```



Options:
- `-h, --help`         Show help message and exit  
- `-o OUTPUT_PATH`     Output path (mode-dependent)  
- `-c CONFIG`          Path to configuration file  
- `-m MODEL`           Path to model file  
- `--fine`             Use fine-grained output format  
- `--debug`            Log to debug experiment and emit extra artifacts  
- `--time`             Print execution time
- `--monthly`          Use monthly output format

### Modes
The program may work in three different modes: train mode, infer mode, and mixed mode. They are invoked in the configuration file,
and their basic functionality is as following: train mode trains a model and outputs it as an artifact, infer mode gets a model as input (via the -m
option) and outputs a csv file of the results, and mixed mode do both training and inference and doesn't output intermediate artifacts (i.e. the model).

### Options explanation
#### General options
| Option         | Description |
|----------------|-------------|
| `-h, --help`   | Prints simple synopsis and exits. |
| `-c, --config` | Path to configuration file. The configuration file decides, among other things, which mode the program runs.<br><br>The default value is `default.conf`, and the program attempts to find the configuration file in this order:<br>1. the given path in CLI<br>2. the default `default.conf` in the working directory<br>3. the program interactively asks the user what parameters to use |
| `--debug`      | The program logs in a dedicated mlflow experiment `re_forecast_debug` (regardless of experiment name in config), and outputs additional artifacts to the local system and mlflow. |
| `--time`       | The program times its execution and prints the results to stdout. |

#### Train mode option
| Option         | Description |
|----------------|-------------|
| `-o, --output` | Path where the trained model will be written to as a `.pkl` binary. |

#### Infer mode option
| Option         | Description |
|----------------|-------------|
| `-o, --output` | Path where the inference will be written to as a CSV file. The exact format depends on the `--fine` option. |
| `-m, --model`  | Path of the model `.pkl` binary file used for inference. The default is `model.pkl` in the current working directory. |
| `--fine`       | If set, the result CSV is a table with a row for each PO (real or generated). Otherwise, the result contains one line for all MOF classes with the total forecast. |
| `--monthly`    | If set, the result CSV is a table with columns for each forecasted month plus a column for the beginning of the year in cases `forecast_to` is 0. Otherwise, the result contains one column for the forecast. |

#### Mixed mode options
Mixed mode uses --output and --fine like infer mode, but doesn't use --model, as the model is trained during the program execution.


### Configuration Guide

The configuration is a simple text file that supplies the program with required information like: where the input dataset reside on the bucket,
which parameters to use for the model, which seed to use, what experiment to log the run in, how to augment the future POs, etc. The configuration file
format allow both inline and all-line comments using hashtags (#), and the options are supplied using key, value pairs or a special augmentation table format
for the augmentation dict. The configuration file ignores unknown keys silently, and if some required keys are missing, the user will be prompted interactively
to enter their values. If the augmentation table is missing, the program assumes there is no augmentation, and outputs it as info line to the stdout.

#### Configuration Format
The configuration is a simple text file with the following format: each row is either a key, value pair in the format
```
key: format
```
or a table row (for augmentation dict) in the format:
```
financial_year | month | fund_year | amount
```

Comments are allowed both inline and in dedicated lines using hashtags (#).

*Do not* use quotation marks to define a string.

#### Examples

```
# This is a typical example
experiment: check-max-depth
orders: oct-po # Comments are also allowed inline like that.
invoices: oct-invoices
order_edits: oct-changes
orders_dates: oct-dates
curr_year: 2025
curr_month: 9
forecast_to: 1
sample_frac: 1e-3
n_estimators: 1000
max_depth: 4
learning_rate: 0.8
mode: infer
2025 | 10 | 2025 | 3343136828.56
2025 | 11 | 2025 | 3343136828.56
2025 | 12 | 2025 | 3343136828.56
2026 | 1  | 2026 | 2093637309.18
2026 | 2  | 2026 | 1805975976.31
2026 | 3  | 2026 | 4262730832.64
2026 | 4  | 2026 | 2760625628.82
2026 | 5  | 2026 | 4126391997.05
2026 | 6  | 2026 | 5280905965.33
2026 | 7  | 2026 | 3167107311.76
2026 | 8  | 2026 | 3126041425.00
2026 | 9  | 2026 | 3464815010.94
2026 | 10 | 2026 | 3343136828.56
2026 | 11 | 2026 | 3343136828.56
2026 | 12 | 2026 | 3343136828.56
smoothing_window: 4
```

--------------------------------

```
# This is an untypical but still valid example

# The order of the lines do not matter
orders: oct-po
order_edits: oct-changes
curr_year: 2025
# The table lines may be written wherever (although it's probablty not a good idea)
2025 | 11 | 2025 | 3343136828.56
2025 | 12 | 2025 | 3343136828.56
curr_month: 9
invoices: oct-invoices
forecast_to: 1
2025 | 10 | 2025 | 3343136828.56
2026 | 1  | 2026 | 2093637309.18
2026 | 2  | 2026 | 1805975976.31
sample_frac: 1e-3
max_depth: 4
# Unknown keys are silenty ignored
unknown_key: 34
learning_rate: 0.8
mode: infer
2026 | 3  | 2026 | 4262730832.64
# Empty lines are ignored




2026 | 4  | 2026 | 2760625628.82
2026 | 10 | 2026 | 3343136828.56
unknown_key2: sdsdsd
# Duplicated table lines are simply added together
2026 | 11 | 2026 | 3343136828.56
2026 | 11 | 2026 | 3343136828.56
2026 | 11 | 2026 | 3343136828.56
2026 | 5  | 2026 | 4126391997.05
2026 | 8  | 2026 | 3126041425.00
2026 | 9  | 2026 | 3464815010.94
2026 | 12 | 2026 | 3343136828.56
smoothing_window: 4
2026 | 6  | 2026 | 5280905965.33
2026 | 7  | 2026 | 3167107311.76
# Missing keys are prompted interactively when the program executes
```


#### Configuration keys
| Option / Key            | Description |
|-------------------------|-------------|
| `experiment`            | Name of the experiment. |
| `orders`                | Key in the bucket for input `orders`. Should be of type `po-3`. |
| `invoices`              | Key in the bucket for input `invoices`. Should be of type `invoices-to-po-3`. |
| `order_edits`           | Key in the bucket for input `order_edits`. Should be of type `po-changes-1`. |
| `orders_dates`          | Key in the bucket for input `orders_dates`. Should be of type `dates-po-1`. |
| `curr_year`             | A numeric value `YYYY` indicating the year where¿combined with `curr_month`¿the data is cut before training/inference. |
| `curr_month`            | A numeric value between 1 and 12 indicating the month where¿combined with `curr_year`¿the data is cut before training/inference. |
| `forecast_to`           | A non-negative integer (usually 0 or 1) indicating the target forecast year.<br><br>If `forecast_to == 0`, the program infers for the current year.<br>If `forecast_to == 1`, the program infers for the next year.<br>In general, inference targets invoices during `current_year + forecast_to`. |
| `sample_frac`           | A decimal between 0 and 1 indicating what fraction of raw training data is used. Required for memory reasons.<br><br>Using 1 (full dataset) often exhausts memory. Since raw data is large (`#POs × #ages rows`), small fractions (e.g. `1e-3`) are still effective. |
| `n_estimators`          | Number of estimators used in the random forest. |
| `max_depth`             | Maximum depth used in the random forest. |
| `learning_rate`         | A decimal between 0 and 1. Learning rate used by the random forest. |
| `smoothing_window`      | Window size used to smooth labels. |
| `seed`                  | Seed for reproducible randomness in the algorithm. |
| `integer_features`      | Comma-separated list of integer features used in training and inference. Must be a subset of `{age, N}`. |
| `floating_features`     | Comma-separated list of floating features used in training and inference. Must be a subset of `{po_net_value, cumulative_portion}`. |
| `categorical_features`  | Comma-separated list of categorical features used in training and inference. Must be a subset of `{po_type, fingroup, huka, porcurment_organization, expenditure_type, quarter}`. |
| `mode`                  | Program mode. `train` = training mode, `infer` = inference mode. Any other value (including `mixed`) selects mixed mode. |


#### Augmentation table
A special kind of row in the configuration file is a row of the augmentation tables. As shown in the examples, these rows do not have to be consecutive or deduplicated,
although it's probably a good idea. Each row contains four numbers separated by pipes (|) in this order: financial year, month, fund year and amount. Each row denotes the total amount
of (additional) POs we assume will be created in this time (financial year + month) in this fund year. Duplicated rows' amounts are added silently.


