# %%
import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import Module, Sequential, Linear, LSTM
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import mse_loss
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from typing import Iterator
Loader = Iterator[Tensor]

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--input_path",  "-i", type = str,   required = True)
parser.add_argument("--output_path", "-o", type = str,   required = True)
parser.add_argument("--model_path",  "-m", type = str,   required = True)
parser.add_argument("--target_year", "-y", type = int,   required = True)
parser.add_argument("--train",       "-t", type = int,   required = False,  default = 1)
parser.add_argument("--seed_len",    "-s", type = int,   required = False,  default = 1)
parser.add_argument("--batch_size",  "-b", type = int,   required = False,  default = 3)
parser.add_argument("--learn_rate",  "-l", type = float, required = False,  default = 1e-3)
parser.add_argument("--hidden_size", "-n", type = int,   required = False,  default = 300)
parser.add_argument("--epochs",      "-e", type = int,   required = False,  default = 250)
args = parser.parse_args()

# %%
PATH  = args.input_path
GROUP = ["doc", "tressure group"]
VAL   = "volume"
YEAR  = "year"
MONTH = "month"

# %%
YSIZE  = 12
THRESH = 1e+6
LASSO  = 1e-2
NOISE  = 1e-2
HIDNS  = args.hidden_size
PYEAR  = args.target_year
EPOCHS = args.epochs
BSIZE  = args.batch_size
LRATE  = args.learn_rate
SLEN   = args.seed_len

# %%
# ### Organizing the data

# %%
data = pd.read_csv(PATH, usecols = [ * GROUP, YEAR, MONTH, VAL])

# %%
data : pd.DataFrame = data[data[YEAR] <= PYEAR]

# %%
data = data.groupby(GROUP).filter(lambda g : g.groupby(YEAR)[VAL].sum().mean() > THRESH)

# %%
# Building the time series.

data = data.pivot_table(index = [YEAR, MONTH], columns = GROUP, values = VAL, aggfunc = 'sum', fill_value = 0).sort_index()

# %%
train : pd.DataFrame = data.loc[data.index.get_level_values(level = YEAR) != PYEAR]
test  : pd.DataFrame = data.loc[data.index.get_level_values(level = YEAR) >= PYEAR - 1]

# %% 
# Filtering for full years only

# %%
train = train.groupby(level = YEAR).filter(lambda g : g.count().min() == YSIZE)

# %% 
# Min-Max normalizing each group. Preserves order and sets a [0, 1] range.

scaler = MinMaxScaler()
scaler.fit(train)

# %% 
# Setting years as batches and months as the sequance.

def pipeline(table: pd.DataFrame) -> Tensor:
    table = table.groupby(level = YEAR)
    table = table.apply(scaler.transform)
    table = table.apply(torch.FloatTensor)
    table = table.values.tolist()
    table = torch.stack([torch.cat([prev, curr]) for prev, curr in zip(table, table[1:])])
    return table

# %%
train_ten = pipeline(train)
test_ten  = pipeline(test)

# %%
target_ten = train_ten[:, 1:, :]
train_ten  = train_ten[:, :-1, :]

# %%
assert train_ten.size(2) == target_ten.size(2) == test_ten.size(2)
assert train_ten.size(1) == target_ten.size(1) == 2 * YSIZE - 1
assert train_ten.size(0) == target_ten.size(0)
assert test_ten.size(0) == 1
assert SLEN < test_ten.size(1) - YSIZE

# %%
tdataset = TensorDataset(train_ten, target_ten)
loader : Loader = DataLoader(tdataset, batch_size = BSIZE, shuffle = True)

# %%
# ### Building architectures


class Forecaster(Module):
    def fit(self, loader: Loader, epochs: int, lr: float): pass
    def forecast(self, seed: Tensor, fh: int): pass
    

class BiYearlyRNN(Forecaster):

    def __init__(self, input_dim: int, lstm_hid: int = 300):
        super().__init__()
        
        self.rnn = LSTM(input_dim, lstm_hid, batch_first = True)
        self.head = Sequential(Linear(lstm_hid, input_dim))


    def forward(self, x: Tensor, hid = None) -> tuple[Tensor, Tensor]:
        x, hid = self.rnn(x, hid)
        x = self.head(x)
        return x, hid
    

    def lasso(self) -> Tensor:
        return torch.cat([p.abs().flatten() for n, p in self.named_parameters() if "weight_ih" in n or "linear" in n]).mean() * LASSO
    

    def loss(self, curr_year_preds: Tensor, target: Tensor) -> Tensor:
        return mse_loss(curr_year_preds, target) + self.lasso()


    def fit(self, loader: Loader, epochs: int, lr: float):
        
        optimizer = Adam(self.parameters(), lr=lr)
        self.train()


        def train_step(data_batch: Tensor, target_batch: Tensor) -> float:
            
            data_batch += torch.randn_like(data_batch) * NOISE
            optimizer.zero_grad()
            curr_year_preds, _ = self.forward(data_batch)
            loss = self.loss(curr_year_preds, target_batch)
            loss.backward()
            clip_grad_norm_(self.parameters(), max_norm = 1.0)
            optimizer.step()
            return loss.item()

        
        last_avg_loss = float('inf')
        strike_count = 0
        for epoch in range(1, epochs + 1):
            
            total_loss = sum(train_step(data_batch, target_batch) for data_batch, target_batch in loader)

            if epoch % 10 == 0:

                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")

                if avg_loss > last_avg_loss:
                    strike_count += 1
                    if strike_count >= 6:
                        print("Early stopping due to loss increase.")
                        break

                last_avg_loss = avg_loss


    @torch.no_grad
    def forecast(self, seed: Tensor, fh: int, hid = None):
        
        self.eval()
        forecast = []

        for step in range(seed.size(1)):
            val = seed[:, step, :]
            forecast.append(val)
            _, hid = self.forward(val, hid)

        for step in range(fh):
            val, hid = self.forward(val, hid)
            forecast.append(val)

        return torch.concat(forecast).relu()


# %%
# ### Training

# %%
model = BiYearlyRNN(input_dim = train_ten.size(-1), lstm_hid = HIDNS)

# %%
# saving / loading the model
if args.train:
    model.fit(loader, EPOCHS, LRATE)
    torch.save(model.state_dict(), args.model_path)
else:
    model.load_state_dict(torch.load(args.model_path))

# %%
# ### Forecasting

curr_year_pred = model.forecast(test_ten[:, : YSIZE + SLEN, :], YSIZE - SLEN)
curr_year_pred = curr_year_pred.unsqueeze(0)
next_year_pred = model.forecast(curr_year_pred[:, : YSIZE, :], YSIZE)


# %%
def organize_forecast(forecast: Tensor, test: pd.DataFrame) -> pd.DataFrame:

    # dimention check
    forecast = forecast.squeeze()
    assert forecast.size(0) == YSIZE * 2
    assert forecast.size(1) == test.shape[1]

    # convertion
    forecast = forecast.detach().numpy()

    # denormalization
    forecast = scaler.inverse_transform(forecast)

    # cat out the warm up
    forecast = forecast[YSIZE:]
    test = test.iloc[YSIZE:]

    # tabularizing results

    test     : pd.DataFrame = test.transpose()
    forecast : pd.DataFrame = pd.DataFrame(index = test.index, data = forecast.transpose())

    test = test.sum(axis = 1).to_frame(name = "actual")
    forecast = forecast.sum(axis = 1).to_frame(name = "forecast")
    results = forecast.join(test)

    results.sort_values(by = "actual", inplace = True, ascending = False)
    results.loc[results["actual"] == 0, "forecast"] = 0
    results["abs err"] = results["forecast"].sub(results["actual"]).abs()
    results["rel err"] = results["abs err"].div(results["actual"]).round(2)


    results = results.reset_index()

    return results


# %%
# saving results

organize_forecast(curr_year_pred, test).to_csv(args.output_path + f"results {PYEAR}.csv",     index = False)
organize_forecast(next_year_pred, test).to_csv(args.output_path + f"results {PYEAR + 1}.csv", index = False)