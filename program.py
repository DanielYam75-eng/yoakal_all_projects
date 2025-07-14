# %%
import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import Module, Sequential, Linear, Dropout, LSTM
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import mse_loss
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from typing import Iterator
Loader = Iterator[Tensor]

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--input_path",  "-i", type = str,  required = True)
parser.add_argument("--output_path", "-o", type = str,  required = True)
parser.add_argument("--model_path",  "-m", type = str,  required = True)
parser.add_argument("--target_year", "-y", type = int,  required = True)
parser.add_argument("--train",       "-t", type = int,  required = False, default = 1)
parser.add_argument("--seed_len",    "-s", type = int,  required = False, default = 1)
parser.add_argument("--batch_size",  "-b", type = int,  required = False, default = 3)
parser.add_argument("--epochs",      "-e", type = int,  required = False, default = 300)
args = parser.parse_args()

# %%
PATH  = args.input_path
GROUP = ["GL"]
VAL   = "volume"
YEAR  = "year"
MONTH = "month"

# %%
YSIZE  = 12
PYEAR  = args.target_year
EPOCHS = args.epochs
BSIZE  = args.batch_size
LRATE  = 1e-3
SLEN   = args.seed_len

# %% [markdown]
# <br>
# 
# ### Organizing the data

# %%
data = pd.read_csv(PATH, usecols = [ * GROUP, YEAR, MONTH, VAL])

# %%
data : pd.DataFrame = data[data[YEAR] <= PYEAR]

# %% [markdown]
# Setting groups as features

# %%
data = data.pivot_table(index = [YEAR, MONTH], columns = GROUP, values = VAL, aggfunc = 'sum', fill_value = 0).sort_index()

# %%
train = data.loc[data.index.get_level_values(level = YEAR) != PYEAR]
test  = data.loc[data.index.get_level_values(level = YEAR) == PYEAR]

# %% [markdown]
# Filtering for full years only

# %%
train = train.groupby(level = YEAR).filter(lambda g : g.count().min() == YSIZE)

# %% [markdown]
# Min-Max normalizing each group. Preserves order and sets a [0, 1] range.

# %%
scaler = MinMaxScaler()
scaler.fit(train)

# %% [markdown]
# Setting years as batches and months as the sequance.

# %%
def to_batches(table: pd.DataFrame) -> Tensor:
    return torch.stack([torch.FloatTensor(scaler.transform(group)) for year, group in table.groupby(level = YEAR)])

# %%
train_ten = to_batches(train)
test_ten  = to_batches(test)

# %%
target_ten = train_ten[:, 1:, :]
train_ten  = train_ten[:, :-1, :]

# %%
assert train_ten.size(2) == target_ten.size(2) == test_ten.size(2)
assert train_ten.size(1) == target_ten.size(1) == YSIZE - 1
assert train_ten.size(0) == target_ten.size(0)
assert test_ten.size(0) == 1
assert SLEN < test_ten.size(1)

# %%
tdataset = TensorDataset(train_ten, target_ten)
loader : Loader = DataLoader(tdataset, batch_size = BSIZE, shuffle = True)

# %% [markdown]
# <br>
# 
# ### Building architactures

# %%
class Forecaster(Module):
    def fit(self, loader: Loader, epochs: int, lr: float): pass
    def forecast(self, seed: Tensor, fh: int): pass

# %%
class MyRNN(Forecaster):

    def __init__(self, input_dim: int, lstm_hid: int = 300, dropr: float = 0.2):
        super().__init__()
        
        self.rnn = LSTM(input_dim, lstm_hid, batch_first = True, dropout = dropr, num_layers = 2)
        self.head = Sequential(Dropout(dropr), Linear(lstm_hid, input_dim))


    def forward(self, x: Tensor, hid = None) -> tuple[Tensor, Tensor]:
        x, hid = self.rnn(x, hid)
        x = self.head(x)
        return x, hid


    def fit(self, loader: Loader, epochs: int, lr: float):
        
        optimizer = Adam(self.parameters(), lr=lr)
        self.train()


        def train_step(data_batch: Tensor, target_batch: Tensor) -> float:
            
            data_batch += torch.randn_like(data_batch) * 1e-2
            optimizer.zero_grad()
            preds, _ = self.forward(data_batch)
            loss = mse_loss(preds, target_batch)
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

        for step in range(seed.size(1)):
            val = seed[:, step, :]
            yield val
            _, hid = self.forward(val, hid)

        for step in range(fh):
            val, hid = self.forward(val, hid)
            yield val


# %% [markdown]
# <br>
# 
# ### Training

# %%
model = MyRNN(input_dim = train_ten.size(-1))

# %%
if args.train:
    model.fit(loader, EPOCHS, LRATE)
    torch.save(model.state_dict(), args.model_path)
else:
    model.load_state_dict(torch.load(args.model_path))

# %% [markdown]
# <br>
# 
# ### Forecasting

# %%
pred = torch.concat(tuple(model.forecast(test_ten[:, :SLEN, :], YSIZE - SLEN))).relu()

# %%
test_ten = test_ten.squeeze()
assert pred.size(0) == YSIZE
assert pred.size(1) == test_ten.size(1)

# %%
pred = pred.detach().numpy()
test_ten = test_ten.detach().numpy()

# %%
pred = scaler.inverse_transform(pred)
test_ten = scaler.inverse_transform(test_ten)

# %%
print(f"total prediction: {pred.sum():.2e}")
print(f"total actual: {test_ten.sum():.2e}")

err = abs(pred.sum() - test_ten.sum())

print(f"total error: {err:.2e}")
print(f"error percentage: {abs(err / test_ten.sum()) * 100:.2f}%")

# %%
# results

test = test.transpose()
pred = pd.DataFrame(index = test.index, data = pred.transpose())

test = test.sum(axis = 1)
pred = pred.sum(axis = 1)

pred.to_csv(args.output_path + f" forecast {PYEAR}.csv")
test.to_csv(args.output_path + f" actual {PYEAR}.csv")