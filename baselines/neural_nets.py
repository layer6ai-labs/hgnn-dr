import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable


def nn_linear(in_sz, out_sz, w_init=nn.init.xavier_normal_, b_init=0, **kwargs):
    m = nn.Linear(in_sz, out_sz, **kwargs)
    w_init(m.weight)
    m.bias.data.fill_(b_init)
    return m


class BinaryMLP(nn.Module):
    def __init__(self,
        transform_X = True,
        hidden_dim = 50,
        hidden_layers = 3,
        seed = 1234,

        lr = 1e-3,
        beta1 = 0.9,
        beta2 = 0.999,
        weight_decay = 0.,
        dropout_p = 0.,

        max_iter = 500,
        early_stopping_rounds = None,
        batch_size = None,

        verbosity = 20,
        use_cuda = True,
    ):
        super(BinaryMLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.transform_X = transform_X
        self.seed = seed

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p

        self.max_iter = max_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.criterion = nn.BCEWithLogitsLoss()

        self.network_initialized = False
        self.best_iter = None

    def initialize_network(self, input_size, **kwargs):
        H = [self.hidden_dim for _ in range(self.hidden_layers)]
        self.fc_input = nn_linear(input_size, H[0])

        self.transforms = [nn.Dropout(p=self.dropout_p), nn.ReLU()]
        if len(H) > 1:
            for h0, h1 in zip(H[:-1], H[1:]):
                self.transforms += [nn_linear(h0, h1), nn.Dropout(p=self.dropout_p), nn.ReLU()]

        self.transforms = nn.ModuleList(self.transforms)
        self.fc_final = nn_linear(H[-1], 1)

        self.to_cuda()
        self.network_initialized = True

    def to_cuda(self, torch_obj=None):
        if torch_obj is None:
            torch_obj = self
        return torch_obj.cuda() if self.use_cuda else torch_obj

    def forward(self, x):
        if not self.network_initialized:
            self.initialize_network(x.shape[1])

        out = self.fc_input(x)
        for transform in self.transforms:
            out = transform(out)
        out = self.fc_final(out)
        return out

    def fit(self, X, y, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        X, y = torch.Tensor(X), torch.Tensor(y).reshape(-1, 1)

        check_val = (X_val is not None) and (y_val is not None)
        early_stop = False
        self.best_iter = self.max_iter
        if check_val:
            X_val = self.to_cuda(torch.Tensor(X_val))
            y_val = self.to_cuda(torch.Tensor(y_val).reshape(-1, 1))
            early_stop_counter = 0
            val_loss_checkpoint = None

        if not self.network_initialized:
            self.initialize_network(X.shape[1])

        self.train()
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     betas=(self.beta1, self.beta2),
                                     weight_decay=self.weight_decay)

        for epoch in range(self.max_iter):
            training_loss = 0

            perm = torch.randperm(X.shape[0])
            if self.batch_size is None:
                batches = [X[perm]], [y[perm]]
            else:
                batches = (torch.split(X[perm], self.batch_size), torch.split(y[perm], self.batch_size))

            for (Xi, yi) in zip(*batches):
                # Convert torch tensor to Variable
                Xi = self.to_cuda(Variable(Xi))
                yi = self.to_cuda(Variable(yi))

                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = self(Xi)
                loss = self.criterion(outputs, yi)
                loss.backward()
                optimizer.step()

                training_loss += loss.data*len(Xi)
            training_loss /= len(X)

            if check_val:
                with torch.no_grad():
                    val_loss = self.criterion(self(X_val), y_val).data

                if self.early_stopping_rounds is not None:
                    if (val_loss_checkpoint is None) or (val_loss <= val_loss_checkpoint):
                        early_stop_counter = 0
                        best_iter_checkpoint = epoch
                        val_loss_checkpoint = val_loss
                        state_dict_checkpoint = deepcopy(self.state_dict())
                    else:
                        early_stop_counter += 1
                        early_stop = early_stop_counter >= self.early_stopping_rounds

            if (self.verbosity > 0) and ( ((epoch+1) % self.verbosity == 0)
                                                                        or early_stop or (epoch+1==self.max_iter) ):
                print('Epoch [%d/%d]: training loss %.4f' %(epoch+1, self.max_iter, training_loss), end="")
                if check_val:
                    print((', val loss %.4f' % val_loss) if check_val else '', end="")
                print("")

            if early_stop:
                if self.verbosity >= 0:
                    print(f'Best epoch {best_iter_checkpoint+1}: val loss %.4f' % val_loss_checkpoint)
                self.load_state_dict(state_dict_checkpoint)
                self.best_iter = best_iter_checkpoint+1
                break

    @torch.no_grad()
    def predict(self, X, to_numpy=True):
        self.eval()
        X = self.to_cuda(torch.Tensor(X))
        prediction = torch.sigmoid(torch.flatten(self(X)))
        return prediction.cpu().numpy() if to_numpy else prediction
