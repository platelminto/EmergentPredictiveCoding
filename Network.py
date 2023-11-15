import torch
import torch.nn.functional as F
import numpy as np
from ModelState import ModelState
import functions
from torch import nn


class Network(torch.nn.Module):
    """
    Recurrent Neural Network class containing parameters of the network
    and computes the forward pass.
    Returns hidden state of the network and preactivations of the units. 
    """
    def __init__(self, input_size: int, hidden_size: int, activation_func, weights_init=functions.init_params, prevbatch=False, conv=False, device=None):
        super(Network, self).__init__()

        self.input_size = input_size
        if conv:
            self.conv = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
        self.is_conv= conv
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.W = torch.nn.Parameter(weights_init(hidden_size, hidden_size))
        self.prevbatch = prevbatch
        self.device = device

    # x is the input at this timestep, state is the hidden state from the previous timestep
    def forward(self, x, state=None, synap_trans=False, mask=None):
        # state is the within-RNN-run state of the network, so 0 at first pass but is then passed as sequence continues
        if state is None:
            state = self.init_state(x.shape[0])
        h = state
        h = h.to(self.device)
        x = x.to(self.device)

        # pad input so it matches the hidden state dimensions
        if not self.is_conv:
            x_pad = F.pad(x, (0, self.hidden_size-self.input_size), "constant", 0)
            if mask is not None:
                a = h @ (self.W * mask) + x_pad
            else:
                a = h @ self.W + x_pad

        h = self.activation_func(a)
        # return state vector and list of parameters (pre-activation, post-act, & weights)
        return h, [a, h, self.W]

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))


class State(ModelState):
    def __init__(self,
                 activation_func,
                 optimizer,
                 lr:float,
                 title:str,
                 input_size:int,
                 hidden_size:int,
                 device:str,
                 deterministic=True,
                 weights_init=functions.init_params,
                 prevbatch=False,
                 conv=False,
                 seed=None):

        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.seed = seed

        ModelState.__init__(self,
                            
                            Network(input_size, hidden_size, activation_func, weights_init=weights_init, prevbatch=prevbatch, conv=conv, device=device).to(device),
                            optimizer,
                            lr,
                            
                            title,
                            {
                                "train loss": np.zeros(0),
                                "test loss": np.zeros(0),
                                "h": np.zeros(0),
                                "Wl1": np.zeros(0),
                                "Wl2": np.zeros(0)
                            },
                            device)

    def run_batch(self, batch, loss_fn: str, state=None):
        """
        Runs a batch of sequences through the model

        Returns:
            loss,
            training metadata
        """
        sequence_length = batch.shape[0]
        batch_size = batch.shape[1]
        if state == None:
            h = self.model.init_state(batch_size)
        else:
            # prevbatch? if this is internal RNN state, shouldnt we use 0 (ie the else)? so prevbatch does always seem
            # to be false, so yes, we always start with a new initial state for each RNN run. but it is passed, in case
            # we want to change this.
            if self.model.prevbatch:
                h = state
            else:
                h = self.model.init_state(batch_size)

        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        loss = loss.to(self.device)
        for i in range(sequence_length):
            # runs for the forward pass from above
            h, [pre_activation_args, post_activation_args, weights] = self.model(batch[i], state=h)  # these 3 are lists of potential losses
            loss = loss + self.loss(loss_fn, pre_activation_args, post_activation_args, weights)
        state = h
        return loss, state
    
    def get_next_state(self, state, x):
        """
        Return next state of model given current state and input

        """
        next_state, _ = self.model(x, state)
        return next_state

    def loss(self, loss: str, pre_activation_args, post_activation_args, weights):
        loss_t1, loss_t2,  beta = loss, None, 1
        # split for weighting
        if 'beta' in loss:
            beta, loss = loss.split('beta')
            beta = float(beta)
        if 'and' in loss:
            loss_t1, loss_t2 = loss.split('and')
    
        # parse loss terms
        loss_fn_t1, loss_arg_t1 = functions.get_loss_fn_and_arg(
            loss_t1, pre_activation_args, post_activation_args, weights
        )
        loss_fn_t2, loss_arg_t2 = functions.get_loss_fn_and_arg(
            loss_t2, pre_activation_args, post_activation_args, weights
        )
                    
        return loss_fn_t1(loss_arg_t1) + beta*loss_fn_t2(loss_arg_t2)

    def predict(self, state, latent=False):
        """
        Returns the networks 'prediction' for the input.
        """
        state = state.to(self.device)
        pred = state @ self.model.W
        if not latent:
            return pred[:,:self.model.input_size]
        return pred[:,:]

    def predict_predonly(self, state, pred_mask, latent=False):
        """
        Returns the networks 'prediction' for the input 
        from the prediction units only
        """
        W_pred = self.model.W.clone().detach()
        # set all non prediction units to zero
        W_pred[pred_mask==1, :] = 0
        pred = state @ W_pred
        if not latent:
            return pred[:,:self.model.input_size]
        return pred[:,:]     
        
    def step(self, loss):
        loss.backward()
        #nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def on_results(self, epoch:int, train_loss, test_loss, m_state):
        """Save training metadata
        """
        h, Wl1,Wl2 = m_state
        functions.append_dict(self.results, {"train loss": train_loss, "test loss": test_loss, "h": h, "Wl1": Wl1, "Wl2":Wl2})
        self.epochs += 1
