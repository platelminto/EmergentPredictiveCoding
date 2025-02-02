import sys
import torch

import Network
import functions
from Dataset import Dataset

import matplotlib.pyplot as plt


def test_epoch(ms: Network.State,
               dataset: Dataset,
               loss_fn: str,
               batch_size: int,
               sequence_length: int):
    batches, labels = dataset.create_batches(batch_size=batch_size, sequence_length=sequence_length, shuffle=True)
    num_batches = batches.shape[0]
    batch_size = batches.shape[2]
    tot_loss = 0
    tot_res = None 
    state = None
    for i, batch in enumerate(batches):

        with torch.no_grad():
            loss, state = test_batch(ms, batch, loss_fn, state)
    
        tot_loss += loss

    tot_loss /= num_batches
    print("Test loss:     {:.8f}".format(tot_loss))
    return tot_loss


def test_batch(ms: Network.State,
               batch: torch.FloatTensor,
               loss_fn: str,
               state):
    loss, state = ms.run_batch(batch, loss_fn, state)
    return loss.item(), state


def train_batch(ms: Network.State,
                batch: torch.FloatTensor,
                loss_fn: str,
                state):

    loss, state = ms.run_batch(batch, loss_fn, state)
   
    ms.step(loss)
    ms.zero_grad()
    return loss.item(), state


def train_epoch(ms: Network.State,
                dataset: Dataset,
                loss_fn: str,
                batch_size: int,
                sequence_length: int,
                verbose = True):

    batches, labels = dataset.create_batches(batch_size=batch_size, sequence_length=sequence_length, shuffle=True)
    num_batches = batches.shape[0]
    batch_size = batches.shape[2]

    t = functions.Timer()
    tot_loss = 0.
    state = None
    for i, batch in enumerate(batches):

        loss, state = train_batch(ms, batch, loss_fn, state)
        tot_loss += loss

        if verbose and (i+1) % int(num_batches/10) == 0:
            dt = t.get(); t.lap()
            print("Batch {}/{}, ms/batch: {}, loss: {:.5f}".format(i, num_batches, dt / (num_batches/10), tot_loss/(i)))

    tot_loss /= num_batches

    print("Training loss: {:.8f}".format(tot_loss))

    return tot_loss, state.detach()


def train(ms: Network.State,
          train_ds: Dataset,
          test_ds: Dataset,
          loss_fn: str,
          num_epochs: int = 1,
          batch_size: int = 32,
          sequence_length: int = 3,
          patience: int = 200,
          verbose = False):
    ms_name = ms.title.split('/')[-1]
    best_epoch = 0; tries = 0
    best_loss = sys.float_info.max
    best_network = None

    for epoch in range(ms.epochs+1, ms.epochs+1 + num_epochs):
        print("Epoch {}, Lossfn {}".format(epoch, ms_name))

        train_loss, h = train_epoch(ms, train_ds, loss_fn, batch_size, sequence_length, verbose=verbose)

        test_loss = test_epoch(ms, test_ds, loss_fn, batch_size, sequence_length)
#        if epoch == 1 or epoch == num_epochs - 10:
#            W = ms.model.W.detach()
#            torch.save(W, 'models/'+ms.title+'W_'+ str(epoch)+'.pt')
        h, W_l1, W_l2 = functions.L1Loss(h),  functions.L1Loss(ms.model.W.detach()), functions.L2Loss(ms.model.W.detach())
        m_state = [[h.cpu().numpy()], [W_l1.cpu().numpy()], [W_l2.cpu().numpy()]]
        ms.on_results(epoch, train_loss, test_loss, m_state)

        if (test_loss < best_loss):
            best_loss = test_loss
            best_epoch = epoch
            best_network = ms.model.W.detach()
            tries = 0
        else:
            print("Loss did not improve from", best_loss)
            tries = tries + 1
            if (tries >= patience):
                print("Stopping early")
                break;
