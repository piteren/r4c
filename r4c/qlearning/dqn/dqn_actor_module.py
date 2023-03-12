import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.layers import LayDense
from typing import Optional


# Deep QNetwork Model
class DQNModel(Module):
    """
    Since in QLearning number of observations may be finite (in QTable is),
    observations received by model may be of int dtype.
    Those are preventively converted to float in froward() and loss().
    """

    def __init__(
            self,
            num_actions: int=   4,
            observation_width=  4,
            hidden_layers=      (12,),
            use_huber: bool=    False,  # for True uses Huber loss
            seed=               121):

        torch.nn.Module.__init__(self)

        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.ln = torch.nn.LayerNorm(observation_width) # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        loss_class = torch.nn.HuberLoss if use_huber else torch.nn.MSELoss
        self.loss_fn = loss_class(reduction='none')


    def forward(self, observations:TNS) -> DTNS:
        out = self.ln(observations.to(torch.float32)) # + safety convert dtype
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            out = ln(out)
        return {'logits': self.logits(out)}


    def loss(
            self,
            observations: TNS,
            labels: TNS,
            mask: Optional[TNS]=    None
    ) -> DTNS:
        out = self(observations)
        loss = self.loss_fn(out['logits'], labels.to(torch.float32)) # + safety convert dtype
        if mask is not None:
            loss *= mask                        # mask
        loss = torch.sum(loss, dim=-1)          # reduce over samples
        out['loss'] = torch.mean(loss)          # average
        return out