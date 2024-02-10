import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.layers import LayDense


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
            n_hidden: int=      2,
            hidden_width: int=  12,
            use_huber: bool=    True, # MSE / Huber loss
            seed=               121,
            logger=             None,
            loglevel=           20,
    ):

        Module.__init__(self, logger=logger, loglevel=loglevel)

        hidden_layers = [hidden_width] * n_hidden
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

    def forward(self, observation:TNS) -> DTNS:
        out = self.ln(observation.to(torch.float32)) # + safety dtype convert
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            out = ln(out)
        return {'qvs': self.logits(out)}

    def loss(
            self,
            observation: TNS,
            action: TNS,
            new_qv: TNS,
    ) -> DTNS:
        out = self(observation)
        qv_pred = out['qvs'][range(len(action)),action]
        loss = self.loss_fn(qv_pred, new_qv)
        out['loss'] = torch.mean(loss)
        return out