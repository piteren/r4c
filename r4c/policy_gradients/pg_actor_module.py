import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes


class PGActorModule(Module):
    """ baseline Policy Gradient Actor Module """

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
            seed=               121,
            logger=             None,
            loglevel=           20,
    ):

        Module.__init__(self, logger=logger, loglevel=loglevel)

        lay_shapeL = []
        next_in = observation_width
        for _ in range(n_hidden):
            lay_shapeL.append((next_in,hidden_width))
            next_in = hidden_width

        self.lay_norm = lay_norm

        self.ln = torch.nn.LayerNorm(observation_width) if self.lay_norm else None # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) if self.lay_norm else None for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            if ln: self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

    def forward(self, observation:TNS) -> DTNS:

        out = observation
        if self.lay_norm:
            out = self.ln(observation)

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm:
                out = ln(out)

        logits = self.logits(out)
        dist = torch.distributions.Categorical(logits=logits)

        return {
            'logits':   logits,
            'probs':    dist.probs,
            'entropy':  dist.entropy().mean(),
            'zeroes':   zsL}

    def loss(
            self,
            observation: TNS,
            action: TNS,
            dreturn: TNS,
    ) -> DTNS:

        out = self(observation)

        actor_ce = torch.nn.functional.cross_entropy(
            input=      out['logits'],
            target=     action,
            reduction=  'none')
        loss = torch.mean(actor_ce * dreturn)

        out.update({
            'cross_entropy':    torch.mean(actor_ce),
            'entropy':          out['entropy'],
            'loss':             loss})

        return out