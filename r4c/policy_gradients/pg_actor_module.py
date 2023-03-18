import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes
from torchness.base_elements import scaled_cross_entropy


# baseline Policy Gradient Actor Module
class PGActorModule(Module):

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
            use_scaled_ce=      False, # experimental Scaled Cross Entropy loss
            seed=               121):

        torch.nn.Module.__init__(self)

        hidden_layers = [hidden_width] * n_hidden
        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

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

        self.use_scaled_ce = use_scaled_ce


    def forward(self, observations:TNS) -> DTNS:

        out = observations
        if self.lay_norm:
            out = self.ln(observations)

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        logits = self.logits(out)
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

        return {
            'logits':       logits,
            'probs':        probs,
            'zeroes':       zsL}


    def loss(
            self,
            observations: TNS,
            actions_taken: TNS,
            dreturns: TNS,
    ) -> DTNS:

        out = self(observations)
        logits = out['logits']

        if self.use_scaled_ce:
            ceo = scaled_cross_entropy(
                labels= actions_taken,
                scale=  dreturns,
                probs=  out['probs'])
            actor_ce = ceo['cross_entropy']
            actor_ce_scaled = ceo['scaled_cross_entropy']
        else:
            actor_ce = torch.nn.functional.cross_entropy(input=logits, target=actions_taken, reduction='none')
            actor_ce_scaled = actor_ce * dreturns

        out.update({
            'cross_entropy':    torch.mean(actor_ce),
            'loss':             torch.mean(actor_ce_scaled)})

        return out