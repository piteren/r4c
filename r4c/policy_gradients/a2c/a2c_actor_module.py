import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.layers import LayDense, zeroes
from typing import Optional



class A2CModule(Module):

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            two_towers: bool=                   False,  # builds separate towers for Actor & Critic
            n_hidden: int=                      1,
            hidden_width: int=                  50,
            lay_norm: bool=                     False,
            clamp_advantage: Optional[float]=   0.5,    # limits advantage abs value
            use_huber: bool=                    False,  # for True uses Huber loss for Critic
            logger=                             None,
            loglevel=                           20,
    ):

        Module.__init__(self, logger=logger, loglevel=loglevel)

        hidden_layers = [hidden_width] * n_hidden
        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.lay_norm = lay_norm

        self.ln = torch.nn.LayerNorm(observation_width) if self.lay_norm else None # input layer norm

        # Actor (or Actor+Critic) tower layers
        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) if self.lay_norm else None for shape in lay_shapeL]
        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            if ln: self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        # Critic tower layers
        self.linL_critic_tower = [LayDense(*shape) for shape in lay_shapeL] if two_towers else []
        self.lnL_critic_tower = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL] if two_towers else []
        lix = 0
        for lin,ln in zip(self.linL_critic_tower, self.lnL_critic_tower):
            self.add_module(f'lay_lin_tower{lix}', lin)
            self.add_module(f'lay_ln_tower{lix}', ln)
            lix += 1

        # Critic value
        self.value = LayDense(
            in_features=    next_in,
            out_features=   1,
            activation=     None)

        # Actor policy logits
        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        self.clamp_advantage = clamp_advantage
        self.use_huber = use_huber

    def forward(self, observation:TNS) -> DTNS:

        inp = observation
        if self.lay_norm:
            inp = self.ln(observation)

        zsL = []
        out = inp
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        out_tower = out
        if self.linL_critic_tower:
            out_tower = inp
            for lin,ln in zip(self.linL_critic_tower, self.lnL_critic_tower):
                out_tower = lin(out_tower)
                zsL.append(zeroes(out_tower))
                if self.lay_norm: out_tower = ln(out_tower)
        value = self.value(out_tower)
        value = torch.reshape(value, (value.shape[:-1])) # remove last dim

        logits = self.logits(out)
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

        return {
            'value':    value,
            'logits':   logits,
            'probs':    probs,
            'zeroes':   zsL}

    def loss(
            self,
            observation: TNS,
            action: TNS,
            dreturn: TNS,
    ) -> DTNS:

        out = self(observation)

        value = out['value']
        logits = out['logits']

        advantage = dreturn - value
        advantage_nograd = advantage.detach() # to prevent flow of Actor loss gradients to Critic network

        if self.clamp_advantage is not None:
            advantage_nograd = torch.clamp(
                input=  advantage_nograd,
                min=    -self.clamp_advantage,
                max=    self.clamp_advantage)

        actor_ce = torch.nn.functional.cross_entropy(
            input=      logits,
            target=     action,
            reduction=  'none')
        loss_actor = torch.mean(actor_ce * advantage_nograd)

        if self.use_huber: loss_critic = torch.nn.functional.huber_loss(input=value, target=dreturn)
        else:              loss_critic = torch.nn.functional.mse_loss(input=value, target=dreturn)

        out.update({
            'advantage':    advantage,
            'loss_actor':   loss_actor,
            'loss_critic':  loss_critic,
            'loss':         loss_actor + loss_critic})

        return out