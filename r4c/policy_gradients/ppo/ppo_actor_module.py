import torch
from torchness.motorch import Module
from torchness.types import TNS, DTNS
from torchness.layers import LayDense, zeroes
from torchness.base_elements import select_with_indices


class PPOActorModule(Module):
    """ baseline PPO Actor Module """

    def __init__(
            self,
            observation_width: int,
            num_actions: int,
            minibatch_num: int,
            n_epochs_ppo: int,
            n_hidden: int=      2,
            hidden_width: int=  12,
            lay_norm=           False,
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
            'logits':       logits,
            'probs':        dist.probs,
            'entropy':      dist.entropy().mean(),
            'zeroes':       zsL}

    def fwd_logprob(self, observation:TNS, action:TNS) -> DTNS:
        """ FWD
        + preparation of logprob (ln(prob) of selected move) """
        out = self(observation)
        prob_move = select_with_indices(source=out['probs'], indices=action)
        out['logprob'] = torch.log(prob_move)
        return out

    def fwd_logprob_ratio(self, observation:TNS, action:TNS, old_logprob:TNS) -> DTNS:
        """ FWD
        + logprob (current)
        + ratio of current logprob vs given old
        + prepares additional metrics """

        logrpob_out = self.fwd_logprob(observation=observation, action=action)
        new_logrpob = logrpob_out.pop('logprob')
        logratio = new_logrpob - old_logprob
        ratio = logratio.exp()

        out = logrpob_out
        out['ratio'] = ratio

        # stats
        with torch.no_grad():
            out.update({
                'approx_kl':    ((ratio - 1) - logratio).mean(),
                'clipfracs':    ((ratio - 1.0).abs() > self.clip_coef).float().mean()})

        return out

    def loss(
            self,
            observation: TNS,
            action: TNS,
            dreturn: TNS,
            logprob: TNS,
    ) -> DTNS:

        # TODO: rewrite with PPO
        # below is a pypoks version (no critic & advantages)

        ratio_out = self.fwd_logprob_ratio(observation=observation, action=action, old_logprob=logprob)




        reward_norm = self.norm(reward)

        loss_actor = self.loss_actor(
            advantage=  reward_norm if self.reward_norm else reward,
            ratio=      ratio_out['ratio'])
        loss_actor *= deciding_state
        loss_actor = torch.mean(loss_actor)

        entropy = ratio_out['entropy']
        loss_entropy_factor = self.entropy_coef * entropy

        loss_nam = self.loss_nam(
            logits=         ratio_out['logits'],
            allowed_moves=  allowed_moves)
        loss_nam *= deciding_state
        loss_nam = torch.mean(loss_nam)
        loss_nam_factor = self.nam_loss_coef * loss_nam

        loss = loss_actor - loss_entropy_factor + loss_nam_factor

        out = ratio_out
        out.update({
            'reward':       reward,
            'reward_norm':  reward_norm,
            'entropy':      entropy,
            'loss':         loss,
            'loss_actor':   loss_actor,
            'loss_nam':     loss_nam})
        return out
        """

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