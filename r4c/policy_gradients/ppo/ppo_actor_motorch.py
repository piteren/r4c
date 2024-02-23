import random
import torch
from torchness.types import DTNS
from torchness.motorch import MOTorch


class MOTorch_PPO(MOTorch):

    def __init__(self, **kwargs):

        MOTorch.__init__(self, **kwargs)

        # INFO: for large PPO updates disables strange CUDA error
        # those backends turn on / off implementations of SDP (scaled dot product attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False) # enables or disables Memory-Efficient Attention
        torch.backends.cuda.enable_flash_sdp(False) # enables or disables FlashAttention
        torch.backends.cuda.enable_math_sdp(True) # enables or disables PyTorch C++ implementation

    def backward(
            self,
            bypass_data_conv=   True,
            set_training: bool= True,
            empty_cuda_cache=   True,
            **kwargs
    ) -> DTNS:
        """ backward in PPO mode """

        batch = {k: self.convert(data=kwargs[k]) for k in kwargs}

        batch_len = batch['action'].shape[0]
        mb_size = batch_len // self.minibatch_num
        batch_spl = {k: torch.split(batch[k], mb_size, dim=0) for k in batch} # split along 0 axis into chunks of mb_size
        minibatches = [
            {k: batch_spl[k][ix] for k in batch}            # list of dicts {key: TNS}, where TNS is a minibatch rectangle
            for ix in range(len(batch_spl['action']))]      # num of minibatches

        if self.n_epochs_ppo > 1:
            mb_more = minibatches * (self.n_epochs_ppo - 1)
            random.shuffle(mb_more)
            minibatches += mb_more

        res = {}
        for mb in minibatches:

            out = self.loss(
                bypass_data_conv=   bypass_data_conv,
                set_training=       set_training,
                **mb)
            self.logger.debug(f'> loss() returned: {list(out.keys())}')

            for k in out:
                if k not in res:
                    res[k] = []
                res[k].append(out[k])

            self._opt.zero_grad()           # clear gradients
            out['loss'].backward()          # build gradients

            gnD = self._grad_clipper.clip() # clip gradients, adds 'gg_norm' & 'gg_norm_clip' to out
            for k in gnD:
                if k not in res:
                    res[k] = []
                res[k].append(gnD[k])

            self._opt.step()                # apply optimizer

        ### merge outputs

        res_prep = {}
        for k in ['probs','zeroes']:
            res_prep[k] = torch.cat(res[k], dim=0)

        for k in [
            'entropy',
            'loss',
            'loss_actor',
            'gg_norm',
            'gg_norm_clip',
            'approx_kl',
            'clipfracs']:
            res_prep[k] = torch.Tensor(res[k]).mean()

        self._scheduler.step()  # apply LR scheduler
        self.train_step += 1    # update step

        res_prep['currentLR'] = self._scheduler.get_last_lr()[0]  # INFO: currentLR of the first group is taken

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        return res_prep