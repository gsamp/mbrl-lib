import .one_dim_tr_model

class BackwardsOneDTransitionRewardModel(OneDTransitionRewardModel):
    def _process_batch(self, batch: mbrl.types.TransitionBatch, _as_float: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        next_obs, action, obs, reward, _ = batch.astuple()
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]
        else:
            target_obs = next_obs
        target_obs = model_util.to_tensor(target_obs).to(self.device)

        model_in, *_ = self._get_model_input(obs, action)
        if self.learned_rewards:
            reward = model_util.to_tensor(reward).to(self.device).unsqueeze(reward.ndim)
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs
        return model_in.float(), target.float()
