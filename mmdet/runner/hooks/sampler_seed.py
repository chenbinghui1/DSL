from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class DistSamplerSeedHook_semi(Hook):

    def before_epoch(self, runner):
        if hasattr(runner.data_loader, "multi_data_loaders"):
            for data_loader in runner.data_loader.multi_data_loaders:
                for d in data_loader.data_loaders:
                    runner.ITER = d.sampler.set_epoch(runner.epoch)
                    next(runner.ITER)
        else:
            runner.data_loader.sampler.set_epoch(runner.epoch)
