from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class EMAOWNHook(Hook):

    def __init__(self,
                 interval=-1,
                 mode="epoch",
                 ratio=0.99,
                 start_point=-1,
                 step_decay=None,
                 decay_ratio=0.1,
                 **kwargs):
        self.interval = interval
        self.mode = mode
        self.start_point = start_point
        self.ratio = ratio
        self.args = kwargs
        self.step_decay = step_decay
        self.decay_ratio = decay_ratio

    def after_train_epoch(self, runner):
        if self.step_decay != None and runner.epoch+1 in self.step_decay:
                self.ratio = max(1.0 - (1.0-self.ratio) / self.decay_ratio, 0.01)
                runner.logger.info("[INFO] ema ratio changes to %f",self.ratio)
        if self.mode == "epoch":
            if self.interval == -1 or self.start_point>runner.epoch+1:
                return
            if not self.every_n_epochs(runner, self.interval):
                return
            runner.EMA(keep_rate = self.ratio, mode = self.mode, start_point=self.start_point, **self.args)
        else:
            return

    def after_train_iter(self, runner):
        if self.mode == "iteration":
            if self.interval == -1 or self.start_point>runner.iter+1:
                return
            if not self.every_n_iters(runner, self.interval):
                return
            runner.EMA(keep_rate = self.ratio, mode = self.mode, start_point=self.start_point, **self.args)
        else:
            return

