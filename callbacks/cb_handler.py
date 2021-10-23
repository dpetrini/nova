# Handler class for call other callback classes


class CallbackHandler():

    def __init__(self, cbs=None):
        self.best_val_acc = 0.05
        print('\nRegistered Callbacks:', ' ', end='')
        for cb in cbs:
            print(cb.__class__.__name__, ' ', end='')
        print()
        self.cbs = cbs if cbs else []

        for cb in self.cbs:
            print(cb.__class__.__name__, ' ', end='')

        self._name = []
        self._elapsed = {}
        self._best_model_f = {}
        self._best_metric_p = {}
        self._best_loss_p = {}
        self._best_metric = {}
        # self._best_metric_ep = []
        self._best_metric_ep = {}

    def __repr__(self):
        for cb in self.cbs:
            print(cb, ' ', end='')
        return ''

    def begin_train_val(self, epochs, model, train_dataloader, val_dataloader, bs_size, optimizer):
        for cb in self.cbs:
            cb.begin_train_val(epochs, model, train_dataloader, val_dataloader, bs_size, optimizer)
        return True

    def update_loss(self, *args, **kwargs):
        for cb in self.cbs:
            cb.update_loss(*args, **kwargs)
        return True

    # def update_LR(self, epoch, model, optimizer, stages):
    def update_LR(self, *args, **kwargs):
        for cb in self.cbs:
            # optimizer = cb.update_LR(epoch, model, optimizer, stages)
            optimizer = cb.update_LR(*args, **kwargs)
        return optimizer  # True

    def begin_epoch(self, current_epoch):
        for cb in self.cbs:
            cb.begin_epoch(current_epoch)
        return True

    def begin_batch(self, *args):
        for cb in self.cbs:
            ret = cb.begin_batch(*args)
        return ret

    def begin_val(self, *args):
        for cb in self.cbs:
            cb.begin_val(*args)
        return True

    def after_epoch(self, model, train_acc, train_loss, val_acc, val_loss, **kwargs):
        for cb in self.cbs:
            cb.after_epoch(model, train_acc, train_loss, val_acc, val_loss, **kwargs)
        return True

    def after_step(self, n_samples, *args):
        for cb in self.cbs:
            cb.after_step(n_samples, *args)
        return True

    def after_step_val(self, n_samples, *args):
        for cb in self.cbs:
            cb.after_step_val(n_samples, *args)
        return True

    def after_train_val(self):
        for cb in self.cbs:
            cb.after_train_val()
        return True

    @property
    def best_metric_epoch(self):
        """
        Validation:
        If callback has best_metric_ep variable will update it to outside.
        So that main trainer can change last epoch to some value after
        training get stable (N epochs after last best_metric_epoch).
        """
        # for cb in self.cbs:
        #     if cb and hasattr(cb, 'best_metric_epoch'):
        #         _best_metric_ep = cb.best_metric_epoch
        #         return _best_metric_ep
        # return True

        for cb in self.cbs:
            # print(cb.__class__.__name__, ' ', end='')
            if cb and hasattr(cb, 'best_metric_epoch') and hasattr(cb, 'metric_name'):
                # print('M: ', cb.best_metric_epoch, cb.metric_name)
                # self._best_metric_ep.append(cb.best_metric_epoch)
                self._best_metric_ep[cb.metric_name] = cb.best_metric_epoch
            # else:
            #     self._best_metric_ep.append(0)
        return self._best_metric_ep # if cb.best_metric_epoch else True
        # return True

    @property
    def best_metric(self):
        """
        Validation:
        If callback has best_metric value will update it to outside.
        Uses metric name from callback itself as index for dictionary.
        Examples: Accuracy, AUC
        """
        # for cb in self.cbs:
        #     if cb and hasattr(cb, 'best_metric'):
        #         _best_metric = cb.best_metric
        #         return _best_metric
        # return True

        for cb in self.cbs:
            if cb and hasattr(cb, 'best_metric') and hasattr(cb, 'metric_name'):
                self._best_metric[cb.metric_name] = cb.best_metric
        return self._best_metric
        return True


    # # Workaround para passar modelo
    @property
    def last_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'last_model'):
                _model = cb.last_model
        return _model
        return True

    @property
    def best_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_model'):
                _best_model = cb.best_model
        return _best_model
        return True

    @property
    def best_auc_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_auc_model'):
                _best_model = cb.best_auc_model
        return _best_model
        return True

    # @property
    # def loss_plot(self):
    #     for cb in self.cbs:
    #         if cb and hasattr(cb, 'loss_plot'):
    #             _return = cb.loss_plot
    #             return _return
    #     return True


    @property
    def loss_plot(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'loss_plot') and hasattr(cb, 'metric_name'):
                self._best_loss_p[cb.metric_name] = cb.loss_plot
        return self._best_loss_p
        return True

    # @property
    # def metric_plot(self):
    #     for cb in self.cbs:
    #         if cb and hasattr(cb, 'metric_plot'):
    #             _return = cb.metric_plot
    #             return _return
    #     return True

    @property
    def metric_plot(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'metric_plot') and hasattr(cb, 'metric_name'):
                self._best_metric_p[cb.metric_name] = cb.metric_plot
        return self._best_metric_p
        return True

    # @property
    # def best_model_file(self):
    #     for cb in self.cbs:
    #         if cb and hasattr(cb, 'best_model_file'):
    #             _return = cb.best_model_file
    #             return _return
    #     return True

    @property
    def best_model_file(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_model_file') and hasattr(cb, 'metric_name'):
                self._best_model_f[cb.metric_name] = cb.best_model_file
        return self._best_model_f
        return True

    @property
    def metric_name(self):
        """
        Save metric name in a non-repetition list.
        Outside processing will need to retrieve proper metric name to index
        other parameters like best metric, best epoch, elapsed time, etc.
        """
        for cb in self.cbs:
            # print('AQUI-', self._name)
            if cb and hasattr(cb, 'metric_name'):
                # _return = cb.metric_name
                if cb.metric_name not in self._name:
                    self._name.append(cb.metric_name)
        return self._name
        return True

    # @property
    # def elapsed_mins(self):
    #     for cb in self.cbs:
    #         if cb and hasattr(cb, 'elapsed_mins'):
    #             _return = cb.elapsed_mins
    #             return _return
    #     return True

    @property
    def elapsed_mins(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'elapsed_mins') and hasattr(cb, 'metric_name'):
                self._elapsed[cb.metric_name] = cb.elapsed_mins
        return self._elapsed
        return True