# Handler class for call other callback classes


class CallbackHandler():

    def __init__(self, cbs=None):
        self.best_val_acc = 0.05
        print('\nRegistered Callbacks:', ' ', end='')
        for cb in cbs:
            print(cb.__class__.__name__, ' ', end='')
        print()
        self.cbs = cbs if cbs else []

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
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_metric_epoch'):
                _best_metric_ep = cb.best_metric_epoch
                return _best_metric_ep
        return True

    @property
    def best_metric(self):
        """
        Validation:
        If callback has best_metric variable will update it to outside.
        """
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_metric'):
                _best_metric = cb.best_metric
                return _best_metric
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

    @property
    def loss_plot(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'loss_plot'):
                _return = cb.loss_plot
                return _return
        return True

    @property
    def metric_plot(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'metric_plot'): # Usar append para multiplos (Acc & AUC)??
                _return = cb.metric_plot
                return _return
        return True

    @property
    def best_model_file(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'best_model_file'):
                _return = cb.best_model_file
                return _return
        return True

    @property
    def metric_name(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'metric_name'):
                _return = cb.metric_name
                return _return
        return True

    @property
    def elapsed_mins(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'elapsed_mins'):
                _return = cb.elapsed_mins
                return _return
        return True