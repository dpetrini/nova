# Handler class for call other callback classes


class CallbackHandler():

    def __init__(self, cbs=None):
        print('\nRegistered Callbacks:', ' ', end='')
        for cb in cbs:
            print(cb.__class__.__name__, ' ', end='')
        print()
        self.cbs = cbs if cbs else []

    def begin_train_val(self, epochs, train_dataloader, val_dataloader, bs_size):
        for cb in self.cbs:
            cb.begin_train_val(epochs, train_dataloader, val_dataloader, bs_size)
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

    # # Workaround para passar modelo

    def get_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'get_model'):
                model = cb.get_model()
                return model
        return True

    def get_best_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'get_best_model'):
                best_model = cb.get_best_model()
                return best_model
        return True

    def get_best_auc_model(self):
        for cb in self.cbs:
            if cb and hasattr(cb, 'get_best_auc_model'):
                best_model = cb.get_best_auc_model()
                return best_model
        return True
