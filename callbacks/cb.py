# Interface class to fill all methods, to be inherited by all
# Eventually put more private vars?

class Callbacks():
    def __init__(self):
        pass

    def begin_train_val(self, epochs, *args):
        self.epochs = epochs
        return True

    def begin_epoch(self, *args):
        return True

    def begin_batch(self, *args):
        return False

    def begin_val(self, *args):
        return True

    def update_loss(self, *args, **kwargs):
        return False

    def update_LR(self, *args, **kwargs):
        return False

    def after_epoch(self, *args, **kwargs):
        return True

    def after_step(self, *args):   # train
        return True

    def begin_val(self, *args):
        return True

    def after_step_val(self, *args):
        return True

    def after_train_val(self, *args):
        return True
