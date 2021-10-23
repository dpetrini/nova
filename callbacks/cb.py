# Interface class to fill all methods, to be inherited by all
# Eventually put more private vars?

class Callbacks():
    """
        Pattern for callbacks. Maybe turn into an abstract class???
        Also: create class vars to share among objects?
    """
    #epochs = 0      # class variable

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

    def after_step_val(self, *args):
        return True

    def after_train_val(self, *args):
        return True
    
    # def best_metric_epoch(self, *args):
    #     return True
    
    # def best_metric(self, *args):
    #     return True

    # def loss_plot(self, *args):
    #     return True

    # def metric_plot(self, *args):
    #     return True

    # def best_model_file(self, *args):
    #     return True

    # def metric_name(self, *args):
    #     return True

    # def elapsed_mins(self, *args):
    #     return True