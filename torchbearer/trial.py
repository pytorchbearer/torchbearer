class Trial:
    def __init__(self, model, criterion=None, optimizer=None, metrics=[], callbacks=[], pass_state=False):
        pass

    def with_train_generator(self, generator, steps=None):
        pass

    def with_train_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        pass

    def with_val_generator(self, generator, steps=None):
        pass

    def with_val_data(self, x, y, batch_size=1, shuffle=True, num_workers=1, steps=None):
        pass

    def with_test_generator(self, generator, steps=None):
        pass

    def with_test_data(self, x, batch_size=1, num_workers=1, steps=None):
        pass

    def with_split(self, split, val_steps=None):
        pass

    def fit(self, epochs=1, verbose=2):
        pass

    def evaluate(self, verbose=2):
        pass

    def predict(self, verbose=2):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, *args, **kwargs):
        pass

    def cuda(self, device=None):
        pass

    def cpu(self):
        pass

    def state_dict(self, **kwargs):
        pass

    def load_state_dict(self, state_dict, **kwargs):
        pass