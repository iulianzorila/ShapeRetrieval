class IOStream():
    def __init__(self, path):
        self.intestation_columns = ['epoch','lr','train_loss','val_loss','best_val_loss']
        self.f = open(path, 'a')
        self.log_intestation()

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def log_intestation(self):
      for col in self.intestation_columns:
        self.f.write(col)
        self.f.write('\t')
      self.f.write('\n')
      self.f.flush()

    def log_training(self, values):
      assert len(values) == len(self.intestation_columns)
      for v in values:
        self.f.write("{0:.6f}".format(v))
        self.f.write('\t')
      self.f.write('\n')
      self.f.flush()

    def close(self):
        self.f.close()