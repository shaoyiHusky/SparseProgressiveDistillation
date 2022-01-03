from bert_spd import BertEncoder


class ConstantReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, replacing_rate, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.replacing_steps is None or self.replacing_rate == 1.0:
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                self.bert_encoder.set_replacing_rate(1.0)
                self.replacing_rate = 1.0
            return self.replacing_rate


class LinearReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, base_replacing_rate, k):
        self.bert_encoder = bert_encoder
        self.base_replacing_rate = base_replacing_rate
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(base_replacing_rate)

    def step(self):
        self.step_counter += 1
        current_replacing_rate = min(self.k * self.step_counter + self.base_replacing_rate, 1.0)
        print('step_counter: ', self.step_counter, 'replacing_rate: ', current_replacing_rate)
        self.bert_encoder.set_replacing_rate(current_replacing_rate)
        return current_replacing_rate


class MixedReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, replacing_rate, k, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.step_counter < self.replacing_steps or self.replacing_rate == 1.0:
            print('step_counter: ', self.step_counter, 'replacing_rate: ', self.replacing_rate)
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                current_replacing_rate = min(self.k * (self.step_counter - self.replacing_steps) + self.replacing_rate, 1.0)
                self.bert_encoder.set_replacing_rate(current_replacing_rate)
                print('step_counter: ', self.step_counter, 'current_replacing_rate: ', current_replacing_rate)
                return current_replacing_rate


class ConstantThenLinearReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, replacing_rate, base_replacing_rate, k, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.base_replacing_rate = base_replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.step_counter < self.replacing_steps or self.replacing_rate == 1.0:
            print('step_counter: ', self.step_counter, 'replacing_rate: ', self.replacing_rate)
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                current_replacing_rate = min(self.k * (self.step_counter - self.replacing_steps) + self.base_replacing_rate, 1.0)
                self.bert_encoder.set_replacing_rate(current_replacing_rate)
                print('step_counter: ', self.step_counter, 'current_replacing_rate: ', current_replacing_rate)
                return current_replacing_rate


class CustomizedLinearReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, replacing_rate, k, constant_replacing_rate, constant_replacing_step):
        self.bert_encoder = bert_encoder
        self.constant_replacing_rate = constant_replacing_rate
        self.constant_replacing_step = constant_replacing_step
        self.base_replacing_rate = replacing_rate
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(self.constant_replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.step_counter < self.constant_replacing_step:
            print('step_counter: ', self.step_counter, 'replacing_rate: ', self.constant_replacing_rate)
            return self.constant_replacing_rate
        else:
            current_replacing_rate = min(self.k * (self.step_counter - self.constant_replacing_step) + self.base_replacing_rate, 1.0)
            print('step_counter: ', self.step_counter, 'replacing_rate: ', current_replacing_rate)
            self.bert_encoder.set_replacing_rate(current_replacing_rate)
            return current_replacing_rate
