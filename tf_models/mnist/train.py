from tf_models.base.train import BaseTrain
from tqdm import tqdm
import numpy as np

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, options):
        super(Trainer, self).__init__(sess, model, data, options)

    def train_epoch(self):
        loop = tqdm(range(self.options.num_iter_per_epoch))
        losses=[]
        accs=[]
        for it in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss=np.mean(losses)
        acc=np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        print("accuracy={acc}, loss={loss}".format(acc=acc, loss=loss))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.options.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
