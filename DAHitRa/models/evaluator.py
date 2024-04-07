import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils

from skimage.filters import threshold_otsu
import cv2

# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)
        self.model_str = args.net_G

        # Which difference block style to use
        self.diff_block = args.diff_block
        
        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        fig, axes = plt.subplots(4, 1, figsize=(24, 16))
        # if np.mod(self.batch_id, 1) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']), pad_value=1.0, padding=5)
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']), pad_value=1.0, padding=5)

        vis_pred = self._visualize_pred().detach().cpu().numpy()
        # thresh = threshold_otsu(vis_pred)
        # vis_pred = vis_pred > thresh
        vis_pred = utils.make_numpy_grid(torch.tensor(vis_pred), pad_value=5, padding=5)

        vis_gt = utils.make_numpy_grid(self.batch['L'], pad_value=5, padding=5)
        #vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #vis = np.clip(vis, a_min=0.0, a_max=1.0)
        
        axes[0].imshow(vis_input[:, :, :3])
        axes[0].set_title("Pre-Disaster")
        axes[0].axis('off')
        
        axes[1].imshow(vis_input2[:, :, :3])
        axes[1].set_title('Post-Disaster')
        axes[1].axis('off')
        
        cmap = ListedColormap(['#000000', '#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ffffff'])
        pred = axes[2].imshow(vis_pred[:, :, 0], cmap=cmap, vmin=0, vmax=5)
        gt = axes[3].imshow(vis_gt[:, :, 0], cmap=cmap, vmin=0, vmax=5)

        axes[2].axis('off')
        axes[2].set_title('Damage Extent Prediction')
        axes[3].axis('off')
        axes[3].set_title('Damage Extent Ground Truth')
        
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        #plt.imsave(file_name, vis)
        fig.savefig(file_name, dpi=100)
        plt.close()


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        # np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2, diff_block=self.diff_block)
        self.G_final_pred = self.G_pred
        
    def _forward_pass_attr(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        attributes = batch['C'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2, attributes, self.diff_block)
        self.G_final_pred = self.G_pred
        
    #def _forward_pass(self, batch):
    #    self.batch = batch
    #    img_in1 = batch['A'].to(self.device)
    #    img_in2 = batch['B'].to(self.device)
        
    #    if  self.model_str == "changeFormerV6":
    #        self.G_pred = self.net_G(img_in1, img_in2)[-1]
    #    else:
    #        self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                if self.diff_block == 0 or self.diff_block == 2:
                    self._forward_pass(batch)
                else:
                    self._forward_pass_attr(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
