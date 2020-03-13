import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            ##Change the argument in step depending on the lr_scheduler used
            v_log = list(val_log.items())
            self.lr_scheduler.step(v_log[0][1])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class deltaTrainer(BaseTrainer):
    """
    deltaTrainer class. For the Delta Model
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        #DataLoader returns ref and trans image, as well as trans labels
        for batch_idx, (ref_img,trans_img, target) in enumerate(self.data_loader):

            ref_img, trans_img, target = ref_img.to(self.device),trans_img.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            ## Difference ##
            ref_output = self.model(ref_img)
            trans_output = self.model(trans_img)
            loss = self.criterion(trans_output - ref_output, target)
            ################
            loss = torch.tensor([1.,1.,1.,1.])*loss

            loss.backward(torch.ones_like(loss))
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', torch.mean(loss).item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(trans_output - ref_output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    torch.mean(loss).item()))
                self.writer.add_image('input', make_grid(ref_img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            ##Change the argument in step depending on the lr_scheduler used
            v_log = list(val_log.items())
            print(val_log)
            self.lr_scheduler.step(v_log[0][1])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            offset_image = self.data_loader.ref_image
            offset_label = self.data_loader.ref_label
            offset = offset_label - self.model(offset_image.unsqueeze(0)).squeeze(0)
            for batch_idx, (ref_image,_, ref_target) in enumerate(self.valid_data_loader):
                ref_image, ref_target = ref_image.to(self.device), ref_target.to(self.device)

                ## Difference ##
                ref_output = self.model(ref_image)
                loss = self.criterion(ref_output + offset,ref_target)
                #loss = self.criterion(trans_output - ref_ouptut, target)
                ################

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', torch.mean(loss).item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met( ref_output + offset, ref_target))
                self.writer.add_image('input', make_grid(ref_image.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class deltaRefTrainer(BaseTrainer):
    """
    deltaTrainer class. For the Delta Model
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        #DataLoader returns ref and trans image, as well as trans labels
        for batch_idx, (ref_img,trans_img, target) in enumerate(self.data_loader):

            ref_img, trans_img, target = ref_img.to(self.device),trans_img.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            ## Difference ##
            ref_output = self.model(ref_img)
            trans_output = self.model(trans_img)

            lambda_target = trans_output[:,2]/ref_output[:,2]
            ref_rot = ref_output[:,3]
            trans_rot = trans_output[:,3]
            rots_target = torch.asin(trans_rot) - torch.asin(ref_rot)

            ref_center = ref_output[:,:2]
            trans_center = trans_output[:,:2]
            center = torch.tensor([32,32])
            r_mat = torch.cat([torch.cos(rots_target),-torch.sin(rots_target),torch.sin(rots_target),torch.cos(rots_target)],dim=0)
            r_mat = r_mat.view(ref_center.size()[0],2,2)
            translation_target = torch.ones((ref_center.size()[0],2))
            for idx in range(ref_center.size()[0]):
                rotation_prod = torch.mv(r_mat[idx],ref_center[idx] - center)
                translation_target[idx] = trans_center[idx] - (lambda_target[idx] * rotation_prod) + center


            
            delta_pred = torch.cat([translation_target[:,0].unsqueeze(1),translation_target[:,1].unsqueeze(1),lambda_target.unsqueeze(1),torch.sin(rots_target).unsqueeze(1)],dim=1)
            loss = self.criterion(delta_pred, target)
            ################ Scaling the scale loss so the network learns that better
            loss = torch.tensor([1.,1.,1.,1.])*loss

            loss.backward(torch.ones_like(loss))
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', torch.mean(loss).item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(trans_output - ref_output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    torch.mean(loss).item()))
                self.writer.add_image('input', make_grid(ref_img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            ##Change the argument in step depending on the lr_scheduler used
            v_log = list(val_log.items())
            print(val_log)
            self.lr_scheduler.step(v_log[0][1])
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            offset_image = self.data_loader.ref_image
            offset_label = self.data_loader.ref_label
            offset = offset_label - self.model(offset_image).squeeze()
            for batch_idx, (trans_image, trans_target) in enumerate(self.valid_data_loader):
                trans_image, trans_target = trans_image.to(self.device), trans_target.to(self.device)

                ## Difference ##
                trans_output = self.model(trans_image)

                center = torch.tensor([32,32])
                trans_scale = trans_output[2] * offset_label[2]
                trans_rot = torch.asin(trans_output[3]) - torch.asin(offset_label[3])
                trans_translation = trans_output[:2] - center + trans_output[2]*torch.dot(torch.tensor([[torch.cos(trans_rot),-torch.sin(trans_rot)],[torch.sin(trans_rot),torch.cos(trans_rot)]]),offset_label[:2] - center)

                target_pred = torch.tensor([trans_translation[0],trans_translation[1],trans_scale,trans_rot])
                loss = self.criterion(target_pred,trans_target)
                #loss = self.criterion(trans_output - ref_ouptut, target)
                ################

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', torch.mean(loss).item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met( ref_output + offset, ref_target))
                self.writer.add_image('input', make_grid(ref_image.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


