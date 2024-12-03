import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report as re_cls_report
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
import math

from utilities.metrics import eval_result

import numpy as np
import nni

class BertVitReTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, ttt_data=None, re_dict=None,
                 model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        # 初始化训练数据、验证数据和测试数据
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.ttt_data = ttt_data

        # 初始化映射字典 关系映射
        self.re_dict = re_dict

        # 初始化模型对象
        self.model = model
        # 初始化处理函数或对象
        self.process = process
        # 初始化日志记录器对象
        self.logger = logger
        # 初始化用于写入日志或其他信息的对象
        self.writer = writer
        # 设置模型训练时的刷新步骤间隔
        self.refresh_step = 2

        # 初始化最佳验证集指标（准确率、微平均F1分数、微平均召回率和微平均精准率）
        self.best_dev_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        # 初始化最佳测试集指标
        self.best_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        # 初始化最终测试集指标
        self.final_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_dev_epoch = None
        self.best_test_epoch = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        if self.test_data is not None:
            self.test_num_steps = len(self.test_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.pbar = None
        self.re_optimizer = None
        self.re_scheduler = None
        self.before_train()

        if self.args.mismatch:
            self.best_test_epoch_match = None
            self.best_test_epoch_mismatch = None
            self.best_mismatch_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
            self.final_mismatch_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
            self.best_match_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
            self.final_match_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}

    def train(self):
        self.step = 0
        self.model.train()
        # 记录训练开始的信息，包括训练数据数量、训练轮数、批次大小、学习率等
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        if self.args.do_test:
            self.logger.info("***** Start testing without training *****")
            self.test(0)
            return

        if self.args.TTA:
            self.logger.info("***** Start test time adaptation. *****")
            self.TTA()
            self.test(0)
            return
        if self.args.TTT:
            self.logger.info("***** Start test time Training . *****")
            self.TTT(0)
            return

        if self.args.do_metric_un:
            self.logger.info("***** Start uncertainty metric without training *****")
            epidemic_uncertainty = self.metric_epidemic_uncertainty(method=self.args.metric_method)
            self.logger.info(f"{self.args.metric_method} epidemic uncertainty {epidemic_uncertainty}")
            print(f"{self.args.metric_method} epidemic uncertainty {epidemic_uncertainty}")
            return

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            re_avg_loss = 0.0
            for epoch in range(1, self.args.num_epochs + 1):
            # for epoch in range(1, 13):
                if self.args.uncertainty_stat:
                    self.no_shuffle_lst, self.shuffle_lst, self.uncertainty = [], [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    if self.args.augment:
                        re_batch = (torch.flatten(tup, start_dim=0, end_dim=1).squeeze().to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    else:
                        re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (re_loss, re_logits), labels, _ = self._step(re_batch,
                                                                 mode="train",
                                                                 task='re',
                                                                 epoch=epoch)
                    # print(re_loss)
                    re_avg_loss += re_loss.detach().cpu().item()
                    re_loss.backward()
                    self.re_optimizer.step()
                    self.re_optimizer.zero_grad()
                    self.re_scheduler.step()

                    if self.step % self.refresh_step == 0:
                        re_avg_loss = float(re_avg_loss) / self.refresh_step
                        print_output = "RE loss:{:<6.5f}".format(re_avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        re_avg_loss = 0

                # 在达到评估开始轮次后，进行评估和测试
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
                    # if epoch >= 10:
                    #     self.TTA()
                    self.test(epoch)

            pbar.close()
            self.pbar = None

            # 记录最佳验证集性能、最佳测试集性能和最终测试集性能
            self.logger.info("Get best dev performance at epoch {}, "
                             "best dev f1 is {}".format(self.best_dev_epoch,
                                                        self.best_dev_metrics['micro_f1'],
                                                        ))

            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 is {}".format(self.best_test_epoch,
                                            self.best_test_metrics['micro_f1'],
                                            ))
            # 最佳验证集性能
            self.logger.info(
                "Get final test performance according to validation results at epoch {}, "
                "final f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_dev_epoch,
                    self.final_test_metrics['micro_f1'],
                    self.final_test_metrics['micro_r'],
                    self.final_test_metrics['micro_p'],
                    self.final_test_metrics['acc']))

            # 最佳测试集性能
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_test_epoch,
                    self.best_test_metrics['micro_f1'],
                    self.best_test_metrics['micro_r'],
                    self.best_test_metrics['micro_p'],
                    self.best_test_metrics['acc']))

            # 最佳测试集性能
            if self.args.mismatch:
                self.logger.info(
                    "Get best match test performance at epoch {}, "
                    "best test f1 {}, "
                    "recall {}, "
                    "precision {}, "
                    "acc {}".format(
                        self.best_test_epoch_match,
                        self.best_match_metrics['micro_f1'],
                        self.best_match_metrics['micro_r'],
                        self.best_match_metrics['micro_p'],
                        self.best_match_metrics['acc']))

                self.logger.info(
                    "Get best mismatch test performance at epoch {}, "
                    "best test f1 {}, "
                    "recall {}, "
                    "precision {}, "
                    "acc {}".format(
                        self.best_test_epoch_mismatch,
                        self.best_mismatch_metrics['micro_f1'],
                        self.best_mismatch_metrics['micro_r'],
                        self.best_mismatch_metrics['micro_p'],
                        self.best_mismatch_metrics['acc']))
        if self.args.nni:
            # nni.report_intermediate_result({'f1_score_mismatch': float(self.best_mismatch_metrics['micro_f1']),
            #                                 'f1_score_match': float(self.best_match_metrics['micro_f1'])})
            nni.report_final_result({'default': float(self.best_test_metrics['micro_f1'])})


    def evaluate(self, epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                step = 0
                re_true_labels, re_pred_labels = [], [] # Lists to store true and predicted labels.
                total_loss = 0
                # Initialize tensor to track hit rates for different ranks and relations.
                hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.dev_data:
                    step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, _ = self._step(re_batch,
                                                           mode="dev",
                                                           task='re',
                                                           epoch=epoch,)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    # Get predictions by selecting the class with the highest logit.
                    re_preds = logits.argmax(-1)
                    # Extend the lists with true and predicted labels.
                    re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    re_pred_labels.extend(re_preds.view(-1).detach().cpu().tolist())
                    # Calculate the rank of the true label in the predicted logits.
                    re_pred_ranks = 1 + torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1, descending=False)[torch.arange(labels.shape[0], device=self.args.device), labels]
                    re_pred_ranks = re_pred_ranks.float()
                    # Update the hit rates for different ranks and relations.
                    for rel_id in range(len(self.re_dict)):
                        ranks = re_pred_ranks[labels == rel_id]
                        for k in range(len(self.re_dict)):
                            hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + hits[rel_id, k + 1]
                    pbar.update()
                # evaluate done
                pbar.close()
                # Generate and log classification report based on true and predicted labels.
                re_cls_result = re_cls_report(y_true=re_true_labels, y_pred=re_pred_labels,
                                              labels=list(self.re_dict.values())[1:],
                                              target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", re_cls_result)
                result = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)

                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                        .format(epoch, self.args.num_epochs, self.best_dev_metrics['micro_f1'],
                                self.best_dev_epoch,
                                result['micro_f1'], ))
                if result['micro_f1'] >= self.best_dev_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better dev performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metrics['micro_f1'] = result['micro_f1']  # update best metric
                    self.best_dev_metrics['micro_r'] = result['micro_r']
                    self.best_dev_metrics['micro_p'] = result['micro_p']
                    self.best_dev_metrics['acc'] = result['acc']
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path)
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch=0):
        self.model.eval()
        self.logger.info(f"\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None and self.args.do_test:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        if self.args.uncertainty_stat:
            self.no_shuffle_lst, self.shuffle_lst, self.uncertainty = [], [], []

        # 初始化存储真实标签、预测标签、样本单词列表和样本图像 ID 的列表
        re_true_labels, re_pred_labels, sample_word_lists, sample_image_ids = [], [], [], []
        re_pred_labels_match, re_true_labels_match, re_pred_labels_mismatch, re_true_labels_mismatch = [], [], [], []
        re_pred_logits = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.test_data:
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    if self.args.write_path is not None and self.args.do_test:
                        outputs = self._step(re_batch, mode="test", task='re', epoch=epoch,)
                        if self.args.mismatch:
                            (loss, logits), labels, extend_word_lists, imgids, shuffle = outputs
                        else:
                            (loss, logits), labels, extend_word_lists, imgids = outputs
                    else:
                        outputs = self._step(re_batch, mode="test", task='re', epoch=epoch,)  # logits: batch, 3
                        if self.args.mismatch:
                            (loss, logits), labels, _, shuffle = outputs
                        else:
                            (loss, logits), labels, _ = outputs
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    re_pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    re_pred_logits.extend(logits.detach().cpu().tolist())
                    re_pred_ranks = 1 + torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1, descending=False)[
                        torch.arange(labels.shape[0], device=self.args.device), labels]
                    re_pred_ranks = re_pred_ranks.float()
                    for rel_id in range(len(self.re_dict)):
                        ranks = re_pred_ranks[labels == rel_id]
                        for k in range(len(self.re_dict)):
                            hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + hits[rel_id, k + 1]
                    if self.args.write_path is not None and self.args.do_test:
                        sample_word_lists.extend([*extend_word_lists])
                        sample_image_ids.extend([*imgids])

                    if self.args.mismatch:
                        preds_match = preds[shuffle == 1]
                        preds_mismatch = preds[shuffle == 0]
                        labels_match = labels[shuffle == 1]
                        labels_mismatch = labels[shuffle == 0]
                        re_true_labels_match.extend(labels_match.view(-1).detach().cpu().tolist())
                        re_pred_labels_match.extend(preds_match.view(-1).detach().cpu().tolist())
                        re_true_labels_mismatch.extend(labels_mismatch.view(-1).detach().cpu().tolist())
                        re_pred_labels_mismatch.extend(preds_mismatch.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                # 如果指定了写入路径且需要进行测试
                if self.args.write_path is not None and self.args.do_test:
                    # dictionary of lists
                    write_file_dict = {'sample_word_lists': sample_word_lists, 'sample_image_ids': sample_image_ids,
                                       'true_labels': re_true_labels, 'pred_labels': re_pred_labels,
                                       'pred_logits': re_pred_logits}
                    df = pd.DataFrame(write_file_dict)
                    # saving the dataframe
                    df.to_csv(self.args.write_path + '_' + 'test.csv')
                # scikit-learn 的 报告结果
                sk_result = re_cls_report(y_true=re_true_labels, y_pred=re_pred_labels,
                                          labels=list(self.re_dict.values()),
                                          target_names=list(self.re_dict.keys()), digits=4)
                self.logger.info("%s\n", sk_result)
                #评估结果
                result = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)

                ############
                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, " \
                        .format(epoch, self.args.num_epochs,
                                self.best_test_metrics['micro_f1'],
                                self.best_test_epoch,
                                result['micro_f1'], ))

                if epoch == self.best_dev_epoch:
                    if result['micro_f1'] > self.final_test_metrics['micro_f1']:
                        self.final_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                        self.final_test_metrics['micro_r'] = result['micro_r']
                        self.final_test_metrics['micro_p'] = result['micro_p']
                        self.final_test_metrics['acc'] = result['acc']

                if result['micro_f1'] >= self.best_test_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                    self.best_test_metrics['micro_r'] = result['micro_r']
                    self.best_test_metrics['micro_p'] = result['micro_p']
                    self.best_test_metrics['acc'] = result['acc']

                if self.args.mismatch:
                    self.logger.info("Get the match test performance.")
                    result_match = eval_result(re_true_labels_match, re_pred_labels_match, self.re_dict, self.logger)
                    self.logger.info("Get the mismatch test performance.")
                    result_mismatch = eval_result(re_true_labels_mismatch, re_pred_labels_mismatch, self.re_dict, self.logger)

                    if result_match['micro_f1'] >= self.best_match_metrics['micro_f1']:  # this epoch get best performance
                        self.logger.info("Get better test performance at epoch {}".format(epoch))
                        self.best_test_epoch_match = epoch
                        self.best_match_metrics['micro_f1'] = result_match['micro_f1']  # update best metric
                        self.best_match_metrics['micro_r'] = result_match['micro_r']
                        self.best_match_metrics['micro_p'] = result_match['micro_p']
                        self.best_match_metrics['acc'] = result_match['acc']

                    if result_mismatch['micro_f1'] >= self.best_mismatch_metrics['micro_f1']:  # this epoch get best performance
                        self.logger.info("Get better test performance at epoch {}".format(epoch))
                        self.best_test_epoch_mismatch = epoch
                        self.best_mismatch_metrics['micro_f1'] = result_mismatch['micro_f1']  # update best metric
                        self.best_mismatch_metrics['micro_r'] = result_mismatch['micro_r']
                        self.best_mismatch_metrics['micro_p'] = result_mismatch['micro_p']
                        self.best_mismatch_metrics['acc'] = result_mismatch['acc']

                if self.args.uncertainty_stat:
                    self.no_shuffle_lst = torch.cat(self.no_shuffle_lst, dim=0)
                    self.shuffle_lst = torch.cat(self.shuffle_lst, dim=0)
                    self.uncertainty = torch.cat(self.uncertainty, dim=0)
                    # self.no_shuffle_lst = torch.norm(self.no_shuffle_lst, dim=-1)
                    # self.shuffle_lst = torch.norm(self.shuffle_lst, dim=-1)
                    # self.uncertainty = torch.norm(self.uncertainty, dim=-1)

                    self.no_shuffle_lst = torch.abs(self.no_shuffle_lst).sum(dim=-1)
                    self.shuffle_lst = torch.abs(self.shuffle_lst).sum(dim=-1)
                    self.uncertainty = torch.abs(self.uncertainty).sum(dim=-1)

                    self.logger.info(f"mismatch sample's uncertainty median {torch.median(self.no_shuffle_lst).item()}")
                    self.logger.info(f"match sample's uncertainty median {torch.median(self.shuffle_lst).item()}")

                    self.logger.info(f"mismatch sample's uncertainty min {torch.min(self.no_shuffle_lst).item()}")
                    self.logger.info(f"match sample's uncertainty max {torch.max(self.shuffle_lst).item()}")

                    self.logger.info(f"Avg uncertainty {torch.mean(self.uncertainty).item()}")
                    self.logger.info(f"Median uncertainty {torch.median(self.uncertainty).item()}")
                    self.logger.info(f"Max uncertainty {torch.max(self.uncertainty).item()}")
                    self.logger.info(f"Min uncertainty {torch.min(self.uncertainty).item()}")

        self.model.train()

    def _step(self, batch, mode="train", task='re', epoch=0):
        if self.args.write_path is not None and mode == 'test' and self.args.do_test:
            if self.args.model_name not in ['bert-only-re']:
                input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids, shuffle = batch
            else:
                input_ids, token_type_ids, attention_mask, labels, extend_word_lists = batch
                images, aux_imgs, rcnn_imgs, imgids = None, None, None, None
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 task=task,
                                 epoch=epoch,
                                 uncertainty_stat=self.args.uncertainty_stat)
            if self.args.uncertainty_stat:
                uncertainty = outputs[-1]
                self.no_shuffle_lst.append(uncertainty[shuffle == 0])
                self.shuffle_lst.append(uncertainty[shuffle == 1])
                self.uncertainty.append(uncertainty)
                outputs = outputs[:2]
            if self.args.mismatch and mode in ['test']:
                return outputs, labels, extend_word_lists, imgids, shuffle
            return outputs, labels, extend_word_lists, imgids
        else:
            if self.args.model_name not in ['bert-only-re']:
                re_input_ids, re_token_type_ids, re_attention_mask, re_labels, images, aux_imgs, rcnn_imgs, shuffle = batch
            else:
                re_input_ids, re_token_type_ids, re_attention_mask, re_labels = batch
                images, aux_imgs, rcnn_imgs = None, None, None
            if task == 're':
                input_ids = re_input_ids
                token_type_ids = re_token_type_ids
                attention_mask = re_attention_mask
                labels = re_labels
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 task=task,
                                 epoch=epoch,
                                 uncertainty_stat=self.args.uncertainty_stat)

            if self.args.uncertainty_stat:
                outputs = outputs[:2]
                if mode == "test":
                    uncertainty = outputs[-1]
                    self.no_shuffle_lst.append(uncertainty[shuffle == 0])
                    self.shuffle_lst.append(uncertainty[shuffle == 1])
                    self.uncertainty.append(uncertainty)
            if self.args.mismatch and mode in ['test']:
                return outputs, labels, attention_mask, shuffle
            return outputs, labels, attention_mask

    def before_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            # Check if the parameter belongs to the model or starts with 're_classifier'.
            if 'model' in name or name.startswith('re_classifier'):
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)
        self.re_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.re_scheduler = get_linear_schedule_with_warmup(optimizer=self.re_optimizer,
                                                            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def metric_epidemic_uncertainty(self, method='energy'):
        if method in ['MC']:
            self.model.train()
        else:
            self.model.eval()
        self.logger.info(f"\n***** Running uncertainty Metric *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", 1)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        epidemic_uncertainty = []

        with torch.no_grad():
            # 这里batch size 要改为1
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Uncertainty")
                # total_loss = 0
                for batch in self.test_data:
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    preds = []
                    if method in ['MC']:
                        # 默认num_sample 为100.
                        for _ in range(self.args.num_samples):
                            if self.args.write_path is not None and self.args.do_test:
                                (loss, logits), labels, extend_word_lists, imgids = self._step(re_batch,
                                                                                               mode="test",
                                                                                               task='re',
                                                                                               epoch=0, )
                            else:
                                (loss, logits), labels, _ = self._step(re_batch,
                                                                       mode="test",
                                                                       task='re',
                                                                       epoch=0, )  # logits: batch, 3

                            # preds = logits.argmax(-1)
                            # preds = torch.softmax(logits, dim=-1)
                            pred = torch.softmax(logits, dim=-1)
                            preds.append(pred)
                        preds = torch.stack(preds)
                        epidemic_uncertainty.append(torch.var(preds).mean())

                    elif method in ['energy']:
                        if self.args.write_path is not None and self.args.do_test:
                            (loss, logits), labels, extend_word_lists, imgids = self._step(re_batch,
                                                                                           mode="test",
                                                                                           task='re',
                                                                                           epoch=0, )
                        else:
                            (loss, logits), labels, _ = self._step(re_batch,
                                                                   mode="test",
                                                                   task='re',
                                                                   epoch=0, )  # logits: batch, 3
                        uncertainty = -torch.logsumexp(logits * 8.0, dim=1)
                        epidemic_uncertainty.append(uncertainty)

                    pbar.update()
                # evaluate done
                pbar.close()
                epidemic_uncertainty = torch.cat(epidemic_uncertainty, dim=0)
                return epidemic_uncertainty.mean()

    def before_TTA(self):
        self.model = configure_model(self.model)
        params, param_names = collect_params(self.model)
        self.tta_optimizer = optim.AdamW([{'params': params, 'lr': 1e-6}],
                                             weight_decay=1e-2,
                                             betas=(0.9, 0.999))
        # optimizer_grouped_parameters = []
        # params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        # for name, param in self.model.named_parameters():
        #     # Check if the parameter belongs to the model or starts with 're_classifier'.
        #     if 'model' in name or name.startswith('re_classifier'):
        #         params['params'].append(param)
        # optimizer_grouped_parameters.append(params)
        # self.tta_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        # self.tta_scheduler = get_linear_schedule_with_warmup(optimizer=self.tta_optimizer,
        #                                             num_warmup_steps=self.args.warmup_ratio * self.test_num_steps,
        #                                             num_training_steps=self.test_num_steps)

    def TTA(self):
        self.before_TTA()
        # for _ in range(2):
        #     for batch in self.test_data:
        #         re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
        #         # 在这里修改
        #         (_, outputs), _, _ = self._step(re_batch, mode="train", task='re', )
        #         # adapt
        #         p_sum = outputs.softmax(dim=-1).sum(dim=-2)
        #         loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()
        #
        #         pred = outputs.softmax(dim=-1)
        #         pred_max = pred.max(dim=-1)[0]
        #         gamma = math.exp(-1.0)
        #         t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
        #         loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()
        #
        #         loss = 0.0 * loss_ra - 1.0 * loss_bal
        #
        #         loss.backward()
        #
        #         self.tta_optimizer.step()
        #         self.tta_optimizer.zero_grad()
        #     # self.tta_scheduler.step()
        for _ in range(1):
            with tqdm(total=len(self.ttt_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing time training")
                for batch in self.ttt_data:
                    # input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids, shuffle = batch
                    # re_batch =
                    re_batch = (tup.squeeze().to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                                batch)  # to cpu/cuda device

                    (_, logits), _, _ = self._step(re_batch,
                                                            mode="test",
                                                            task='re',
                                                            epoch=0,)  # logits: batch, 3

                    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
                    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.3)]
                    logits = logits[idx].mean(dim=0)

                    loss = avg_entropy(logits)

                    loss.backward()
                    self.tta_optimizer.step()
                    self.tta_optimizer.zero_grad()

                    pbar.update()
                pbar.close()

        self.model.requires_grad_(True)


    def TTT(self, epoch=0):
        self.model.eval()
        self.logger.info(f"\n***** Running testing  *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None and self.args.do_test:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        # 初始化存储真实标签、预测标签、样本单词列表和样本图像 ID 的列表
        re_true_labels, re_pred_labels, sample_word_lists, sample_image_ids = [], [], [], []
        re_pred_logits = []
        with torch.no_grad():
            with tqdm(total=len(self.ttt_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing time training")
                total_loss = 0
                hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.ttt_data:
                    # input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids, shuffle = batch
                    # re_batch =
                    re_batch = (tup.squeeze().to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                                batch)  # to cpu/cuda device
                    if self.args.write_path is not None and self.args.do_test:
                        (loss, logits), labels, extend_word_lists, imgids = self._step(re_batch,
                                                                                       mode="test",
                                                                                       task='re',
                                                                                       epoch=epoch,)
                    else:
                        (loss, logits), labels, _ = self._step(re_batch,
                                                               mode="test",
                                                               task='re',
                                                               epoch=epoch,)  # logits: batch, 3

                    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
                    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
                    logits = logits[idx].mean(dim=0).unsqueeze(0)
                    labels = labels[0].unsqueeze(0)
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    re_pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    re_pred_logits.extend(logits.detach().cpu().tolist())
                    re_pred_ranks = 1 + torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1, descending=False)[torch.arange(labels.shape[0], device=self.args.device), labels]
                    re_pred_ranks = re_pred_ranks.float()
                    for rel_id in range(len(self.re_dict)):
                        ranks = re_pred_ranks[labels == rel_id]
                        for k in range(len(self.re_dict)):
                            hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + hits[rel_id, k + 1]
                    if self.args.write_path is not None and self.args.do_test:
                        sample_word_lists.extend([*extend_word_lists])
                        sample_image_ids.extend([*imgids])
                    pbar.update()
                # evaluate done
                pbar.close()
                # 如果指定了写入路径且需要进行测试
                if self.args.write_path is not None and self.args.do_test:
                    # dictionary of lists
                    write_file_dict = {'sample_word_lists': sample_word_lists, 'sample_image_ids': sample_image_ids,
                                       'true_labels': re_true_labels, 'pred_labels': re_pred_labels,
                                       'pred_logits': re_pred_logits}
                    df = pd.DataFrame(write_file_dict)
                    # saving the dataframe
                    df.to_csv(self.args.write_path + '_' + 'test.csv')
                # scikit-learn 的 报告结果
                sk_result = re_cls_report(y_true=re_true_labels, y_pred=re_pred_labels,
                                          labels=list(self.re_dict.values()),
                                          target_names=list(self.re_dict.keys()), digits=4)
                self.logger.info("%s\n", sk_result)
                #评估结果
                result = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)
                ############
                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, " \
                        .format(epoch, self.args.num_epochs,
                                self.best_test_metrics['micro_f1'],
                                self.best_test_epoch,
                                result['micro_f1'], ))

                if epoch == self.best_dev_epoch:
                    if result['micro_f1'] > self.final_test_metrics['micro_f1']:
                        self.final_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                        self.final_test_metrics['micro_r'] = result['micro_r']
                        self.final_test_metrics['micro_p'] = result['micro_p']
                        self.final_test_metrics['acc'] = result['acc']

                if result['micro_f1'] >= self.best_test_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metrics['micro_f1'] = result['micro_f1']  # update best metric
                    self.best_test_metrics['micro_r'] = result['micro_r']
                    self.best_test_metrics['micro_p'] = result['micro_p']
                    self.best_test_metrics['acc'] = result['acc']

        self.model.train()

def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if 're_classifier' in nm:
        # if 'vision_and_text_k_proj' in nm:
        # if 'cross_modal_att_layer' in nm:
        # if 'cross_modal_att_layer' in nm or 're_classifier' in nm:
            m.requires_grad_(True)

    return model

def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion = []
    names_fusion = []

    for nm, m in model.named_modules():
        if 're_classifier' in nm:
        # if 'vision_and_text_k_proj' in nm:
        # if 'cross_modal_att_layer' in nm:
        # if 'cross_modal_att_layer' in nm or 're_classifier' in nm:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion.append(p)
                    names_fusion.append(f"{nm}.{np}")

    return params_fusion, names_fusion

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
