import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration,\
    GPTJForCausalLM, GPTNeoForCausalLM, \
    AutoModelForCausalLM, BitsAndBytesConfig


class T5ForTextStyleTransfer(pl.LightningModule):
    """
    Class for the T5 Transformer LightningModule.
    """

    def __init__(self, model_name, learning_rate, weight_decay, batch_size,
                 gpu_device, train_dataset, val_dataset, test_dataset, to_log):
        """
        Initialize model class.
        :param model_name: name of the model
        :type model_name: str
        :param learning_rate: learning rate
        :type learning_rate: float
        :param weight_decay: weight decay
        :type weight_decay: float
        :param batch_size: batch size
        :type batch_size: int
        :param gpu_device: gpu device for the model
        :type gpu_device: str
        :param train_dataset: train dataset
        :type train_dataset: SequenceDataset
        :param val_dataset: val dataset
        :type val_dataset: SequenceDataset
        :param test_dataset: test dataset
        :type test_dataset: SequenceDataset
        :param to_log: whether to log info to a logger
        :type to_log: bool
        """

        super(T5ForTextStyleTransfer, self).__init__()

        assert model_name in ['t5-small', 't5-base', 't5-large',
                              'flan-t5-small', 'flan-t5-base', 'flan-t5-large']

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gpu_device = gpu_device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.to_log = to_log

        if 'flan' in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f'google/{self.model_name}')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def train_dataloader(self):
        """
        Create dataloader for the training dataset.
        """

        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=40)

    def val_dataloader(self):
        """
        Create dataloader for the validation dataset.
        """

        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def test_dataloader(self):
        """
        Create dataloader for the test dataset.
        """

        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def predict(self, dataset_type, tokenizer, max_length=20):
        """
        Generate predictions for the specified dataset type.
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer: tokenizer
        :type tokenizer: AutoTokenizer
        :param max_length: maximum length of the generated sentences
        :type max_length: int
        :return: ground truth labels, predicted labels, and accuracy
        :type: tuple(list, list, float)
        """

        if dataset_type == 'train':
            predict_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        elif dataset_type == 'val':
            predict_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        else:
            predict_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        originals = []
        predictions = []
        ground_truths = []

        self.model.to(self.gpu_device)

        with torch.no_grad():
            for i, predict_batch in enumerate(tqdm(predict_dataloader, total=len(predict_dataloader))):
                data_x_input_ids = predict_batch[0].to(self.gpu_device)
                data_x_attention_mask = predict_batch[1].to(self.gpu_device)
                data_x_wo_prefix_input_ids = predict_batch[2].to(self.gpu_device)
                data_y_input_ids = predict_batch[3].to(self.gpu_device)

                output_ids = self.model.generate(input_ids=data_x_input_ids,
                                                 attention_mask=data_x_attention_mask,
                                                 max_new_tokens=max_length,
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 num_return_sequences=1,
                                                 no_repeat_ngram_size=2,
                                                 repetition_penalty=2.0)

                for inp, out_gt, out_pred in zip(data_x_wo_prefix_input_ids, data_y_input_ids, output_ids):
                    inp_s = tokenizer.decode(inp, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    originals.append(inp_s)

                    out_gt_s = tokenizer.decode(out_gt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    ground_truths.append(out_gt_s)

                    out_pred_s = tokenizer.decode(out_pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    predictions.append(out_pred_s)

        return originals, ground_truths, predictions


class LlamaForTextStyleTransfer():
    """
    Class for the Llama-based model for TST.
    """

    def __init__(self, model_name, batch_size,
                 train_dataset, val_dataset, test_dataset, to_log):
        """
        Initialize model class.
        :param model_name: name of the model
        :type model_name: str
        :param batch_size: batch size
        :type batch_size: int
        :param train_dataset: train dataset
        :type train_dataset: SequenceDataset
        :param val_dataset: val dataset
        :type val_dataset: SequenceDataset
        :param test_dataset: test dataset
        :type test_dataset: SequenceDataset
        :param to_log: whether to log info to a logger
        :type to_log: bool
        """

        super(LlamaForTextStyleTransfer, self).__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.to_log = to_log

        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16,
                                                 bnb_4bit_quant_type='nf4')

        self.model = AutoModelForCausalLM.from_pretrained(f'meta-llama/{model_name}',
                                                          device_map='cuda:0',
                                                          quantization_config=quantization_config)

    def train_dataloader(self):
        """
        Create dataloader for the training dataset.
        """

        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=40)

    def val_dataloader(self):
        """
        Create dataloader for the validation dataset.
        """

        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def test_dataloader(self):
        """
        Create dataloader for the test dataset.
        """

        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def predict(self, dataset_type, tokenizer, max_length=20):
        """
        Generate predictions for the specified dataset type.
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer: tokenizer
        :type tokenizer: AutoTokenizer
        :param max_length: maximum length of the generated sentences
        :type max_length: int
        :return: ground truth labels, predicted labels, and accuracy
        :type: tuple(list, list, float)
        """

        if dataset_type == 'train':
            predict_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        elif dataset_type == 'val':
            predict_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        else:
            predict_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)

        originals = []
        predictions = []
        ground_truths = []

        gpu_device = 'cuda:0'

        with torch.no_grad():
            for i, predict_batch in enumerate(tqdm(predict_dataloader, total=len(predict_dataloader))):
                data_x_input_ids = predict_batch[0].to(gpu_device)
                data_x_attention_mask = predict_batch[1].to(gpu_device)
                data_x_wo_prefix_input_ids = predict_batch[2].to(gpu_device)
                data_y_input_ids = predict_batch[3].to(gpu_device)

                output_ids = self.model.generate(input_ids=data_x_input_ids,
                                                 attention_mask=data_x_attention_mask,
                                                 max_new_tokens=max_length,
                                                 pad_token_id=tokenizer.bos_token_id,
                                                 num_return_sequences=1,
                                                 no_repeat_ngram_size=2,
                                                 repetition_penalty=2.0,
                                                 do_sample=False,
                                                 top_p=0.5)

                for inp_prompt, inp_orig, out_gt, out_pred in zip(data_x_input_ids, data_x_wo_prefix_input_ids,
                                                                  data_y_input_ids, output_ids):
                    inp_s = tokenizer.decode(inp_orig, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    originals.append(inp_s)

                    out_gt_s = tokenizer.decode(out_gt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    ground_truths.append(out_gt_s)

                    out_pred_s = tokenizer.decode(out_pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    inp_p = tokenizer.decode(inp_prompt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    out_pred_s = out_pred_s[len(inp_p):]
                    predictions.append(out_pred_s)

        return originals, ground_truths, predictions


class GPTForTextStyleTransfer(pl.LightningModule):
    """
    Class for the GPT LightningModule.
    """

    def __init__(self, model_name, learning_rate, weight_decay, batch_size,
                 gpu_device, train_dataset, val_dataset, test_dataset, to_log):
        """
        Initialize model class.
        :param model_name: name of the model
        :type model_name: str
        :param learning_rate: learning rate
        :type learning_rate: float
        :param weight_decay: weight decay
        :type weight_decay: float
        :param batch_size: batch size
        :type batch_size: int
        :param gpu_device: gpu device for the model
        :type gpu_device: str
        :param train_dataset: train dataset
        :type train_dataset: SequenceDataset
        :param val_dataset: val dataset
        :type val_dataset: SequenceDataset
        :param test_dataset: test dataset
        :type test_dataset: SequenceDataset
        :param to_log: whether to log info to a logger
        :type to_log: bool
        """

        super(GPTForTextStyleTransfer, self).__init__()

        assert model_name in ['gpt-j-6b', 'gpt-neo-1.3b']

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gpu_device = gpu_device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.to_log = to_log

        if 'j' in model_name:
            self.model = GPTJForCausalLM.from_pretrained(f'EleutherAI/{self.model_name}')
        else:
            self.model = GPTNeoForCausalLM.from_pretrained(f'EleutherAI/{self.model_name}')

    def train_dataloader(self):
        """
        Create dataloader for the training dataset.
        """

        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=40)

    def val_dataloader(self):
        """
        Create dataloader for the validation dataset.
        """

        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def test_dataloader(self):
        """
        Create dataloader for the test dataset.
        """

        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=40)

    def predict(self, dataset_type, tokenizer, max_length=20):
        """
        Generate predictions for the specified dataset type.
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer: tokenizer
        :type tokenizer: AutoTokenizer
        :param max_length: maximum length of the generated sentences
        :type max_length: int
        :return: ground truth labels, predicted labels, and accuracy
        :type: tuple(list, list, float)
        """

        if dataset_type == 'train':
            predict_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        elif dataset_type == 'val':
            predict_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        else:
            predict_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                            shuffle=False, num_workers=40)
        originals = []
        predictions = []
        ground_truths = []

        self.model.to(self.gpu_device)

        with torch.no_grad():
            for i, predict_batch in enumerate(tqdm(predict_dataloader, total=len(predict_dataloader))):
                data_x_input_ids = predict_batch[0].to(self.gpu_device)
                data_x_attention_mask = predict_batch[1].to(self.gpu_device)
                data_x_wo_prefix_input_ids = predict_batch[2].to(self.gpu_device)
                data_y_input_ids = predict_batch[3].to(self.gpu_device)

                output_ids = self.model.generate(input_ids=data_x_input_ids,
                                                 attention_mask=data_x_attention_mask,
                                                 max_new_tokens=max_length,
                                                 pad_token_id=tokenizer.eos_token_id,
                                                 num_return_sequences=1,
                                                 no_repeat_ngram_size=2,
                                                 repetition_penalty=2.0,
                                                 temperature=0)

                for inp_prompt, inp_orig, out_gt, out_pred in zip(data_x_input_ids, data_x_wo_prefix_input_ids,
                                                                  data_y_input_ids, output_ids):
                    inp_s = tokenizer.decode(inp_orig, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    originals.append(inp_s)

                    out_gt_s = tokenizer.decode(out_gt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    ground_truths.append(out_gt_s)

                    out_pred_s = tokenizer.decode(out_pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    inp_p = tokenizer.decode(inp_prompt, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    out_pred_s = out_pred_s[len(inp_p):]
                    predictions.append(out_pred_s)

        return originals, ground_truths, predictions
