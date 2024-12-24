import logging
import sys

import torch
# from fairlib.src.networks.augmentation_layer import Augmentation_layer # 如果不需要augmentation，可以注释掉
# from fairlib.src.networks.utils import BaseModel # 注释掉BaseModel，因为我们不用它了
from sklearn.metrics import accuracy_score

# from fairlib.src.networks import get_main_model, MLP # 注释掉MLP
from fairlib.src.utils.utils import seed_everything
from fairlib.src import base_options
from fairlib.src import dataloaders
import os
import sys
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers # 导入 transformers

class dummy_args:
    def __init__(self, dataset, data_dir):
        # Creating objects
        self.dataset = dataset
        self.data_dir = data_dir
        self.regression = False
        self.GBT = False
        self.BT = None
        # self.BT = BT
        # self.BTObj = BTObj
        self.adv_BT = None
        self.adv_decoupling = False
        self.encoder_architecture = "Fixed"
        self.emb_size = 2304
        self.num_classes = 2
        self.batch_size = 4


class TransformerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer = transformers.AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.args.num_classes,
        )
        self.transformer.to(self.device)
        self.optimizer = torch.optim.AdamW(self.transformer.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask): #接受input_ids,attention_mask作为输入
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits



def eval_epoch(model, iterator, device, tokenizer): # 添加 tokenizer 参数
    epoch_loss = 0

    model.eval()

    criterion = model.criterion

    preds = []
    labels = []

    for batch in iterator:

        text = batch[0] # 获取文本数据
        p_tags = batch[2] # 获取标签数据

        text = text.to(device) if isinstance(text, torch.Tensor) else text  # 如果text已经是tensor则直接.to(device)
        p_tags = p_tags.to(device).long()

        if isinstance(text, list):  # 处理 text 为列表的情况
            encoded_inputs = [tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device) for t in text]
            predictions = [model(encoded_input["input_ids"], encoded_input["attention_mask"]).detach().cpu() for
                           encoded_input in encoded_inputs]
            predictions = torch.stack(predictions).mean(dim=0)

        else:
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            predictions = model(encoded_input["input_ids"], encoded_input["attention_mask"]).detach().cpu()

        loss = criterion(predictions, p_tags.cpu())
        epoch_loss += loss.item()
        p_tags = p_tags.cpu().numpy()
        preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(p_tags)

    return (epoch_loss / len(iterator)), accuracy_score(labels, preds)


if __name__ == '__main__':

    batch_per_class = False
    eo_optimization = False
    dataset = 'Moji'
    name_repo = 'demonic_' + dataset

    embedding = 2304
    data_dir = "./data/moji"
    n_classe = 2

    args = {
        "dataset": dataset,
        "emb_size": embedding,
        "num_classes": n_classe,
        "batch_size": 128,
        "lr": 0.001,
        "data_dir": data_dir,
        "device_id": 0,
        "exp_id": name_repo,
        "adv_level": "last_hidden",
    }

    debias_options = base_options.BaseOptions()
    debias_state = debias_options.get_state(args=args, silence=True)

    seed_everything(2022)

    model = TransformerModel(args) # 使用 TransformerModel

    # Prepare data
    data_args = dummy_args(args['dataset'], args['data_dir'])  # , self.args.BT, self.args.BTObj)
    task_dataloader = dataloaders.loaders.name2loader(data_args)

    train_data = task_dataloader(args=data_args, split="train")
    dev_data = task_dataloader(args=data_args, split="dev")
    test_data = task_dataloader(args=data_args, split="test")

    train_dataloader_params = {
        'batch_size': args['batch_size'],
        'shuffle': True}

    eval_dataloader_params = {
        'batch_size': args['batch_size'],
        'shuffle': True}
    train_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
    dev_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
    test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)


    device = model.device # 获取device

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased") # 加载 tokenizer


    best_performance = 0
    best_epoch = 0

    for epoch in range(50):
        optimizer = model.optimizer
        criterion = model.criterion

        torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0
        for it, batch in enumerate(train_generator):
            text = batch[0] # 获取文本数据
            # tokenize
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = encoded_input["input_ids"]
            p_tags = encoded_input["attention_mask"]

            # 传入forward
            predictions = model(input_ids, p_tags)

            loss = criterion(predictions, p_tags)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print('Loss at epoch : ', epoch, " is ", epoch_loss / len(train_generator))

        eval_loss, eval_accuracy = eval_epoch(model, test_generator, device, tokenizer) # 传递 tokenizer

        print('Accuracy at epoch : ', epoch, " is ", eval_accuracy)

        if eval_accuracy > best_performance:
            best_performance = eval_accuracy
            best_epoch = epoch

            filename = './demon_transformer_moji.pt' # 修改文件名

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, filename)

    print('Best model has accuracy of : ', best_performance, 'at epoch :', best_epoch)