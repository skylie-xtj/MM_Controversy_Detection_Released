import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import warnings
from usfutils.load import instantiate_from_config
from mcd.src.dataset import MCD_collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from usfutils.logger import get_root_logger
from usfutils.config import load_yaml, copy_opt_file
from usfutils.dir import scandir
from usfutils.utils import set_seed_everything
from usfutils.load import instantiate_from_config
from usfutils.format import dict_to_str
from mcd.utils.metrics import metrics
from torch.utils.tensorboard import SummaryWriter
from mcd.module.mcd_model import MCDModel

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    config_path = "MM_Controversy_Detection_Released/mcd/config/train.yaml"
    config = load_yaml(config_path)
    device = "cuda" if (torch.cuda.is_available() and config.gpu) else "cpu"
    set_seed_everything(seed=config.seed)
    experiments_root = os.path.join(current_dir, "experiments/" + config.name)
    state_dict_path = os.path.join(
        experiments_root, f"model_dir/{config.baseline_type}_{config.log_type}"
    )
    if not os.path.exists(state_dict_path):
        os.makedirs(state_dict_path)
    copy_opt_file(config_path, experiments_root)
    logger = get_root_logger(
        log_path=experiments_root, log_name=f"{config.baseline_type}_{config.log_type}"
    )
    logger.info(dict_to_str(config))

    dataset_train = instantiate_from_config(config.data.train)
    dataset_valid = instantiate_from_config(config.data.valid)
    dataset_test = instantiate_from_config(config.data.test)
    train_dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
    )
    valid_dataloader = DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
    )
    model = MCDModel(config=config.model)
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    is_early_stop = False
    current_iter = 0
    best_acc = 0
    best_epoch = 0
    writer = SummaryWriter("logs")
    test_best_results = ""
    for epoch in range(config.epochs):
        if is_early_stop:
            break
        t_predict = []
        t_label = []
        # train
        for idx, data in enumerate(train_dataloader):
            current_iter += 1
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            outputs, moe_loss = model(**data, device=device)
            _, predicts = torch.max(outputs, 1)
            loss = criterion(outputs, label) + moe_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (current_iter + 1) % config.log_freq == 0:
                logger.info(
                    f"Epoch:{epoch}, Data: {idx}/{len(dataset_train) // config.batch_size}, Loss: {round(loss.item(), 5)}"
                )
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
        results1 = metrics(t_label, t_predict)
        logger.info(f"After epoch {epoch}, training results: {results1}")
        writer.add_scalar(
            "train/loss", round(float(loss.detach().cpu()), 5), global_step=epoch
        )
        writer.add_scalar("train/accuracy", results1["acc"], global_step=epoch)
        # valid
        t_predict.clear()
        t_label.clear()
        for idx, data in enumerate(valid_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            with torch.no_grad():
                outputs, moe_loss = model(**data, device=device)
                _, predicts = torch.max(outputs, 1)
                loss = criterion(outputs, label) + moe_loss
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
        results2 = metrics(t_label, t_predict)
        writer.add_scalar(
            "valid/loss", round(float(loss.detach().cpu()), 5), global_step=epoch
        )
        writer.add_scalar("valid/accuracy", results2["acc"], global_step=epoch)
        logger.info(f"After epoch {epoch}, validing results: {results2}")
        if results2["acc"] > best_acc:
            best_acc = results2["acc"]
            best_epoch = epoch + 1
            if best_acc > config.save_threshold:
                try:
                    for remove_file in scandir(
                        state_dict_path, suffix=".pth", full_path=True
                    ):
                        os.remove(remove_file)
                    save_path = os.path.join(
                        state_dict_path, f"{config.baseline_type}_epoch{best_epoch}_b.pth"
                    )
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Saved {save_path}")
                except:
                    pass
        else:
            if epoch - best_epoch >= config.epoch_stop - 1:
                is_early_stop = True
                logger.info(f"Early Stopping on Epoch {epoch}...")
                save_path = os.path.join(
                    state_dict_path, f"{config.baseline_type}_epoch{epoch}_l.pth"
                )
                torch.save(model.state_dict(), save_path)
        # test
        t_predict.clear()
        t_label.clear()
        for idx, data in enumerate(test_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            with torch.no_grad():
                outputs, moe_loss = model(**data, device=device)
                _, predicts = torch.max(outputs, 1)
                loss = criterion(outputs, label) + moe_loss
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
        results3 = metrics(t_label, t_predict)
        if results2["acc"] == best_acc:
            test_best_results = results3
        writer.add_scalar(
            "test/loss", round(float(loss.detach().cpu()), 5), global_step=epoch
        )
        writer.add_scalar("test/accuracy", results3["acc"], global_step=epoch)
        logger.info(
            f"{config.baseline_type}: After epoch {epoch}, testing results: {results3}"
        )
        logger.info(f"{config.baseline_type}: Testing best results: {test_best_results}")
