import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Average, RunningAverage
import torch
from torch.utils import tensorboard
import torch.nn.functional as F

from src.clutter_data import Dataset
from src.model import FcnResnet50


LAMBDA_NO_GRASP = 1.0
LAMBDA_GRASP = 5.0
LAMBDA_CONFIG = 3.0

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using {} as device for training".format(device))

    kwargs = {"num_workers": 5, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_dataset={},architecture=ResNet50,lr={:.0e}".format(
        time_stamp,
        args.dataset.name,
        args.lr,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset,
        args.batch_size, args.val_split, kwargs
    )

    # build the network
    net = FcnResnet50().to(device)

    net = torch.nn.DataParallel(net)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "width": Average(lambda out: out[4]),
        "rot": Average(lambda out: out[6]),
        "qual": Average(lambda out: out[5]),
        "qual_gtn": Average(lambda out: out[7]),
        "qual_gtp": Average(lambda out: out[8]),
        
    }

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'width')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'qual')
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'rot')
    RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'qual_gtn')
    RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'qual_gtp')

    ProgressBar(persist=True, ascii=True).attach(trainer, ['loss', 'width', 'qual',
                                                           'rot', 'qual_gtn', 'qual_gtp'])

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("quality_loss", metrics["qual"], epoch)
        train_writer.add_scalar("width_loss", metrics["width"], epoch)
        train_writer.add_scalar("rotation_loss", metrics["rot"], epoch)
        train_writer.add_scalar("quality_gtn", metrics["qual_gtn"], epoch)
        train_writer.add_scalar("quality_gtp", metrics["qual_gtp"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("quality_loss", metrics["qual"], epoch)
        val_writer.add_scalar("width_loss", metrics["width"], epoch)
        val_writer.add_scalar("rotation_loss", metrics["rot"], epoch)
        val_writer.add_scalar("quality_gtn", metrics["qual_gtn"], epoch)
        val_writer.add_scalar("quality_gtp", metrics["qual_gtp"], epoch)

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        n_saved=None,
        global_step_transform=global_step_from_engine(trainer),
        require_empty=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {'gpnet_plus': net}
    )

    with open('{}/scaling.txt'.format(logdir), 'w') as f:
        f.write('lambda_grasp,{:.1f}\n'.format(LAMBDA_GRASP))
        f.write('lambda_no_grasp,{:.1f}\n'.format(LAMBDA_NO_GRASP))
        f.write('lambda_config,{:.1f}\n'.format(LAMBDA_GRASP))

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)

    torch.save(net.module.state_dict(), '{}/ros_gpnet_plus.pt'.format(logdir), _use_new_zipfile_serialization=False)


def create_train_val_loaders(root, batch_size, val_split, kwargs):
    # load the dataset
    try:
        train_set = Dataset(root / Path('train'))
        val_set = Dataset(root / Path('val'))

    except FileNotFoundError:
        print("Splits not found in {}. Use random split.".format(root))
        # split into train and validation sets
        dataset = Dataset(root)
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader

def prepare_batch(batch, device):
    depth_im, y_true = batch
    depth_im = depth_im.to(device)
    y_true = y_true.float().to(device)
    return depth_im, y_true

def select(pred, y_all):
    """
    Filter ground-truth and output tensors to those, where ground-truth information is available.
    :param pred: Predictions from network
    :param y_all: Ground-truth information tensor

    :return: Predicted and ground-truth tensors for confidence, orientation and width, as well as a mask for
                where actual grasp-samples are available (excluding background pixels).
    """
    indices = torch.nonzero(y_all[:, :, :, 0])
    grasp_sample_mask = (y_all[indices[:, 0], indices[:, 1], indices[:, 2], 0] == 2)

    qual_pred = pred[0][indices[:, 0], 0, indices[:, 1], indices[:, 2]]
    rot_pred = pred[1][indices[:, 0], :, indices[:, 1], indices[:, 2]]
    params_pred = pred[2][indices[:, 0], :, indices[:, 1], indices[:, 2]]

    qual_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 1].float()
    rot_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 2:6].float()
    params_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 6:6+pred[2].shape[1]].float()
    return (qual_pred, rot_pred, params_pred), (qual_gt, rot_gt, params_gt), grasp_sample_mask

def loss_fn(y_pred, y_true, batch_size, grasp_samples):
    """
    Calculate the loss of the current prediction
    :param y_pred: Predictions of the network, where ground-truth information is available
    :param y_true: Ground-truth information of the data samples
    :param batch_size: Size of the batch, used to normalise the loss
    :grasp_samples: Indices of samples with ground-truth information which do not fall on the background (only actually
                    sampled and tested grasps during dataset creation, no background)

    :return: overall loss and individual loss parts with width, confidence, orientation, as well as confidence for for
                negative and positive ground-truth grasps individually

    """
    label_pred, rotation_pred, params_pred = y_pred
    label, rotation, params = y_true
    gtp_indices = torch.nonzero(label)

    if gtp_indices.size()[0] > 0:
        predicted_quality = label_pred[gtp_indices].squeeze()
        loss_rot = _quat_loss_fn(rotation_pred[gtp_indices][:, 0, :],
                                 rotation[gtp_indices][:, 0, :])
        width_loss = _mse_loss_fn(params_pred[gtp_indices, 0][:, 0], params[gtp_indices, 0][:, 0])
        config_loss = (loss_rot.sum() + width_loss.sum()) / batch_size
        with torch.no_grad():
            combined = 2 * (_quat_ang_distance(rotation_pred[gtp_indices][:, 0, :], rotation[gtp_indices][:, 0, :]) +
                            _l1_loss_fn(params_pred[gtp_indices, 0][:, 0], params[gtp_indices, 0][:, 0]))
            weight = 1 - torch.minimum(torch.tensor(1.0), combined)
        loss_grasp = ((weight - predicted_quality) ** 2).sum() / batch_size

    else:
        loss_grasp = torch.tensor(0.0)
        config_loss = torch.tensor(0.0)
        width_loss = torch.tensor(0.0)
        loss_rot = torch.tensor(0.0)

    loss_background = (label_pred[~grasp_samples].squeeze()**2).sum() / batch_size
    loss_gtn_grasps = (label_pred[grasp_samples][(label[grasp_samples] == 0).nonzero()]**2).sum() / batch_size
    loss_no_grasp = loss_background + loss_gtn_grasps
    loss_confidence = loss_background + loss_gtn_grasps + loss_grasp

    loss = (LAMBDA_NO_GRASP * loss_background + LAMBDA_GRASP * loss_gtn_grasps +
            LAMBDA_GRASP * loss_grasp + LAMBDA_CONFIG * config_loss)

    return loss, width_loss.sum() / batch_size, loss_confidence, loss_rot.sum() / batch_size, loss_no_grasp, loss_grasp

def _quat_ang_distance(pred, target):
    """
    Calculates angular distance between two quaternions, normalised to 1
    """
    return 2.0/3.14159 * torch.arccos(torch.abs(torch.sum(pred * target, dim=1)))

def _quat_loss_fn(pred, target):
    """
    Calculate the distance metric of two quaternions according to Kuffner at al.
    """
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

def _l1_loss_fn(pred, target):
    """
    Calculate L1 loss
    """
    return F.l1_loss(pred, target, reduction="mean")

def _mse_loss_fn(pred, target):
    """
    Calculate MSE loss
    """
    return F.mse_loss(pred, target, reduction="mean")

def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        """
        Update network for a single batch
        """
        net.train()
        optimizer.zero_grad()

        # forward
        x, y = prepare_batch(batch, device)
        predicted = net(x)
        y_pred, y_true, grasp_samples = select(predicted, y)
        batch_size = x.shape[0]
        loss, width_loss, q_loss, rot_loss, q_gtn, q_gtp = loss_fn(y_pred, y_true, batch_size, grasp_samples)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y_true, loss, width_loss, q_loss, rot_loss, q_gtn, q_gtp

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        """
        Run inference with the network for a single batch.
        """
        net.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            predicted = net(x)
            y_pred, y_true, grasp_samples = select(predicted, y)
            batch_size = x.shape[0]

            loss, width_loss, q_loss, rot_loss, q_gtn, q_gtp = loss_fn(y_pred, y_true, batch_size, grasp_samples)

        return x, y_pred, y_true, loss, width_loss, q_loss, rot_loss, q_gtn, q_gtp

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
