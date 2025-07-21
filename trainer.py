import os
from datetime import datetime
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from loss import MultiResolutionSTFTLoss
from utils.audio import batch_convolution, add_noise_batch, audio_normalize_batch

torch.autograd.set_detect_anomaly(True)


def reduce_loss(loss: torch.Tensor, rank: int, world_size: int):
    """
    同步和汇总各个卡上的loss
    :param loss: loss
    :param rank:
    :param world_size:
    :return:
    """
    with torch.no_grad():
        dist.reduce(loss, dst=0)
    if rank == 0:
        loss /= world_size
    return loss


class Trainer:
    def __init__(self, model, local_rank, global_rank, world_size,
                 train_dataset: torch.utils.data.Dataset,
                 valid_dataset: torch.utils.data.Dataset, config, eval_config, args):
        # 参数赋值
        self.model_name = f"{args.save_name}-{datetime.now().strftime('%y%m%d-%H%M%S')}"

        self.global_rank = global_rank
        self.local_rank = local_rank
        self.config = config
        self.eval_config = eval_config
        self.args = args
        self.model = model

        self.device = f'cuda:{self.local_rank}'
        self.world_size = world_size
        self.model_checkpoint_dir = os.path.join(config.checkpoint_dir, self.model_name)
        if self.global_rank == 0:
            print(f'Model saved at {self.model_checkpoint_dir}')
            if not os.path.exists(self.model_checkpoint_dir):
                os.makedirs(self.model_checkpoint_dir, exist_ok=True)


        # 模型初始化
        self._init_model()

        # 数据初始化
        self.train_data_sampler = DistributedSampler(train_dataset)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,   # DataSampler内置了打乱功能，因此这里不需要打乱
            drop_last=True,
            num_workers=config.num_workers,
            sampler=self.train_data_sampler,
        )
        self.valid_data_sampler = DistributedSampler(valid_dataset, shuffle=False)
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=config.num_workers,
            sampler=self.valid_data_sampler,
        )

    def _init_model(self):
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.global_rank == 0:
            print(f"Total params: {total_params}")

        # Loss
        fft_sizes = [64, 512, 2048, 8192]
        hop_sizes = [32, 256, 1024, 4096]
        win_lengths = [64, 512, 2048, 8192]
        sc_weight = 1.0
        mag_weight = 1.0

        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.recon_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.loss_dict = {}

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-6)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_decay_factor,
        )

        # Load checkpoint if resuming
        if self.args.checkpoint_path:
            state_dicts = torch.load(self.args.checkpoint_path, map_location=self.device)

            self.model.load_state_dict(state_dicts["model_state_dict"])

            if "optim_state_dict" in state_dicts.keys():
                self.optimizer.load_state_dict(state_dicts["optim_state_dict"])

            if "sched_state_dict" in state_dicts.keys():
                self.scheduler.load_state_dict(state_dicts["sched_state_dict"])

    def make_batch_data(self, batch):
        flipped_rir = batch["flipped_rir"].to(self.device)
        source = batch['source'].to(self.device)

        reverberated_source = batch_convolution(source, flipped_rir)

        noise = batch['noise'].to(self.device)
        snr_db = batch['snr_db'].to(self.device)

        batch_size, _, _ = noise.size()

        reverberated_source, normalize_factors = \
            audio_normalize_batch(reverberated_source, "rms", self.config.rms_level)

        # Noise SNR
        reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_db)

        # Noise for late part
        rir_length = int(self.config.rir_duration * self.config.sr)
        stochastic_noise = torch.randn((batch_size, 1, rir_length), device=self.device)
        batch_stochastic_noise = stochastic_noise.repeat(1, self.config.num_filters, 1)

        # Noise for decoder conditioning
        batch_noise_condition = torch.randn((batch_size, self.config.noise_condition_length), device=self.device)

        return (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        )

    def train(self):
        for epoch in range(self.args.resume_step, self.config.num_epochs):
            self.model.train()
            self.train_data_sampler.set_epoch(epoch)
            self.valid_data_sampler.set_epoch(0)
            torch.cuda.empty_cache()
            for i, batch in enumerate(self.train_dataloader):
                batch, graph, room_info_vector = batch
                graph = graph.to(self.device)
                rir = batch['rir'].to(self.device)
                room_info_vector = room_info_vector.to(self.device)
                # Make batch data
                (
                    reverberated_source_with_noise,
                    reverberated_source,
                    batch_stochastic_noise,
                    batch_noise_condition,
                ) = self.make_batch_data(batch)


                # Model forward
                predicted_rir, _ = self.model(
                    reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, graph, room_info_vector
                )

                total_loss = 0.0

                # Compute loss
                stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
                stft_loss = stft_loss_dict["total"]
                sc_loss = stft_loss_dict["sc_loss"]
                mag_loss = stft_loss_dict["mag_loss"]

                total_loss = total_loss + stft_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                self.optimizer.step()

                # 汇总各个卡的loss
                sc_loss = reduce_loss(sc_loss, self.global_rank, self.world_size).item()
                mag_loss = reduce_loss(mag_loss, self.global_rank, self.world_size).item()
                stft_loss = reduce_loss(stft_loss, self.global_rank, self.world_size).item()
                total_loss = reduce_loss(total_loss, self.global_rank, self.world_size).item()

                if self.global_rank == 0 and i % 10 == 0:
                    print(
                        "epoch",
                        epoch,
                        "batch",
                        i,
                        "total loss",
                        total_loss,
                    )
                
            # Validate
            if (epoch + 1) % self.config.validation_interval == 0:
                if self.global_rank == 0:
                    print("Validating...")
                self.model.eval()

                with torch.no_grad():
                    valid_loss = self.validate()
                    # 汇总验证loss
                    valid_loss = reduce_loss(valid_loss, self.global_rank, self.world_size).item()
                    if self.global_rank == 0:
                        print(f"Validation loss : {valid_loss}")
                self.model.train()
                
            self.scheduler.step()
            # Log
            if self.global_rank == 0:
                print(self.model_name)
                print(
                    f"Train {epoch}/{self.config.num_epochs} - loss: {total_loss:.3f}, stft_loss: {stft_loss:.3f}, sc_loss: {sc_loss:.3f}, mag_loss: {mag_loss:.3f}"
                )
                print(f"Curr lr : {self.scheduler.get_last_lr()}")

                # Save model
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    print("Saving model at epoch", epoch)
                    # save model
                    state_dicts = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                        "sched_state_dict": self.scheduler.state_dict(),
                    }

                    torch.save(state_dicts, os.path.join(self.model_checkpoint_dir, f"epoch-{epoch}.pt"))

    def validate(self):

        total_loss = 0.0
        count = 0
        for i, batch in enumerate(self.valid_dataloader):
            batch, graph, room_info_vector = batch
            graph = graph.to(self.device)
            rir = batch['rir'].to(self.device)
            room_info_vector = room_info_vector.to(self.device)

            # Make batch data
            (
                reverberated_source_with_noise,
                reverberated_source,
                batch_stochastic_noise,
                batch_noise_condition,
            ) = self.make_batch_data(batch)

            # Model forward
            predicted_rir, _ = self.model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, graph, room_info_vector
            )

            # Compute loss
            stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
            stft_loss = stft_loss_dict["total"]

            total_loss = total_loss + stft_loss
            count += 1

        return total_loss / count

    def plot(self, batch, nth_batch, epoch):
        if self.global_rank == 0:
            print("Plotting...")
        batch, graph, room_info_vector = batch
        graph = graph.to(self.device)
        # Make batch data
        (
            total_reverberated_source_with_noise,
            total_reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = self.make_batch_data(batch)

        # Model forward
        predicted_rir, _ = self.model(total_reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition, graph, room_info_vector)

        rir = batch['rir'].to(self.device)
        source = batch['source'].to(self.device)

        flip_predicted_rir = torch.flip(predicted_rir, dims=[2])

        reverberated_speech_predicted = batch_convolution(source, flip_predicted_rir)
        reverberated_speech_predicted = audio_normalize_batch(
            reverberated_speech_predicted, "rms", self.config.rms_level
        )

        # Plot to tensorboard
        for i in range(self.config.batch_size):
            curr_true_rir = rir[i, 0]
            curr_predicted_rir = predicted_rir[i, 0]
            plt.ylim([-self.config.peak_norm_value, self.config.peak_norm_value])
            plt.plot(curr_true_rir.cpu().numpy()[:10000])
            plt.plot(curr_predicted_rir.cpu().numpy()[:10000])
           
    def log_gradients(self, epoch):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(tag=f'gradients/{name}', values=param.grad, global_step=epoch)
