import argparse

from misc.reduction_utils.trainer import AutoencoderTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-dim", type=int, default=384)
    parser.add_argument("--encoder-dims", type=int, nargs="+", default=[256, 256, 128, 128, 64])
    parser.add_argument("--decoder-dims", type=int, nargs="+", default=[64, 128, 128, 256, 256])
    args = parser.parse_args()

    trainer = AutoencoderTrainer(
        max_epoches=args.max_epoches,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        lr=args.lr,
        input_dim=args.input_dim,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
    )
    trainer.train()
    trainer.reduce()
