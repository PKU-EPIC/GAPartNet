# training with debug mode: without wandb logging
CUDA_VISIBLE_DEVICES=0 \
python train.py fit -c gapartnet.yaml \
--model.init_args.training_schedule "[5,10]" \
--model.init_args.ckpt ckpt/release.ckpt \
--model.init_args.debug True

# test
CUDA_VISIBLE_DEVICES=0 \
python train.py test -c gapartnet.yaml \
--model.init_args.training_schedule "[0,0]" \
--model.init_args.ckpt ckpt/release.ckpt

# training
CUDA_VISIBLE_DEVICES=0 \
python train.py fit -c gapartnet.yaml \
--model.init_args.training_schedule "[5,10]" \
--model.init_args.ckpt ckpt/release.ckpt

