CUDA_VISIBLE_DEVICES=7 \
python train.py fit -c gapartnet.yaml \
--model.init_args.ckpt ckpt/release.ckpt \
--model.init_args.debug True


CUDA_VISIBLE_DEVICES=5 \
python train.py test -c gapartnet.yaml \
--model.init_args.ckpt ckpt/release.ckpt

CUDA_VISIBLE_DEVICES=0 \
python train.py fit -c gapartnet.yaml \
--model.init_args.ckpt ckpt/release.ckpt