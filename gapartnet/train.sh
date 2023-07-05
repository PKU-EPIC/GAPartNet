CUDA_VISIBLE_DEVICES=7 \
python train.py fit -c gapartnet.yaml \
--model.init_args.ckpt ckpt/sem_seg_accu_82.7.ckpt \
--model.init_args.debug True


CUDA_VISIBLE_DEVICES=0 \
python train.py test -c gapartnet.yaml \
--model.init_args.ckpt ckpt/new.ckpt

CUDA_VISIBLE_DEVICES=0 \
python train.py fit -c gapartnet.yaml