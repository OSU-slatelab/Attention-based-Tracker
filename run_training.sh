python main.py \
	--nnodes 1 \
	--gpus 1 \
	--node_rank 0 \
	--gpu-num 1 \
	--nepochs 60 \
	--epochs-done 0 \
	--train-path '/research/nfs_fosler_1/vishal/text/readr/train1.csv' \
	--alignment-path '/research/nfs_fosler_1/vishal/alignments/mfa/char' \
	--logging-file "logs/MFA_readrONLY_tracker_attAdd_noAug_lam1.0.log" \
	--save-path "/research/nfs_fosler_1/vishal/saved_models/MFA_readrONLY_tracker_attAdd_noAug_lam1.0.pth.tar" \
	--ckpt-path "" \
	--attn-type "additive" \
	--batch-size 32 \
	--bsz-small 1 \
	--hidden 512 \
	--t-layer 2 \
	--s-layer 4 \
	--nspeech-feat 80 \
	--sample-rate 16000 \
	--lam 1.0 \
	--lr 0.0005 \
	--clip 1.0 \
	--corpus 'readr' \
	--dropout 0.15
#'/research/nfs_fosler_1/vishal/text/libri/train_full_960.csv'
#
