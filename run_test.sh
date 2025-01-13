python main.py \
	--nnodes 1 \
	--gpus 1 \
	--node_rank 0 \
	--gpu-num 1 \
	--roll-fac 1 \
	--valid-path '/research/nfs_fosler_1/vishal/text/readr/test1.csv' \
	--gt-path '/research/nfs_fosler_1/vishal/text/readr/test1_align_discrete.csv' \
	--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/MFA_readrONLY_tracker_attAdd_noAug_lam1.0.pth.tar" \
	--logging-file "logs/valid.log" \
	--temp 0.1 \
	--res 40 \
	--batch-size 1 \
	--bsz-small 1 \
	--hidden 512 \
	--t-layer 2 \
	--s-layer 4 \
	--nspeech-feat 80 \
	--sample-rate 16000 \
	--attn-type "additive" \
	--corpus 'readr' \
	--test \
	--load-norm
#--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/readrLong_tracker_attAdd_noAug_lam0.1.pth.tar" \
#--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/MFA_readrONLY_tracker_attAdd_noAug_lam1.0.pth.tar" \

