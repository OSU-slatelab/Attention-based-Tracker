python main.py \
	--nnodes 1 \
	--gpus 1 \
	--node_rank 0 \
	--gpu-num 0 \
	--roll-fac 1 \
	--valid-path '/research/nfs_fosler_1/vishal/text/cmu_kids/test.csv' \
	--logging-file "logs/valid.log" \
	--alignment-path '/research/nfs_fosler_1/vishal/alignments' \
	--ckpt-path "/research/nfs_fosler_1/vishal/saved_models/cmuk_tracker_attAdd_noAug_lam0.0.pth.tar" \
	--batch-size 1 \
	--bsz-small 1 \
	--hidden 512 \
	--t-layer 2 \
	--s-layer 4 \
	--nspeech-feat 80 \
	--sample-rate 16000 \
	--attn-type "additive" \
	--corpus 'cmu_kids' \
	--att-path 'cmu_kids' \
	--get-attn \
	--load-norm
#'/research/nfs_fosler_1/vishal/text/libri/dev_other.csv' \
#'/research/nfs_fosler_1/vishal/audio/timit/test.csv'\
