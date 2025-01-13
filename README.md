# Attention-based Tracker

This repository contains an implementation of attention based tracker model introduced in: 

```
@inproceedings{sunder2024end,
  title={End-To-End Real Time Tracking of Children’s Reading with Pointer Network},
  author={Sunder, Vishal and Karrolla, Beulah and Fosler-Lussier, Eric},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={11731--11735},
  year={2024},
  organization={IEEE}
}
```

To train the pointer network, run `run_training.sh`. The pointer network will train using alignments from an AED-based ASR model.

To test the pointer network, run `run_test.sh`

## Acknowledgement

This work was funded by NSF grant 2008043, "RI: Small: Early Elementary Reading Verification in Challenging Acoustic Environments."
