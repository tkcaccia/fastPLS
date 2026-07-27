# Pipeline 4: ImageNet requested-SIMPLS rSVD classifier scaling

Pipeline 4 requests `fastPLS` SIMPLS with randomized SVD for ImageNet-scale
DINOv2 feature classification. The output records both `requested_method` and
`executed_method` and rejects a row if they differ. A requested SIMPLS fit is
never replaced by PLS-SVD. If the CUDA dense-indicator memory guard is
exceeded, the run fails explicitly and is retained in the output as an error.
The prepared task has 1,000,000 training samples, 281,167 test samples, 1,024
DINOv2 features, and 1,000 classes.

## Run

```sh
bash benchmark/run_pipeline4_imagenet_simpls_rsvd.sh
```

## Required input

The script expects a prepared task RDS containing training/test features and
labels.  Override the path with:

```sh
TASK_RDS=/path/to/imagenet_task.rds bash benchmark/run_pipeline4_imagenet_simpls_rsvd.sh
```

## Classifiers

Pipeline 4 supports:

- `argmax`
- `lda`
- `cknn`

Candidate-kNN can use blocked or streamed prediction internally to reduce host
RAM pressure on high component counts.

## Main outputs

- `imagenet_simpls_rsvd_classifiers_raw.csv`
- `imagenet_simpls_rsvd_classifiers_time.csv`
- `imagenet_simpls_rsvd_classifiers_joined.csv`
- run manifest
- stdout, timing, and failure logs

The output reports requested and executed estimators, fitting time, prediction
time, total runtime, top-1 accuracy, top-5 accuracy where available, peak host
RAM, peak GPU memory where available, and execution status.
