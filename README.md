Sparse Progressive Distillation: Resolving Overfitting under Pretrain-and-Finetune Paradigm
======== 
This is the Pytorch implementation for sparse progressive distillation (SPD). For more details about the motivation, techniques and experimental results, refer to our paper [(https://arxiv.org/pdf/2110.08190.pdf)](https://arxiv.org/pdf/2110.08190.pdf).

Experimental results on GLUE Benchmark
============
The model training and evaluation are performed with CUDA 11.1 on Quadro RTX6000 GPU and Intel(R) Xeon(R) Gold 6244 @ 3.60GHz CPU. The results may vary due to different GPU models, drivers, CUDA SDK versions.
<img width="761" alt="image" src="https://github.com/shaoyiHusky/SparseProgressiveDistillation/assets/66193101/44a85974-1882-4ef6-94b4-3b2ce681d456">


Running
============
* **Environment** Preparation (**using python3**)

  ```bash
  pip install -r requirements.txt
  ```

* **Dataset** Preparation

  The original GLUE dataset could be downloaded [here](https://gluebenchmark.com/tasks).

BERT_base fine-tuning on GLUE 
====================

We use finetuned BERT_base as the teacher. For each task of GLUE benchmark, we obtain the finetuned model using the original huggingface [transformers](https://github.com/huggingface/transformers) [code](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification) with the following script.


```
python run_glue.py \
          --model_name_or_path $INT_DIR \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --data_dir $GLUE_DIR/$TASK_NAME/ \
          --max_seq_length 128 \
          --per_gpu_train_batch_size 32 \
          --per_gpu_eval_batch_size 32 \
          --learning_rate 3e-5 \
          --num_train_epochs 4.0 \
          --output_dir $OUT_DIR \
          --evaluate_during_training \
          --overwrite_output_dir \
          --logging_steps 400 \
          --logging_dir $OUT_DIR \
          --save_steps 10000
```

Sparse Progressive Distillation
====================

We use `run_glue.py` to run the **sparse progressive distillation**. --num_prune_epochs is the epochs for pruning. --num_train_epochs is the total number of epochs (pruning, progressive distillation, finetuning).

```
python run_glue.py \
  --model_name_or_path PATH_TO_FINETUNED_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 6.4e-4 \
  --save_steps 50 \
  --num_prune_epochs 30 \
  --num_train_epochs 60 \
  --sparsity 0.9 \
  --output_dir $OUT_DIR \
  --evaluate_during_training \
  --replacing_rate 0.8 \
  --overwrite_output_dir \
  --steps_for_replacing 0 \
  --scheduler_type linear
```

Citation
====================

If you find this repo is helpful, please cite

```
@inproceedings{huang2022sparse,
  title={Sparse Progressive Distillation: Resolving Overfitting under Pretrain-and-Finetune Paradigm},
  author={Huang, Shaoyi and Xu, Dongkuan and Yen, Ian and Wang, Yijue and Chang, Sung-En and Li, Bingbing and Chen, Shiyang and Xie, Mimi and Rajasekaran, Sanguthevar and Liu, Hang and others},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={190--200},
  year={2022}
}
```
