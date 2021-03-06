<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# RESULTS

## Environments
- date: `Thu Oct 28 16:54:32 2021 -0400`
- python version: `3.9.5 (default, Jun  4 2021, 12:28:51) [GCC 7.5.0]`
- espnet version: `espnet 0.10.3a2`
- pytorch version: `pytorch 1.8.1+cu102`
- Git hash: `d7093719d98692774bb47d3c9470a1ca94d33866`
  - Commit date: `Thu Oct 28 16:54:32 2021 -0400`

## Using Conformer based encoder and Transformer based decoder with spectral augmentation and predicting transcript along with intent
- ASR config: [conf/train_asr.yaml](conf/tuning/train_asr_conformer.yaml)
- token_type: word
- Entity classification code borrowed from SLURP [1] official repo - https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation
- Pretrained Model
  - Zenodo : https://zenodo.org/record/5651224
  - Hugging Face : https://huggingface.co/espnet/siddhana_slurp_entity_asr_train_asr_conformer_raw_en_word_valid.acc.ave_10best

|dataset|Snt|Entity Classification (F1 Score)|
|---|---|---|
|inference_asr_model_valid.acc.ave_10best/test|13078|71.9|

### Intent Classification Results


|dataset|Snt|Intent Classification (%)|
|---|---|---|
|inference_asr_model_valid.acc.ave_10best/test|13078|84.4|
|inference_asr_model_valid.acc.ave_10best/valid|8690|85.4|


## Citation

```
@inproceedings{slurp,
    author = {Emanuele Bastianelli and Andrea Vanzo and Pawel Swietojanski and Verena Rieser},
    title={{SLURP: A Spoken Language Understanding Resource Package}},
    booktitle={{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)}},
    year={2020}
}
```
