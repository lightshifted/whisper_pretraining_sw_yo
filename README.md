# Fine-Tune Whisper for Speech Recognition

In this project, we fine-tune the Whisper model and develop cutting-edge speech recognition systems for the Swahili language. The techniques we used can also be applied to other languages.

Further, we evaluate our final model on the "test" split of the Common Voice 11 dataset for Swahili, allowing us to assess the effectiveness of our approach and compare it with other models.

## Overview

We fine-tuned the Whisper model to improve the accuracy of speech recognition in the Swahili language. To do this, we used a large dataset of audio samples in this languages to train the model and optimize its performance.

We also developed a user-friendly interface for the speech recognition system that allows users to easily interact with the model and obtain accurate transcriptions of their speech in real-time. This interface can be easily integrated into existing speech recognition applications and platforms to expand their capabilities and make them more accessible to users of the Swahili and Yoruba languages. The interface is accessible [here](https://huggingface.co/spaces/hedronstone/whisper-large-v2-demo-sw).

![Gradio Demo](https://i.ibb.co/BTqGCSt/example-image.jpg)

Our long term objective is to create a community of users and developers who can contribute to the project by providing feedback, suggestions, and additional data to further improve the performance of the speech recognition system. This community allows us to continuously improve the model and make it more effective for users of the Swahili language.

## Model Weights

The pretrained model weights from this project are stored in HuggingFace repositories: 

| Model  | Repository |
|--------|------------------|
| whisper-small-sw  | [link](https://huggingface.co/hedronstone/whisper-large-v2-sw) |
| whisper-medium-sw | [link](https://huggingface.co/hedronstone/whisper-medium-sw) |
| whisper-large-v2-sw | [link](https://huggingface.co/hedronstone/whisper-small-sw) |

## Set Up an Environment

## Running the Script

1. **Clone model repository**

The steps for running training with a Python script assume that you are SSH'd into your GPU device and have set up your environment according to the previous section [Set Up an Environment](#set-up-an-environment). 

We chose to host our pretrained model files in model repositories on the Hugging Face Hub. These repositories contain all the required files to reproduce the training run, alongside model weights, training logs and a README.md card.
Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone https://huggingface.co/hedronstone/whisper-medium-sw
```

We can then enter the repository using the `cd` command:

```bash
cd whisper-medium-sw
```

2. **Add training script and `run` command**

We provide a Python training script for fine-tuning Whisper with 🤗 Datasets' streaming mode: `run_speech_recognition_seq2seq_streaming.py`

We can then define the model, training and data arguments for fine-tuning:
```bash
echo 'python run_speech_recognition_seq2seq_streaming.py \
    --model_name_or_path="hedronstone/whisper-medium-sw" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --dataset_config_name="sw" \
    --language="swahili" \
    --train_split_name="train+validation" \
    --eval_split_name="test" \
    --model_index_name="Whisper Medium Swahili" \
    --max_steps="5000" \
    --output_dir="./" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="32" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --warmup_steps="500" \
    --evaluation_strategy="steps" \
    --eval_steps="1000" \
    --save_strategy="steps" \
    --save_steps="1000" \
    --generation_max_length="225" \
    --length_column_name="input_length" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --report_to="tensorboard" \
    --metric_for_best_model="wer" \
    --greater_is_better="False" \
    --load_best_model_at_end \
    --gradient_checkpointing \
    --fp16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --do_normalize_eval \
    --streaming \
    --use_auth_token \
    --push_to_hub=' >> run.sh
```

Make sure to change the `--dataset_config_name` and `--language` to the correct values for your language! See also how we combine the train and validation splits as `--train_split_name="train+validation"`. This is recommended for low-resource languages (it probably isn't strictly necessary for Swahili, where the `"train"` split for Common Voice 11 contains ample training data). If you are training on a very small dataset (< 10 hours), it is advisable to disable streaming mode: `--streaming="False"`.

3. **Launch training 🚀**

We recommend running training through a `tmux` session. This means that training won't be interrupted when you close 
your SSH connection. To start a `tmux` session named `mysession`: 

```bash
tmux new -s mysession
```
(if `tmux` is not installed, you can install it through: `sudo apt-get install tmux`)

Once in the `tmux` session, we can launch training:

```bash
bash run.sh
```

Training should take approximately 8 hours, with a final cross-entropy loss of **1e-4** and word error rate of **19.6%**.

Since we're in a `tmux` session, we're free to close our SSH window without stopping training!

If you close your SSH connection and want to rejoin the `tmux` window, you can SSH into your GPU and then connect to 
your session with the following command:

```bash
tmux a -t mysession
```

`tmux` guide: https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/

## Recommended Training Configurations
Here, we will provide some guidance for appropriate training and evaluation batch sizes depending on your GPU device. The Whisper model expects log-Mel input features of a fixed dimension, so GPU memory requirements are independent of any sample's audio length. Thus these recommendations should stand for all 16/40GB GPU devices.

#### V100 / 16 GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size |
|--------|------------------|--------------------|-----------------|
| small  | 16               | 2                  | 8               |
| medium | 2                | 16                 | 1               |

It is advised to run the "small" checkpoint if training on a V100 device. Running the medium checkpoint will take upwards of 12 hours for 5k training steps. 
#### A100 / 40GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size |
|--------|------------------|--------------------|-----------------|
| small  | 64               | 1                  | 32              |
| medium | 32               | 1                  | 16              |

### Punctuation, Casing and Normalisation

When using the Python training script, removing casing for the training data is enabled by passing the flag `--do_lower_case`. Removing punctuation in the training data is achieved by passing the flag `--do_remove_punctuation`. Both of these flags default to False, and we **do not** recommend setting either of them to True. This will ensure your fine-tuned model learns to predict casing and punctuation. Normalisation is only applied during evaluation by setting the flag `--do_normalize_eval` (which defaults to True and recommend setting). 

Normalisation is performed according to the 'official' Whisper normaliser. This normaliser applies the following basic standardisation for non-English text:
1. Remove any phrases between matching brackets ([, ]).
2. Remove any phrases between matching parentheses ((, )).
3. Replace any markers, symbols, and punctuation characters with a space, i.e. when the Unicode category of each character in the NFKC-normalized string starts with M, S, or P.
4. Make the text lowercase.
5. Replace any successive whitespace characters with a space.

Similarly, in the notebooks, removing casing in the training data is enabled by setting the variable `do_lower_case = True`, 
and punctuation by `do_remove_punctuation = True`. We do not recommend setting either of these to True to ensure that 
your model learns to predict casing and punctuation. Thus, they are set to False by default. Normalisation is only 
applied during evaluation by setting the variable `do_normalize_eval=True` (which we do recommend setting). 

## How to Contribute

If you are interested in contributing to this project, there are several ways you can help:

- Provide feedback and suggestions on the performance of the speech recognition system.
- Offer additional data for fine-tuning the Whisper model.
- Join the community of users and developers to discuss and collaborate on the project.

## Branching Strategy

Only well-tested and reviewed code is merged into the master branch. This helps to avoid merge conflicts and ensures the quality of the codebase.

**Steps:**

1. Create a new branch for each feature or bug fix, based on the master branch.
2. Develop and test the feature or fix in the branch.
3. Create a pull request to merge the branch into the master branch.
4. Review the changes in the pull request and discuss any issues or suggestions with the team.
5. If there are no conflicts, we'll merge the branch into the master branch.
6. If there are conflicts, resolve them in the branch and create a new pull request for our review.
7. Repeat the process for each new feature or bug fix.

To get started, fork this repository and submit a pull request with your proposed changes. We look forward to working with you to improve the accuracy of speech recognition in the Swahili and Yoruba languages.

## Tips and Tricks


### Adam 8bit
The [Adam optimiser](https://arxiv.org/abs/1412.6980a) requires two params (betas) for every model parameter. So the memory requirement of the optimiser is **two times** that of the model. You can switch to using an 8bit version of the Adam optimiser from [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes#bitsandbytes). This will cast the optimiser parameters into 8bit precision, saving you a lot of memory and potentially allowing you to run bigger batch sizes.

To use Adam 8bti, you first need to pip install `bitsandbytes`:

```bash
pip install bitsandbytes
```

Then, set `optim="adamw_bnb_8bit"`, either in your `run.sh` file if running from a Python script, or when you instantiate the Seq2SeqTrainingArguments from a Jupyter Notebook or Google Colab:

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    optim="adamw_bnb_8bit"
)
```
### Adafactor

Rather than using Adam, you can use a different optimiser all together. Adam requires two optimiser params per one model param, but [Adafactor](https://arxiv.org/abs/1804.04235) uses only one. To enable Adafactor, set `optim="adafactor"` in the `Seq2SeqTrainingArguments`. You can expect to double your training batch size when using Adafactor compared to Adam. 

A word of caution: Adafactor is untested for fine-tuning Whisper, so we are unsure sure how 
Adafactor performance compares to Adam! For this reason, we recommend Adafactor as an **experimental feature** only.

## Scripts & Example Notebooks

1. [Processing Local Datasets Example Notebook](https://github.com/masslightsquared/whisper_pretraining_sw_yo/blob/jilp/example_custom_dataset_preprocessing.ipynb)
2. [8bit inference for Whisper large model (6.5 gig VRAM) 🤯](https://colab.research.google.com/drive/1EMOwwfm1V1fHxH7eT1LLg7yBjhTooB6j?usp=sharing)
3. [Fine-Tune Whisper Example Notebook](https://github.com/masslightsquared/whisper_pretraining_sw_yo/blob/jilp/example_whisper_training_run.ipynb)