# LARCH: Large Language Model-based Automatic Readme Creation with Heuristics

LARCH is an automatic readme generation system using language models.

## Usage

### Prerequisite

Install LARCH with pip:

```
pip install larch-readme
```

### Python CLI

You can then test out generation without setting up a server.

```
larch --local --model openai/text-davinci-003 --openai-api-key ${YOUR_OPENAI_API_KEY}
```

or you can rely on a server to do generation (See following for setting up a server):

```
larch --endpoint https://${YOUR_SERVER_ADDRESS} --model openai/text-davinci-003
```

### Server

Start the server.

```
OPENAI_API_KEY=${MY_API_KEY} larch-server
```

You can access http://localhost:8000/docs to see the API.

You may want to specify `--host ${YOUR_HOST_NAME_OR_IP_ADDRESS}` if you intend to access from a remote machine.

Both environmental variables are optional.
Spcify OPENAI_API_KEY if you want to allow users to use OpenAI-based models.
Specify ENTRYPOINT_EXTRACTOR if you want to use entrypoint-based generation (strongly recommended; trained with [script/entrypoint_extractor.py](./script/entrypoint_extractor.py)).

You can limit the models to load with `LOADED_MODELS` environmental variable (not setting anything loads all models).
You can also load pretrained encoder-decoder model by passing json serialization mapping from their names to their paths with `ENCODER_DECODER_MODEL_PATHS`.

```bash
# This loads gpt2, gpt2-xl and a pretrained encoder-decoder model from ./path-to-model/
LOADED_MODELS='gpt2,gpt2-xl' ENCODER_DECODER_MODEL_PATHS='{"my-encdec": "./path-to-model/"}' larch-server

# This only loads a pretrained encoder-decoder model. Notice that empty LOADED_MODELS and unset LOADED_MODELS have different behaviors.
LOADED_MODELS='' ENCODER_DECODER_MODEL_PATHS='{"my-encdec": "./path-to-model/"}' larch-server
```

You can download VSCode plugin (vsix file) to interact with the server from [here](https://github.com/hitachi-nlp/larch-vscode/releases/download/v0.0.5/larch-0.0.5.vsix).

## Usage with Docker

Build docker image (you need to set up proxy settings appriopriately if you are behind a proxy server).

```bash
docker build -t larch .
```

You may need to pass `--build-arg CURL_CA_BUNDLE=""` if you are behind a proxy and getting a SSL error.
WARNING: This disables SSL connection thus make your connection vulnerable against attacks.

Then you can start the server with the following command:

```bash
docker run \
 --rm \
  -p ${YOUR_HOST_IP}:${PORT}:80/tcp \
   \
  larch
```

You need to pass `-e OPENAI_API_KEY=${YOUR_OPENAI_API_KEY}` if you wish to use OpenAI models.

You may need to pass `-e CURL_CA_BUNDLE=""` if you are behind a proxy and getting a SSL error.
WARNING: This disables SSL connection thus make your connection vulnerable against attacks.

## Development

Alternatively, you can run CLI without using pip for better debugging and development.

```
pip install -r requirements.txt
export PYTHONPATH=`pwd`

# test out generation
python larch/cli.py --local --model gpt2

# start debug server
python larch/server.py --reload --log-level debug
```

For testing:

```bash
pip install 'pytest>=7.2.0' 'pytest-dependency>=0.5.1'
export PYTHONPATH=`pwd`
py.test -v tests
```

## Model Training and Evaluation

### Training Encoder-Decoder Models

You can train your own Encoder-Decoder Model with [scripts/finetune_encdec.py](scripts/finetune_encdec.py).

```bash
# Make sure you have CUDA 11.6 installed
# We do custom torch installation to enble GPU
pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r <(cat requirements.txt | grep -v torch)
pip install -r requirements-dev.txt

export PYTHONPATH=`pwd`

python scripts/finetune_encdec.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file ./path-to-train.jsonl \
    --validation_file ./path-to-dev.jsonl \
    --output_dir ./tmp-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir
```

Supported models are BART, mBART, T5, mT5 and LED.
Only T5 models t5-small, t5-base, t5-large, t5-3b and t5-11b must use an additional argument: --source_prefix "summarize: ".
