## Clift: C Loop-based Inference Transformer

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

This project is a simple loop-based implementation of the LLaMa 3.x LLM transformer architecture, based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project and its fork by James Delancey [llama3.c](https://github.com/jameswdelancey/llama3.c) (we warmly thank you!). The primary goal of Clift is to serve as a benchmark and real-life example for optimizing compiler projects, e.g., polyhedral compilers.

## Get and compile Clift

```bash
git clone git@github.com:Ced/clift.git
cd clift
make clift
```

## Create the model file

You will need to build a model file containing pre-trained parameters that can be processed by Clift. So step 1 is to get the desired LLaMa 3.x checkpoint from either Meta or HuggingFace.

### Getting and converting checkpoint file from Meta

To get LLaMa 3.x checkpoints from Meta, you should follow [Meta instructions](https://www.llama.com/). Go to the Download page, fill the form and select relevant models, e.g., Llama3.2-1B, then use the suggested commands, e.g.:

```bash
pip install llama-stack
llama model download --source meta --model-id Llama3.2-1B
```

Once we have those checkpoints, we have to convert them into the Clift format. For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 1B model:

```bash
python export.py llama3.2_1b.bin --meta-llama ~/.llama/checkpoints/Llama3.2-1B/
```

### Getting and converting checkpoint file from HuggingFace

Alternatively you can get LLaMa 3.x checkpoints from HuggingFace. In this case you will need to create an [HuggingFace Account](https://huggingface.co/), and get an access token (click on your profile icon, then "Access Tokens"). Finally you'll need to login then to download the desired model, e.g.:

```bash
pip install 'huggingface_hub[cli]'
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
```

Once we have those checkpoints, we have to convert them into the Clift format. For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 1B model:

```bash
python export.py llama3.2_1b.bin --hf ./Llama-3.2-1B/
```

## Run Clift!

Run Clift with `-h` option to get all possible options (in general they match [Llama.cpp](https://github.com/ggml-org/llama.cpp) options). Here is an example of a command line with a 32-token long prompt and asking to generate 32 tokens (beyond the one generated from prompt analysis):

```bash
./clift -m ./llama3.2_1b.bin -n 33 -p "Once upon a time there were three Bears, who lived together in a house of their own, in a wood. One of them was a Little Wee"
```

And here is the output on my M4 Mac:

```
transformer_t configuration:
- embedding_dim:  2048
- hidden_dim:     8192
- layer_count:    16
- q_head_count:   32
- kv_head_count:  8
- vocabulary_len: 128256
- context_len:    2048

[Once upon a time there were three Bears, who lived together in a house of their own, in a wood. One of them was a Little Wee]le Bear, the other was a little Young Bear and the other one was an Old Bear. All the Bears made a plan, and this was their plan: The

Prompt processing (prefill): 32 tokens in  1.895 s (16.886544 tok/s)
Token generation  (decode):  32 tokens in  2.188 s (14.625229 tok/s)
```