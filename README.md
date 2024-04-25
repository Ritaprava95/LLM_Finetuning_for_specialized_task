# We will finetune an open-source LLM using PEFT, LoRA to perform sepecialized task using Prompt Engineering.

## Environment setup
conda create -n llm_finetuning python=3.11 <br />
conda install anaconda::jupyter <br />
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia <br />
pip install transformers <br />
pip install datasets <br />
pip install peft accelerate <br />
pip install trl <br />
pip install scipy <br />
pip install bitsandbytes <br />