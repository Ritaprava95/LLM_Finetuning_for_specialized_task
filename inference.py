from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import common


# loading the original model first then attaching the peft model from the checkpoint
model_name= 'microsoft/phi-2'
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)
# loding the peft-lora from last checkpoint
peft_model = PeftModel.from_pretrained(original_model, "../models/peft-dialogue-summary-training/checkpoint-3", torch_dtype=torch.float16,is_trainable=False)

# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

huggingface_dataset_name = "neil-code/dialogsum-test"
dataset = load_dataset(huggingface_dataset_name)


prompt = dataset['test'][index]['dialogue']
response_gt = dataset['test'][index]['summary']

prompt = common.create_prompt_formats(prompt)
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

output = peft_model.generate(**inputs, max_length=500)
output = tokenizer.batch_decode(output)[0]
response_pred = output.split('Output:\n')[1]
print(response_pred



# alternate way is to load the merged model (I still need to verify this way, so commenting this)
# from transformers import AutoModelForCausalLM
# peft_model_2 = AutoModelForCausalLM.from_pretrained("../models/merged",
#                                                     device_map=device_map,
#                                                     quantization_config=bnb_config,
#                                                     trust_remote_code=True,
#                                                     use_auth_token=True
#                                                     )
# peft_tokenizer_2 = AutoTokenizer.from_pretrained("../models/merged",
#                                                  trust_remote_code=True,padding_side="left",
#                                                  add_eos_token=True,
#                                                  add_bos_token=True,
#                                                  use_fast=False)
# peft_tokenizer_2.pad_token = peft_tokenizer_2.eos_token
