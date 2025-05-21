from loguru import logger
from vllm import LLM, SamplingParams

class VLLMModel:
    def __init__(self, model, tokenizer, model_name, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id
        
        if generation_config is None:
            self.generation_config = SamplingParams()
        else:
            if isinstance(generation_config, SamplingParams):
                self.generation_config = generation_config
            else:
                self.generation_config = SamplingParams(**generation_config)


    def apply_chat_template(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        prefill = False
        if messages[-1]['role'] == 'assistant':
            prefill = True
        
        try:
            # first try the model's own tokenizer
            if prefill:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        except Exception as e:
            conversation = self.conversation.copy()
            if messages[-1]['role'] != 'assistant':
                messages.append({"role": "assistant", "content": None})
        
            if messages[0]['role'] == 'system':
                conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            for msg in messages:
                conversation.append_message(msg['role'], msg['content'])
            
            prompt = conversation.get_prompt()
            if conversation.name == 'vicuna_v1.1':
                prompt = prompt.replace('user:', 'User:').replace('assistant:', 'ASSISTANT:')
            
            
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            # if there are two bos tokens, remove one
            # prompt = prompt.replace(self.tokenizer.bos_token, '', 1).lstrip()
            prompt = prompt.replace(self.tokenizer.bos_token, '', 1)
        
        if self.tokenizer.bos_token and not prompt.startswith(self.tokenizer.bos_token):
            prompt = self.tokenizer.bos_token + prompt
            
        if prefill:
            if self.tokenizer.eos_token and prompt.strip().endswith(self.tokenizer.eos_token):
                idx = prompt.rindex(self.tokenizer.eos_token)
                prompt = prompt[:idx].rstrip()
            
        return prompt
    
    def chat(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]
        input_text = self.apply_chat_template(messages)
        
        if "sampling_params" in kwargs:
            temp_generation_config = kwargs["sampling_params"]
        else:
            temp_generation_config = self.generation_config.clone()
            for k in kwargs:
                if k in self.generation_config.__annotations__.keys():
                    setattr(temp_generation_config, k, kwargs[k])

        outputs = self.model.generate([input_text], temp_generation_config)
        generated_text = outputs[0].outputs[0].text
        return generated_text
        
    def batch_chat(self, batch_messages, **kwargs):
        input_texts = []
        for messages in batch_messages:
            # print(messages)
            try:
                if isinstance(messages, str):
                    messages = [
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
                input_text = self.apply_chat_template(messages)
            except:
                try:
                    input_text = messages[-1]["content"]
                except:
                    input_text = messages
                
            # print(input_text)
                
            input_texts.append(input_text)
            
        if "sampling_params" in kwargs:
            temp_generation_config = kwargs["sampling_params"]
        else:
            temp_generation_config = self.generation_config.clone()
            for k in kwargs:
                if k in self.generation_config.__annotations__.keys():
                    setattr(temp_generation_config, k, kwargs[k])

        outputs = self.model.generate(input_texts, temp_generation_config)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts