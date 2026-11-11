"""Pluggable local, OpenAI-compatible, cached and replay generation backends."""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def json_object_complete(text):
    text=text.strip()
    if not text.startswith('{'): return False
    try: value,end=json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError: return False
    return isinstance(value,dict) and not text[end:].strip()


def first_object_end(text):
    leading=len(text)-len(text.lstrip())
    if not text.lstrip().startswith('{'): return None
    try: value,end=json.JSONDecoder().raw_decode(text.lstrip())
    except json.JSONDecodeError: return None
    return leading+end if isinstance(value,dict) else None


class ReplayModel:
    """Explicit fixture backend. Never represents a trained model."""
    def __init__(self, responses): self.responses = iter(responses)
    def generate(self, messages):
        value = next(self.responses)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return {"text": content, "input_tokens": None, "output_tokens": None,
                "token_source": "unavailable", "backend": "replay", "cache_hit": False}


class ContextLimitError(ValueError):
    def __init__(self, tokens, limit):
        super().__init__(f"Input exceeds configured context ({tokens} > {limit}); no silent truncation")
        self.tokens = tokens
        self.limit = limit


class LocalModel:
    def __init__(self, model, revision=None, adapter=None, max_new_tokens=1024, max_input_tokens=4096,
                 device_map="auto", seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                    torch_dtype=torch.bfloat16, device_map=device_map)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval(); self.max_new_tokens = max_new_tokens; self.max_input_tokens = max_input_tokens
    def generate(self, messages):
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if tokens.shape[-1] > self.max_input_tokens:
            raise ContextLimitError(tokens.shape[-1], self.max_input_tokens)
        device = self.model.get_input_embeddings().weight.device
        tokens = tokens.to(device)
        tokenizer=self.tokenizer; prompt_length=tokens.shape[-1]
        class CompleteObject(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return first_object_end(tokenizer.decode(input_ids[0,prompt_length:],skip_special_tokens=True)) is not None
        if self.torch.cuda.is_available(): self.torch.cuda.synchronize()
        with self.torch.inference_mode():
            generated = self.model.generate(input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                      max_new_tokens=self.max_new_tokens, do_sample=False,
                      temperature=1.0, top_p=1.0, top_k=50,
                      stopping_criteria=StoppingCriteriaList([CompleteObject()]),
                      pad_token_id=self.tokenizer.eos_token_id)
        if self.torch.cuda.is_available(): self.torch.cuda.synchronize()
        output = generated[0, tokens.shape[-1]:]
        raw=self.tokenizer.decode(output,skip_special_tokens=True); end=first_object_end(raw)
        return {"text":raw[:end] if end is not None else raw,"raw_text":raw,
                "json_stop_overhang":raw[end:] if end is not None else '',
                "input_tokens": tokens.shape[-1], "output_tokens": len(output),
                "output_limit_hit": end is None and len(output) == self.max_new_tokens
                    and int(output[-1]) != tokenizer.eos_token_id,
                "token_source": "tokenizer", "backend": "local", "cache_hit": False}


class APIModel:
    def __init__(self, base_url, model, key_env="TABLEQA_API_KEY", max_new_tokens=1024):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key_env, self.max_new_tokens = model, key_env, max_new_tokens
    def generate(self, messages):
        data = json.dumps({"model": self.model, "messages": messages,
                           "temperature": 0, "max_tokens": self.max_new_tokens}).encode()
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json",
              "Authorization": "Bearer " + os.environ[self.key_env]})
        with urllib.request.urlopen(req, timeout=180) as r: value = json.load(r)
        usage = value.get("usage", {})
        return {"text": value["choices"][0]["message"]["content"],
                "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                "token_source": "api_usage", "backend": "api", "cache_hit": False}


class CachedModel:
    def __init__(self, backend, directory, identity):
        self.backend, self.directory, self.identity = backend, Path(directory), identity
        self.directory.mkdir(parents=True, exist_ok=True)
    def generate(self, messages):
        key = hashlib.sha256(json.dumps([self.identity, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.directory / (key + ".json")
        if path.exists(): return {**json.loads(path.read_text(encoding="utf-8")), "cache_hit": True}
        value = self.backend.generate(messages)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8"); temp.replace(path)
        return value


def make_model(config):
    backend = config.get("backend", "local")
    if backend == "local":
        result = LocalModel(config["model"], config.get("revision"), config.get("adapter"),
                            config.get("max_new_tokens", 1024), config.get("max_input_tokens", 4096),
                            config.get("device_map", "auto"), config.get("seed", 42))
    elif backend == "api":
        result = APIModel(config["base_url"], config["model"], config.get("key_env", "TABLEQA_API_KEY"))
    elif backend == "replay": result = ReplayModel(config["responses"])
    else: raise ValueError(backend)
    if config.get("cache"):
        # Config must include frozen revisions, prompt version and generation settings.
        if backend == "local" and not config.get("revision"):
            raise ValueError("Cache requires a pinned model revision")
        if config.get("adapter") and not config.get("adapter_revision"):
            raise ValueError("Cache with adapter requires an explicit immutable adapter_revision identity")
        result = CachedModel(result, config["cache"], config)
    return result
