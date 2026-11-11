# 模块职责：统一回放、本地 Transformers、兼容 API 和文件缓存的生成接口及 token 统计。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Pluggable local, OpenAI-compatible, cached and replay generation backends."""
import hashlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path


# 检查文本是否只包含一个完整 JSON 对象。
# 输入：text（待解析文本）。
def json_object_complete(text):
    text=text.strip()
    if not text.startswith('{'): return False
    try: value,end=json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError: return False
    return isinstance(value,dict) and not text[end:].strip()


# 定位首个 JSON 对象的结束字符，用于本地生成停止与越界文本登记。
# 输入：text（待解析文本）。
def first_object_end(text):
    leading=len(text)-len(text.lstrip())
    if not text.lstrip().startswith('{'): return None
    try: value,end=json.JSONDecoder().raw_decode(text.lstrip())
    except json.JSONDecodeError: return None
    return leading+end if isinstance(value,dict) else None


# ReplayModel：按预置响应回放，用于确定性测试和轨迹重放。
class ReplayModel:
    """Explicit fixture backend. Never represents a trained model."""
    # 初始化 ReplayModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：responses。
    def __init__(self, responses): self.responses = iter(responses)
    # 取出下一条预置响应，未知 token 成本保留为 None。
    # 输入：messages（模型可见消息）。
    # 返回形式含：字段 text、input_tokens、output_tokens、token_source、backend、cache_hit。
    def generate(self, messages):
        value = next(self.responses)
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return {"text": content, "input_tokens": None, "output_tokens": None,
                "token_source": "unavailable", "backend": "replay", "cache_hit": False}


# ContextLimitError：携带实际输入 token 和上限的错误，供超限统计使用。
class ContextLimitError(ValueError):
    # 初始化 ContextLimitError 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：tokens；limit。
    def __init__(self, tokens, limit):
        super().__init__(f"Input exceeds configured context ({tokens} > {limit}); no silent truncation")
        self.tokens = tokens
        self.limit = limit


# LocalModel：本地 BF16 生成后端，负责聊天模板、输入预算、可选 LoRA 和 JSON 停止。
class LocalModel:
    # 初始化 LocalModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：model（生成后端）；revision；adapter；max_new_tokens；max_input_tokens；device_map；seed（随机种子）。
    # 相关调用：AutoTokenizer.from_pretrained、AutoModelForCausalLM.from_pretrained、PeftModel.from_pretrained。
    def __init__(self, model, revision=None, adapter=None, max_new_tokens=1024, max_input_tokens=4096,
                 device_map="auto", seed=42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(seed)
        self.torch = torch
        # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision,
                    torch_dtype=torch.bfloat16, device_map=device_map)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval(); self.max_new_tokens = max_new_tokens; self.max_input_tokens = max_input_tokens
    # 核对聊天模板输入长度后调用本地模型，按完整 JSON 停止并登记实际 token。
    # 输入：messages（模型可见消息）。
    # 相关调用：self.model.generate。
    # 返回形式含：字段 text、raw_text、json_stop_overhang、input_tokens、output_tokens、output_limit_hit、token_source。
    def generate(self, messages):
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # 在搬入 GPU 和生成之前拒绝超长提示，避免静默截断破坏题意与证据。
        if tokens.shape[-1] > self.max_input_tokens:
            raise ContextLimitError(tokens.shape[-1], self.max_input_tokens)
        device = self.model.get_input_embeddings().weight.device
        tokens = tokens.to(device)
        tokenizer=self.tokenizer; prompt_length=tokens.shape[-1]
        # CompleteObject：检查新增生成是否已包含完整 JSON 对象的停止条件。
        class CompleteObject(StoppingCriteria):
            # 实现调用协议，在当前模型状态下判断停止条件或返回计算结果。
            # 输入：input_ids；scores；**kwargs。
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


# APIModel：兼容聊天 API 的后端，从环境变量读取密钥并透传生成上限。
class APIModel:
    # 初始化 APIModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：base_url；model（生成后端）；key_env；max_new_tokens。
    def __init__(self, base_url, model, key_env="TABLEQA_API_KEY", max_new_tokens=1024):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key_env, self.max_new_tokens = model, key_env, max_new_tokens
    # 提交固定参数的聊天请求，从 API 使用量字段读取输入与输出 token。
    # 输入：messages（模型可见消息）。
    # 返回形式含：字段 text、input_tokens、output_tokens、token_source、backend、cache_hit。
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


# CachedModel：按冻结身份与消息摘要缓存响应，采用独立临时文件原子提交。
class CachedModel:
    # 初始化 CachedModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：backend；directory（数据或模型目录）；identity。
    def __init__(self, backend, directory, identity):
        self.backend, self.directory, self.identity = backend, Path(directory), identity
        self.directory.mkdir(parents=True, exist_ok=True)
    # 按冻结身份和消息查缓存，未命中时生成并通过独立临时文件提交。
    # 输入：messages（模型可见消息）。
    # 相关调用：hashlib.sha256(json.dumps([self.identity, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest、self.backend.generate、temp.write_text、path.read_text。
    # 返回形式含：value / 字段 cache_hit。
    def generate(self, messages):
        key = hashlib.sha256(json.dumps([self.identity, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = self.directory / (key + ".json")
        if path.exists(): return {**json.loads(path.read_text(encoding="utf-8")), "cache_hit": True}
        value = self.backend.generate(messages)
        temp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        # 进程 ID 不能区分同进程线程；随机后缀保证同一缓存键的并发写入互不抢临时文件。
        try:
            temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            try:
                temp.replace(path)
            except PermissionError:
                # Windows 上并发原子替换同一目标可能被拒绝。已有完整缓存时保留赢家；
                # 没有有效目标则继续抛错，避免把真实权限错误当作写入成功。
                if not path.is_file():
                    raise
                json.loads(path.read_text(encoding="utf-8"))
        finally:
            temp.unlink(missing_ok=True)
        return value


# 按配置创建回放、本地或 API 后端，按需包装带身份校验的缓存。
# 输入：config（显式运行配置）。
# 返回形式含：result。
# 异常约束：Cache requires a pinned model revision
def make_model(config):
    backend = config.get("backend", "local")
    if backend == "local":
        result = LocalModel(config["model"], config.get("revision"), config.get("adapter"),
                            config.get("max_new_tokens", 1024), config.get("max_input_tokens", 4096),
                            config.get("device_map", "auto"), config.get("seed", 42))
    elif backend == "api":
        result = APIModel(config["base_url"], config["model"], config.get("key_env", "TABLEQA_API_KEY"),
                          config.get("max_new_tokens", 1024))
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
