# 模块职责：在独立进程内加载已校验的上游 Chain-of-Table 源码，适配本地文本模型及 TabFact 输出。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Local text-backend adapter for a pinned official Chain-of-Table checkout.

Upstream files/prompts remain separate and retain their licenses. This module
does not reimplement those prompts. The runner is a TabFact/Qwen migration,
not a reproduction of the old OpenAI model or the paper's complete evaluation.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import ModuleType


PINNED_COMMIT = '1fd716c608f56e7d9d156ead432219c7ab9008af'
SYSTEM = 'I will give you some examples, you need to follow the examples and complete the text, and no other content.'


# 只解析上游已知静态分派名或 Python 字面量，拒绝动态表达式执行。
# 输入：text（待解析文本）；scope。
# 异常约束：Bounded source literal or static operation name required
def safe_source_eval(text, scope):
    """Resolve known static dispatch names; parse all other inputs as literals."""
    allowed = {'add_column_act', 'select_row_act', 'select_column_act',
               'group_column_act', 'sort_column_act', 'plan_add_column_demo',
               'plan_select_row_demo', 'plan_select_column_demo',
               'plan_group_column_demo', 'plan_sort_column_demo'}
    if not isinstance(text, str) or len(text) > 131072:
        raise ValueError('Bounded source literal or static operation name required')
    if text in allowed:
        return scope[text]
    return ast.literal_eval(text)


# 按冻结提交与文件摘要核验上游源码和许可文件，发现漂移则终止。
# 输入：audit_directory。
# 相关调用：(root / 'source-manifest.json').read_text、hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\x00' + data).hexdigest、hashlib.sha256(data).hexdigest。
# 返回形式含：2 项组成的结果。
# 异常约束：Unexpected upstream source commit
def verify_source(audit_directory):
    root = Path(audit_directory).resolve()
    manifest = json.loads((root / 'source-manifest.json').read_text(encoding='utf-8'))
    if manifest['commit'] != PINNED_COMMIT:
        raise ValueError('Unexpected upstream source commit')
    source = root / ('source-' + PINNED_COMMIT)
    for entry in manifest['files']:
        path = (source / entry['path']).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Missing or unsafe upstream source path')
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if hashlib.sha256(data).hexdigest() != entry['sha256'] or blob != entry['git_blob_sha1']:
            raise ValueError('Upstream source hash mismatch')
    required = {'utils/chain.py', 'utils/helper.py', 'operations/__init__.py',
                'operations/add_column.py', 'operations/final_query.py',
                'operations/select_row.py', 'operations/select_column.py',
                'operations/group_by.py', 'operations/sort_by.py',
                'third_party/select_column_row_prompts/select_column_row_prompts.py',
                'LICENSE', 'third_party/select_column_row_prompts/LICENSE'}
    if not required <= {e['path'] for e in manifest['files']}:
        raise ValueError('Required official modules or licenses are missing')
    return source, manifest


# 在专用进程中导入已经校验的上游模块，并接入受限的模型适配接口。
# 输入：audit_directory。
# 返回形式含：2 项组成的结果。
# 异常约束：Use a dedicated process to avoid upstream module collisions
def load_upstream(audit_directory):
    """Load only in a dedicated runner process; no old OpenAI dependency needed."""
    source, manifest = verify_source(audit_directory)
    if any(name in sys.modules for name in ['utils', 'operations', 'third_party']):
        raise ValueError('Use a dedicated process to avoid upstream module collisions')
    # Explicit namespaces avoid accidentally importing an installed `utils`.
    for name in ['utils', 'third_party']:
        module = ModuleType(name); module.__path__ = [str(source / name)]
        sys.modules[name] = module
    sys.path.insert(0, str(source))
    chain = importlib.import_module('utils.chain')
    for name in ['utils.chain', 'operations.add_column', 'operations.select_row',
                 'operations.select_column', 'operations.group_by']:
        module = importlib.import_module(name)
        # Global lookup shadows eval only in these verified upstream modules.
        module.eval = lambda text, scope=module.__dict__: safe_source_eval(text, scope)
    return chain, manifest


# LocalTextModel：为上游方法提供文本生成后端，使用其原有停止串规则。
class LocalTextModel:
    """Greedy/sampled text generation with EOS/stop strings, never JSON stopping."""
    # 初始化 LocalTextModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：config（显式运行配置）。
    def __init__(self, config):
        from .models import LocalModel
        self.base = LocalModel(config['model'], revision=config.get('revision'),
                               max_input_tokens=config.get('max_input_tokens', 8192),
                               device_map=config.get('device_map', 'balanced'), seed=config.get('seed', 42))

    # 使用上游文本生成协议返回候选文本和生成使用量。
    # 输入：messages（模型可见消息）；max_new_tokens；temperature；top_p；stop。
    # 相关调用：base.model.generate。
    # 返回形式含：字段 text、raw_text、input_tokens、output_tokens、model_seconds、cache_hit。
    # 异常约束：Input exceeds configured context; no extra truncation in local text backend
    def generate_text(self, messages, *, max_new_tokens, temperature, top_p, stop=None):
        from transformers import StoppingCriteria, StoppingCriteriaList
        base = self.base; torch = base.torch
        tokens = base.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_tensors='pt')
        if tokens.shape[-1] > base.max_input_tokens:
            raise ValueError('Input exceeds configured context; no extra truncation in local text backend')
        tokens = tokens.to(base.model.get_input_embeddings().weight.device)
        length = tokens.shape[-1]; stops = [stop] if isinstance(stop, str) else list(stop or [])
        if any(not isinstance(s, str) or not s for s in stops):
            raise ValueError('Nonempty text stop strings required')
        # TextStop：依据上游停止串判断是否结束文本生成。
        class TextStop(StoppingCriteria):
            # 实现调用协议，在当前模型状态下判断停止条件或返回计算结果。
            # 输入：ids；scores；**kwargs。
            def __call__(self, ids, scores, **kwargs):
                text = base.tokenizer.decode(ids[0, length:], skip_special_tokens=True)
                return any(s in text for s in stops)
        kwargs = {'max_new_tokens': max_new_tokens, 'do_sample': temperature > 0,
                  'pad_token_id': base.tokenizer.eos_token_id,
                  'stopping_criteria': StoppingCriteriaList([TextStop()]) if stops else StoppingCriteriaList()}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=0)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = base.model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), **kwargs)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        raw = base.tokenizer.decode(generated[0, length:], skip_special_tokens=True)
        boundaries = [raw.find(s) for s in stops if s in raw]
        return {'text': raw[:min(boundaries)] if boundaries else raw,
                'raw_text': raw, 'input_tokens': length,
                'output_tokens': generated.shape[-1] - length,
                'model_seconds': time.perf_counter() - started, 'cache_hit': False}


# UpstreamTextAdapter：把本地生成输出适配成上游接口需要的样本与置信信息。
class UpstreamTextAdapter:
    """Match upstream llm API and return-order confidence, recording every sample."""
    # 初始化 UpstreamTextAdapter 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：backend；max_generations。
    # 异常约束：Positive candidate generation cap required
    def __init__(self, backend, max_generations=256):
        if type(max_generations) is not int or max_generations < 1:
            raise ValueError('Positive candidate generation cap required')
        self.backend = backend; self.max_generations = max_generations
        self.calls = []; self.requests = 0

    # 向上游调用方提供当前适配器支持的解码选项。
    # 输入：temperature；per_example_max_decode_steps；per_example_top_p；n_sample。
    def get_model_options(self, temperature=0, per_example_max_decode_steps=150,
                          per_example_top_p=1, n_sample=1):
        return dict(temperature=temperature, n=n_sample, top_p=per_example_top_p,
                    max_tokens=per_example_max_decode_steps)

    # 按上游返回顺序构造候选及分数，并登记每次真实生成。
    # 输入：prompt；options；end_str。
    # 相关调用：self.backend.generate_text。
    # 返回形式含：results。
    # 异常约束：Unsupported upstream candidate or output bound
    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        options = deepcopy(options or self.get_model_options())
        n = options.get('n', 1); cap = options.get('max_tokens', 150)
        temperature = options.get('temperature', 0); top_p = options.get('top_p', 1)
        if type(n) is not int or not 1 <= n <= 8 or type(cap) is not int or not 1 <= cap <= 2048:
            raise ValueError('Unsupported upstream candidate or output bound')
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Invalid upstream temperature')
        if type(top_p) not in (int, float) or not 0 < top_p <= 1:
            raise ValueError('Invalid upstream top_p')
        if len(self.calls) + n > self.max_generations:
            raise ValueError('Total candidate generation budget exceeded')
        self.requests += 1; results = []
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        for i in range(n):
            record = {'request': self.requests, 'candidate_index': i,
                      'messages': messages, 'options': options, 'stop': end_str}
            self.calls.append(record)
            try:
                response = self.backend.generate_text(messages, max_new_tokens=cap,
                    temperature=temperature, top_p=top_p, stop=end_str)
                record.update(response, status='ok')
            except Exception as exc:
                record.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
                raise
            # This is the original return-order proxy, NOT a model likelihood.
            results.append((response['text'], math.log((n - i) / n)))
        return results

    # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
    # 输入：prompt；options；end_str。
    # 相关调用：self.generate_plus_with_score。
    def generate(self, prompt, options=None, end_str=None):
        options = deepcopy(options or self.get_model_options()); options['n'] = 1
        return self.generate_plus_with_score(prompt, options, end_str)[0][0]


# 读取上游精确声明映射，核对映射键与题面原文。
# 输入：path（输入或输出路径）。
# 相关调用：read_jsonl。
# 返回形式含：mapping。
# 异常约束：Dater mappings require nonempty statement strings
def load_dater_mapping(path):
    """Read the upstream exact-statement JSONL mapping."""
    from .data import read_jsonl

    mapping = {}
    for row in read_jsonl(path):
        raw = row['statement']
        cleaned = row['cleaned_statement']
        if not isinstance(raw, str) or not raw.strip() or not isinstance(cleaned, str) or not cleaned.strip():
            raise ValueError('Dater mappings require nonempty statement strings')
        if raw in mapping and mapping[raw] != cleaned:
            raise ValueError('Conflicting Dater mappings for the same statement')
        mapping[raw] = cleaned
    return mapping


# 提取 TabFact 推理允许使用的题面、标题和原表，隔离参考标签。
# 输入：sample（规范样本记录）；dater_mapping。
# 返回形式含：字段 statement、cleaned_statement、table_caption、table_text、chain。
# 异常约束：This upstream migration currently supports TabFact only
def public_tabfact(sample, *, dater_mapping=None):
    if sample['dataset'] != 'tabfact' or sample['task'] != 'verification':
        raise ValueError('This upstream migration currently supports TabFact only')
    # Fresh dictionaries explicitly exclude label, reference, split and highlights.
    statement = sample['question']
    cleaned = statement if dater_mapping is None else dater_mapping.get(statement, statement)
    return {'statement': statement, 'cleaned_statement': cleaned,
            'table_caption': sample.get('title', ''),
            'table_text': [list(sample['table']['headers'])] +
                          [[str(v) for v in row] for row in sample['table']['rows']], 'chain': []}


# 把上游输出按冻结的标签格式解析，格式不合规时显式失败。
# 输入：text（待解析文本）。
def parse_tabfact_text(text):
    # Upstream simple_query strips surrounding whitespace before evaluation.
    return {'yes': 1, 'true': 1, 'no': 0, 'false': 0}.get(text.strip().lower())


# 用固定上游规划和操作执行 TabFact，并记录模型迁移与格式失败。
# 输入：sample（规范样本记录）；backend；chain_module；max_generations；dater_mapping。
# 返回形式含：字段 id、dataset、mode、status、answer、failure、metrics。
# 异常约束：Final text is not an exact upstream YES/NO/TRUE/FALSE label
def predict_tabfact(sample, backend, chain_module, max_generations=256, *, dater_mapping=None):
    public = public_tabfact(sample, dater_mapping=dater_mapping)
    adapter = UpstreamTextAdapter(backend, max_generations)
    started = time.perf_counter(); failure = None; answer = None; logs = []; completed = public
    try:
        completed, logs = chain_module.dynamic_chain_exec_one_sample(public, adapter,
            llm_options=adapter.get_model_options(temperature=0, per_example_max_decode_steps=200), strategy='top')
        table_info = chain_module.get_table_info(completed)
        from operations.final_query import simple_query
        completed = simple_query(completed, table_info, adapter, use_demo=True,
            llm_options=adapter.get_model_options(temperature=0, per_example_max_decode_steps=200))
        answer = parse_tabfact_text(completed['chain'][-1]['parameter_and_conf'][0][0])
        if answer is None: raise ValueError('Final text is not an exact upstream YES/NO/TRUE/FALSE label')
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    known = all('input_tokens' in c and 'output_tokens' in c for c in adapter.calls)
    metrics = {'total_seconds': time.perf_counter() - started,
               'model_seconds': sum(c.get('model_seconds', 0) for c in adapter.calls),
               'model_calls': len(adapter.calls), 'upstream_api_requests': adapter.requests,
               'input_tokens': sum(c['input_tokens'] for c in adapter.calls) if known else None,
               'output_tokens': sum(c['output_tokens'] for c in adapter.calls) if known else None,
               'cache_hits': 0, 'repairs': 0, 'latency_comparable': False}
    mode = 'CoT-upstream-Qwen-raw' if dater_mapping is None else 'CoT-upstream-Qwen-Dater'
    preprocessing = 'raw_statement_no_Dater_map' if dater_mapping is None else 'Dater_exact_lookup_add_column'
    return {'id': sample['id'], 'dataset': 'tabfact', 'mode': mode,
            'status': 'ok' if failure is None else 'failed', 'answer': answer,
            'failure': failure, 'metrics': metrics, 'responses': adapter.calls,
            'upstream_chain': completed.get('chain', []), 'planning_logs': logs,
            'operations': [], 'trace': [], 'evidence': [], 'executable_answer': False,
            'semantic_correctness': 'not_proven', 'upstream_commit': PINNED_COMMIT,
            'preprocessing': preprocessing,
            'dater_map_hit': dater_mapping is not None and public['statement'] in dater_mapping,
            'dater_text_changed': public['statement'] != public['cleaned_statement'],
            'candidate_execution': 'sequential_local_samples',
            'limits': 'TabFact-only source migration; different model and preprocessing; not full paper reproduction or formal speed benchmark.'}
