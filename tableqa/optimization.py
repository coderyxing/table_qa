"""Training-time deletion search and state-cost partitioning, never a test oracle."""
from copy import deepcopy
from itertools import combinations
import json
import math
from .core import TableState, run_plan, ExecutionError


def verify_subsequence_rewrite(state, operations, candidate):
    """Validate externally prepared targets before the record builder uses them."""
    cursor=0
    for op in candidate:
        while cursor<len(operations) and operations[cursor]!=op:cursor+=1
        if cursor==len(operations):raise ValueError('Prepared operations are not an ordered source subsequence')
        cursor+=1
    if [o for o in operations if o.get('requires_feedback')]!=[o for o in candidate if o.get('requires_feedback')]:
        raise ValueError('Prepared operations removed a feedback boundary')
    if run_plan(state,operations)[0].signature()!=run_plan(state,candidate)[0].signature():
        raise ValueError('Prepared operations changed execution/provenance')


def optimize_operations(state, operations, max_candidates=4096, preserve_kinds=()):
    """Minimize length within subsequences that preserve the full execution signature.

    Answer and explicit feedback operations are protected. The search is bounded;
    optimality is reported only if the minimal-length candidate tier was exhausted.
    No reference answer or task score is an input to this optimization.
    """
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('Positive integer candidate budget required')
    original, _ = run_plan(state, operations)
    signature = original.signature()
    protected=set(preserve_kinds)|{'answer'}
    required = {i for i, op in enumerate(operations) if op['op'] in protected or op.get('requires_feedback')}
    optional = [i for i in range(len(operations)) if i not in required]
    best = deepcopy(operations); best_indices = list(range(len(operations))); attempted = 0
    objective = lambda ops: (len(ops), len(json.dumps(ops, ensure_ascii=False, sort_keys=True)))
    certified = False; exhausted = False
    for size in range(len(optional)+1):
        found = False
        for subset in combinations(optional, size):
            if attempted >= max_candidates:
                exhausted = True; break
            indices = sorted(required | set(subset)); candidate = [operations[i] for i in indices]
            attempted += 1
            try:
                final, _ = run_plan(state, candidate)
            except (ExecutionError, ValueError, KeyError, TypeError):
                continue
            if final.signature() != signature: continue
            found = True
            if objective(candidate) < objective(best):
                best = deepcopy(candidate); best_indices = indices
        if exhausted: break
        if found:
            certified = True; break
    # Recheck the returned object independently before it can become supervision.
    if run_plan(state, best)[0].signature() != signature: raise ValueError('Compression invariant violated')
    return best, {'policy': 'replay_subsequence_search', 'candidate_attempts': attempted,
                  'candidate_budget_exhausted': exhausted, 'optimal_in_deletion_family': certified,
                  'objective': 'operation count, then serialized JSON characters',
                  'original_operations': len(operations), 'optimized_operations': len(best),
                  'protected_operation_kinds':sorted(protected),
                  'retained_indices': best_indices, 'signature_equal': True,
                  'limits': 'Equality on this table, not natural-language semantic proof or unrestricted program optimality.'}


def effective_fixed_blocks(operations, max_size=4):
    """The fixed control after separating answer supervision, as build_records does."""
    from .distillation import blocks
    out = []
    for group in blocks(operations, max_size):
        if group[-1]['op'] == 'answer':
            if len(group) > 1: out.append(group[:-1])
            out.append(group[-1:])
        else: out.append(group)
    return out


def state_cost_partition(sample, operations, tokenizer, max_size=4, max_length=4096,
                         input_weight=1., output_weight=1., call_overhead=128.,
                         state_sensitive=True, allow_validation=False, max_output_tokens=512,
                         typed_quantities=False, semantic_extract=False, tool_demonstrations=False):
    """Shortest-path teacher partition; students learn termination from the targets.

    Only source training/explicit validation trajectories are allowed. At test time
    the student predicts blocks; this DP cannot access a future teacher trajectory.
    Weights define a token-cost proxy, not measured GPU seconds.
    """
    from .inference import context, system_prompt
    from .quantities import uses_quantities
    from .semantic_columns import uses_extraction
    if uses_quantities(operations) and not typed_quantities:
        raise ValueError('Quantity partition requires the matching typed_quantities prompt')
    if uses_extraction(operations) and not semantic_extract:
        raise ValueError('Extraction partition requires the matching semantic_extract prompt')
    permitted = {'train', 'validation'} if allow_validation else {'train'}
    if sample['split'] not in permitted: raise ValueError('Teacher partitioning requires an allowed training/validation split')
    if type(max_size) is not int or max_size < 1 or type(max_length) is not int or max_length < 1:
        raise ValueError('Positive integer block/context bounds required')
    if type(max_output_tokens) is not int or max_output_tokens<1:raise ValueError('Positive output-token bound required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in (input_weight,output_weight,call_overhead)):
        raise ValueError('Cost weights must be finite and nonnegative')
    if not operations or operations[-1].get('op') != 'answer': raise ValueError('Complete teacher trajectory required')
    state = TableState.from_table(sample['table']); states = [state]
    for op in operations:
        state, _ = run_plan(state, [op]); states.append(state)
    # Validate cross-operation dependencies too, including premature projections.
    run_plan(states[0], operations)
    n = len(operations); edges = {}
    for i in range(n):
        def input_tokens(view_state):
            messages = [{'role':'system','content':system_prompt('B5', typed_quantities, semantic_extract,
                        tool_demonstrations)}, {'role':'user','content':
                context(sample,view_state,operations[:i],mode='B5',max_block=max_size)}]
            return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        actual_input = input_tokens(states[i])
        objective_input = actual_input if state_sensitive else input_tokens(states[0])
        for j in range(i+1,min(n,i+max_size)+1):
            group = operations[i:j]
            if any(o.get('requires_feedback') for o in group[1:]): continue
            if any(o['op']=='answer' for o in group) and len(group)!=1: continue
            target = json.dumps({'operations':group},ensure_ascii=False)
            output_tokens = len(tokenizer.encode(target,add_special_tokens=False))+1  # EOS
            if actual_input + output_tokens > max_length or output_tokens>max_output_tokens: continue
            edges[(i,j)] = {'start':i,'end':j,'input_tokens':actual_input,'objective_input_tokens':objective_input,
                'output_tokens':output_tokens,'cost':input_weight*objective_input+output_weight*output_tokens+call_overhead}
    distance = [math.inf]*(n+1); distance[0]=0.; paths = {0:()}
    for j in range(1,n+1):
        options = [(distance[i]+edges[(i,j)]['cost'],len(paths[i])+1,paths[i]+(j,))
                   for i in range(max(0,j-max_size),j) if (i,j) in edges and i in paths]
        if options:
            distance[j],_,paths[j]=min(options)
    if n not in paths: raise ValueError('No feasible teacher partition; no truncation or overlength target allowed')
    bounds = [0]+list(paths[n]); grouped = [deepcopy(operations[i:j]) for i,j in zip(bounds,bounds[1:])]
    selected_edges = [edges[(i,j)] for i,j in zip(bounds,bounds[1:])]
    fixed = effective_fixed_blocks(operations,max_size); cursor=0; fixed_edges=[]
    for group in fixed:
        edge=edges.get((cursor,cursor+len(group)));fixed_edges.append(edge);cursor+=len(group)
    fixed_cost=sum(e['cost'] for e in fixed_edges) if all(e is not None for e in fixed_edges) else None
    return grouped, {'policy':'state_cost_dp' if state_sensitive else 'state_blind_cost_dp',
        'boundaries':bounds,'cost':distance[n],'fixed_control_cost':fixed_cost,'selected_edges':selected_edges,
        'feasible_edges':[edges[k] for k in sorted(edges)],
        'optimal_in_feasible_partition_space':True,'uses_future_teacher_trajectory_offline':True,
        'cost_units':'weighted token proxy, NOT GPU seconds','input_weight':input_weight,'output_weight':output_weight,
        'call_overhead':call_overhead,'max_block':max_size,'max_length':max_length,
        'max_output_tokens':max_output_tokens,
        'online_contract':'Student generates block termination from current state; no oracle teacher plan is available at test time.'}
