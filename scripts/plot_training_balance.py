# 脚本用途：展示训练目标权重与实际监督分布，供损失审计使用。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Training objective-weight audit, not a performance figure."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p=argparse.ArgumentParser();p.add_argument('preflight');p.add_argument('output');a=p.parse_args()
raw=Path(a.preflight).read_bytes();data=json.loads(raw);u,w=data['U'],data['W']
if u['sampling_sequence_sha256']!=w['sampling_sequence_sha256']:raise ValueError('Sampling differs')
before=Counter();after=Counter();tf_before=Counter();tf_after=Counter()
for row in u['source_exposure']:before[row['dataset']]+=row['record_exposures']
for row in w['loss_weight_strata']:
    after[row['dataset']]+=row['weighted_mass']
    if row['dataset']=='tabfact':tf_before[row['label']]+=row['records'];tf_after[row['label']]+=row['weighted_mass']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
fig,axes=plt.subplots(1,2,figsize=(9,4.3));fig.subplots_adjust(bottom=.21,top=.77,wspace=.33)
for ax,labels,keys,before_count,after_count,title in [
    (axes[0],['WikiTQ','TabFact'],['wikitq','tabfact'],before,after,'Weight across tasks'),
    (axes[1],['Refuted (0)','Entailed (1)'],[0,1],tf_before,tf_after,'Weight within TabFact')]:
    for offset,counts,color,name in [(-.18,before_count,'#a2b2c7','U: unweighted'),(.18,after_count,'#437b6b','W: balanced')]:
        values=[counts[k]/sum(counts.values())*100 for k in keys]
        bars=ax.bar([i+offset for i in range(len(keys))],values,width=.34,color=color,label=name)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(range(len(keys)),labels);ax.set_ylim(0,100);ax.set_ylabel('Share of record weight (%)')
    ax.set_title(title);ax.grid(axis='y',color='#e5e7eb')
fig.suptitle('Same training records and tokens; different objective weighting',fontsize=12,y=.97)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2,frameon=False)
fig.text(.5,.02,'172 training sources, 372 sampled records, 20,754 target tokens/epoch. Not model accuracy.',ha='center',fontsize=9)
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(root/f'training_weight_balance.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
(root/'sources.json').write_text(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),
    'sampling_sequence_sha256':u['sampling_sequence_sha256'],'performance_claim':False},indent=2))
print('TRAINING_WEIGHT_FIGURE_SAVED',root)
