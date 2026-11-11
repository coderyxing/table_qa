"""Standalone, source-bound development figures; no final thesis-test claim."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('analysis');p.add_argument('mechanisms');p.add_argument('output')
args=p.parse_args()
analysis=json.loads(Path(args.analysis).read_text(encoding='utf-8'))
mechanisms=json.loads(Path(args.mechanisms).read_text(encoding='utf-8'))
if analysis['runs']['B']['predictions_structured_sha256']!=mechanisms['predictions_sha256']:
    raise ValueError('Analysis and mechanism sources differ')
if any(r['split']!=['train'] for run in analysis['runs'].values() for r in run['rows']):
    raise ValueError('This figure template is explicitly for source-training development results')
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})

def save(fig,name):
    for extension in ['png','pdf','svg']:
        fig.savefig(root/f'{name}.{extension}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9,4))
for ax,dataset,title in zip(axes,['wikitq','tabfact'],['WikiTQ: official denotation','TabFact: label accuracy']):
    rows=[next(r for r in analysis['runs'][run]['rows'] if r['dataset']==dataset) for run in ['A','B']]
    heights=[r['observed_score']*100 for r in rows]
    bars=ax.bar(['A0: direct','A4: executable chain'],heights,color=['#496e9c','#c18045'],width=.55)
    ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title(f'{title}\nn = {rows[0]["completed"]} paired examples')
    ax.grid(axis='y',color='#e5e7eb')
    for bar,value in zip(bars,heights):ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}%',ha='center')
fig.suptitle('Development comparison: A4 currently underperforms A0',fontsize=13,y=1.03)
fig.text(.5,-.03,'Source-training development samples; failures count as incorrect. Not held-out thesis-test results.',ha='center',fontsize=9)
fig.tight_layout();save(fig,'teacher_quality_development')

by_dataset={r['dataset']:r for r in mechanisms['rows']}
datasets=['wikitq','tabfact','fetaqa']
categories=sorted({k for r in mechanisms['rows'] for k in r['final_failure_categories']})
fig,ax=plt.subplots(figsize=(12,5.2))
fig.subplots_adjust(left=.10,right=.99,top=.90,bottom=.37)
positions=list(range(len(datasets)));left=[0]*len(datasets)
for index,category in enumerate(['execution_ok']+categories):
    values=[by_dataset[d]['counts'].get('execution_ok',0) if category=='execution_ok'
            else by_dataset[d]['final_failure_categories'].get(category,0) for d in datasets]
    color='#447c6b' if category=='execution_ok' else plt.get_cmap('tab20')(index)
    bars=ax.barh(positions,values,left=left,color=color,label=category.replace('_',' '),height=.6)
    for bar,value in zip(bars,values):
        if value>=8:ax.text(bar.get_x()+value/2,bar.get_y()+bar.get_height()/2,str(value),ha='center',va='center',fontsize=9)
    left=[a+b for a,b in zip(left,values)]
if left!=[by_dataset[d]['n'] for d in datasets]:raise ValueError('Failure categories do not cover the complete prediction set')
ax.set_yticks(positions,['WikiTQ','TabFact','FeTaQA']);ax.invert_yaxis()
ax.set_xlabel('Number of examples (200 per task)');ax.set_title('A4 observed execution outcomes on the frozen development cohort')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=4,frameon=False,fontsize=9)
fig.text(.5,.035,'Executable does not imply semantically correct. Categories describe final observed runtime failures.',ha='center',fontsize=9)
save(fig,'teacher_execution_outcomes_development')
(root/'figure-sources.json').write_text(json.dumps({'development_only':True,
    'samples_sha256':mechanisms['samples_sha256'],'A4_predictions_sha256':mechanisms['predictions_sha256'],
    'analysis':str(Path(args.analysis)),'mechanisms':str(Path(args.mechanisms))},indent=2))
print('DEVELOPMENT_FIGURES_SAVED',str(root),flush=True)
