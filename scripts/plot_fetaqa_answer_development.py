"""Standalone scientific figure from the completed automatic-metric report."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results');args=p.parse_args();root=Path(args.results)
    if not (root/'COMPLETE').exists():raise ValueError('Wait for complete failure-inclusive evaluation')
    report=json.loads((root/'report.json').read_text());n=report['conditions']['QD']['n']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=['sacrebleu','rouge1_f1','rouge2_f1','rougeL_f1'];names=['BLEU','ROUGE-1','ROUGE-2','ROUGE-L']
    scales=[1,100,100,100];x=list(range(4));fig,axes=plt.subplots(1,2,figsize=(11.2,4.5))
    for offset,name,color,label in [(-.18,'QD','#8c9aaf','Base 3B'),(.18,'SD','#287e8e','512-answer LoRA')]:
        values=[report['conditions'][name][key]*scale for key,scale in zip(metrics,scales)]
        axes[0].bar([i+offset for i in x],values,.34,color=color,label=label)
        for i,value in enumerate(values):axes[0].text(i+offset,value+1,f'{value:.1f}',ha='center',fontsize=8)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('BLEU; ROUGE F1 multiplied by 100');axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title('Automatic answer overlap')
    comparison=report['comparison_QD_to_SD']['metrics']
    for i,(key,scale) in enumerate(zip(metrics,scales)):
        item=comparison[key];low,high=[v*scale for v in item['ci95']]
        axes[1].vlines(i,low,high,color='#287e8e',linewidth=2)
        axes[1].scatter(i,item['difference_b_minus_a']*scale,color='#287e8e',zorder=3)
        axes[1].hlines([low,high],i-.08,i+.08,color='#287e8e')
    axes[1].axhline(0,color='#6d7480',linewidth=.8,linestyle='--')
    axes[1].set_ylabel('LoRA minus base, same score units');axes[1].set_title('Paired sample bootstrap 95% intervals')
    for ax in axes:
        ax.set_xticks(x,names);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'FeTaQA B0 answer supervision: {n} additional development examples',fontsize=12)
    fig.text(.5,.035,'One training seed / epoch; failures and non-text outputs included. No human factuality evaluation.\n'
             'Fixed 512-answer checkpoint; development only, not a Chain-of-Table test comparison.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.82,wspace=.35)
    out=root/'figures';out.mkdir(exist_ok=True)
    for suffix in ['png','svg']:fig.savefig(out/f'fetaqa-answer-development.{suffix}',dpi=200)
    plt.close(fig)


if __name__=='__main__':main()
