"""Standalone scientific PNG/SVG figures from a completed pilot, no model calls."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');args=p.parse_args();root=Path(args.root)
    report=json.loads((root/'evaluation'/'report.json').read_text())
    if not (root/'COMPLETE').exists():raise ValueError('Only completed experiments may be plotted')
    out=root/'figures';out.mkdir(exist_ok=True)
    names=['P','R','M'];colors=['#777777','#3978ac','#15927e']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
    x=np.arange(2);width=.23
    for i,name in enumerate(names):
        scores={r['dataset']:r['observed_score'] for r in report['development'][name]['rows']}
        values=[100*scores[d] for d in ['wikitq','tabfact']]
        bars=axes[0,0].bar(x+(i-1)*width,values,width,color=colors[i],label=name)
        axes[0,0].bar_label(bars,fmt='%.0f',padding=2,fontsize=9)
    axes[0,0].set(xticks=x,xticklabels=['WikiTQ (n=20)','TabFact (n=20)'],ylim=(0,105),ylabel='Accuracy (%)',
                  title='A. Previously used development subset')
    axes[0,0].legend(frameon=False,ncol=3)
    delta=report['comparisons']['R-vs-M']
    for i,d in enumerate(['wikitq','tabfact']):
        row=next(r for r in delta if r['dataset']==d);mean=100*row['difference_b_minus_a'];lo,hi=[100*v for v in row['ci95']]
        axes[0,1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[2],capsize=5)
    axes[0,1].axvline(0,color='#777777',linewidth=1,linestyle='--')
    axes[0,1].set(yticks=[0,1],yticklabels=['WikiTQ','TabFact'],ylim=(-.6,1.6),xlabel='M - R (percentage points)',
                  title='B. Paired example bootstrap 95% intervals')
    families=list(report['synthetic']['P']['by_family']);positions=np.arange(len(families))
    labels={'affiliation_count':'Affiliation count','ordinal_scope':'Rank scope','round_mean':'Round mean',
            'unit_extreme':'Mixed units','crossrow_difference':'Cross-row difference','universal_scope':'Universal scope'}
    for i,name in enumerate(names):
        values=[100*report['synthetic'][name]['by_family'][family]['accuracy'] for family in families]
        axes[1,0].barh(positions+(i-1)*width,values,height=width,color=colors[i],label=name)
    axes[1,0].set(yticks=positions,yticklabels=[labels[f] for f in families],xlim=(0,105),xlabel='Exact/template accuracy (%)',
                  title='C. New synthetic instances (n=4 per family)')
    axes[1,0].invert_yaxis()
    for name,color in [('R',colors[1]),('M',colors[2])]:
        history=json.loads((root/f'training-{name}'/'adapter'/'training_history.json').read_text())
        logs=[r for r in history if 'loss' in r and 'step' in r]
        axes[1,1].plot([r['step'] for r in logs],[r['loss'] for r in logs],marker='o',color=color,label=name)
    axes[1,1].set(xlabel='Optimizer step',ylabel='Logged training loss',title='D. One epoch, one seed; loss is not task quality')
    axes[1,1].legend(frameon=False)
    fig.suptitle('Controlled curriculum pilot: matched target tokens and steps',fontsize=14)
    fig.supxlabel('P: original adapter   R: real replay control   M: synthetic curriculum\nDevelopment only; synthetic instances share training templates; no formal speed or superiority claim.',fontsize=9)
    fig.savefig(out/'curriculum-pilot.png',dpi=300)
    fig.savefig(out/'curriculum-pilot.svg')
    plt.close(fig)
    print(str(out/'curriculum-pilot.png'))


if __name__=='__main__':main()
