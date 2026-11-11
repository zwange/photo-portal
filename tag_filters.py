"""Conservative tag variants; original classifier tags stay untouched."""
import re

ALIASES={'indoor':'indoors','outdoor':'outdoors','closeup':'close up','close ups':'close up',
 'people':'person','persons':'person','men':'man','women':'woman','children':'child',
 'babies':'baby','mice':'mouse','geese':'goose','feet':'foot','teeth':'tooth',
 'leaves':'leaf','wolves':'wolf','calves':'calf','sheep':'sheep','fish':'fish',
 'rocky formation':'rock formation','rocky formations':'rock formation','rock formations':'rock formation'}
PROTECTED={'grass','glass','glasses','sunglasses','stairs','shorts','pants','jeans','woods','news','species','clothes','indoors','outdoors','sports','bus','lens','cactus','christmas'}

def normalized(tag):return ' '.join(re.sub(r'[_-]+',' ',str(tag).casefold()).split())

def canonical(tag,vocabulary):
    tag=normalized(tag)
    if tag in ALIASES:return ALIASES[tag]
    words=tag.split()
    if not words:return tag
    last=words[-1]
    if last in ALIASES:return ' '.join(words[:-1]+[ALIASES[last]])
    if last in PROTECTED:return tag
    candidates=[]
    if last.endswith('ies'):candidates.append(last[:-3]+'y')
    if last.endswith(('ches','shes','xes','zes','sses')):candidates.append(last[:-2])
    if last.endswith('s') and not last.endswith(('ss','us','is')):candidates.append(last[:-1])
    # Merge regular plurals only when the complete singular tag is actually present.
    for candidate in candidates:
        value=' '.join(words[:-1]+[candidate])
        if value in vocabulary:return value
    return tag
