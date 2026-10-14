"""Actual text-only VLM instruction probes; independent of rollout budgets."""
from semantic_agent_v3 import SemanticVLMClient, SYMBOLS
from semantic_protocol_v3 import make_cases


def probe(client):
    cases=[]
    for template in ('canonical','paraphrase'):
        for case in make_cases(31001,0,template):
            cases.append({'instruction':case['instruction'],'expected_symbol':case['target_symbol'],
                          'expected_direction':'up','group':template})
    for instruction in ('Turn the star sticker toward the left.',
                        'Turn either the star or the circle upward.',
                        'Make the sticker without a star face upward.'):
        cases.append({'instruction':instruction,'expected_symbol':None,'expected_direction':None,'group':'unsupported_or_ambiguous'})
    rows=[]
    for case in cases:
        result=client.parse_instruction(case['instruction'])
        accepted=result['goal_symbol'] in SYMBOLS and result['goal_direction']=='up'
        correct=(result['goal_symbol']==case['expected_symbol'] and result['goal_direction']=='up') if case['expected_symbol'] else not accepted
        rows.append(dict(case,result=result,accepted=accepted,correct=correct))
    return {'kind':'real_vlm_text_only_diagnostic','calls':len(rows),'correct':sum(r['correct'] for r in rows),'rows':rows}
