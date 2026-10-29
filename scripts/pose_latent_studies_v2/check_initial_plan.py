"""Protocol checks without simulator truth or formal evaluation samples."""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.sources.resolve()))
    from initial_plan import InitialPlanExecutor,validate_initial_plan,plan_tools_schema
    actions=[dict(tool='inspect_face',arguments=dict(color='cyan',steps=10,evidence='Initial observation')),
        dict(tool='reorient_target',arguments=dict(color='red',goal_symbol='T',steps=90,evidence='Initial observation')),
        dict(tool='stop',arguments=dict(color='red',goal_symbol='T',evidence='Initial evidence only'))]
    args={'actions':actions};validate_initial_plan('execute_initial_plan',args)
    executor=InitialPlanExecutor();executor.accept('execute_initial_plan',args,120)
    assert executor.pop_due(119) is None
    assert executor.pop_due(120)['tool']=='inspect_face'
    assert executor.pop_due(129) is None
    assert executor.pop_due(130)['tool']=='reorient_target'
    assert executor.pop_due(219) is None
    assert executor.pop_due(220)['tool']=='stop'
    assert executor.pop_due(3600) is None
    try:executor.accept('execute_initial_plan',args,3601)
    except ValueError:pass
    else:raise AssertionError('Plan replacement accepted')
    rejected=[]
    for name,mutation in [('empty',{'actions':[]}),('too_long',{'actions':actions*3}),
            ('unknown_tool',{'actions':[dict(tool='get_true_pose',arguments={})]}),
            ('boolean_steps',{'actions':[dict(tool='inspect_face',arguments=dict(color='red',steps=True,evidence='x'))]}),
            ('abstain_then_move',{'actions':[dict(tool='abstain',arguments=dict(evidence='x')),actions[0]]}),
            ('extra_hidden_feedback',{'actions':actions,'true_goal':0})]:
        try:validate_initial_plan('execute_initial_plan',mutation)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    schema=plan_tools_schema()
    # If jsonschema is available, check real tool examples against the advertised schema.
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(schema[0]['parameters'])
    jsonschema.validate(args,schema[0]['parameters'])
    for path in a.sources.glob('*.py'):ast.parse(path.read_text())
    result={'completed':True,'scheduled_steps':[120,130,220],'second_plan_rejected':True,
        'malformed_plans_rejected':rejected,'schema_accepts_actual_examples':True,
        'source_sha256':{name:hashlib.sha256((a.sources/name).read_bytes()).hexdigest()
            for name in ('initial_plan.py','astra_initial_plan_bridge.py','evaluate_agent.py')},
        'scope':'Protocol/executor unit checks only; no claim of simulator or live API success.'}
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'protocol_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
