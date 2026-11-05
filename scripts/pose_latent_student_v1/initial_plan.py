"""One-observation Astra baseline with a complete bounded skill sequence.

This executor never reads images, pose estimates, truth or task success. It
schedules the initial plan by control steps. The shared local stopping gate
remains in the evaluator, with the same sensor inputs as the feedback arm.
"""
from collections import deque
import copy
from semantic import SYSTEM,tools_schema,validate_action

INITIAL_PLAN_SYSTEM = SYSTEM + '''
For this experiment you receive exactly one initial observation and its
measured feedback. You will not receive any later image or model call.
Return execute_initial_plan containing between one and six ordered actions.
This must be your complete plan for this episode, not a request for another
model response. Each action uses the same color skills and bounded durations.
Actions run sequentially for their declared duration. verify_target and stop
last 30 steps, and the last non-abstain skill continues if the local measured
stopping gate has not accepted it. The local controller and measured stopping
gate still operate continuously. They do not read new letters or repair your
letter-color association. An inspect action changes orientation but does not
provide you with another image. Do not present unseen letters as observations.
Explain any uncertainty in evidence. You may abstain if the initial evidence
cannot support a target. No later visual correction is available.'''


def plan_tools_schema():
    variants=[]
    for tool in tools_schema():
        variants.append(dict(type='object',properties={
            'tool':dict(type='string',enum=[tool['name']]),
            'arguments':copy.deepcopy(tool['parameters'])},required=['tool','arguments'],additionalProperties=False))
    return [dict(type='function',name='execute_initial_plan',
        description='Commit a complete initial plan, without later model observations.',strict=True,
        parameters=dict(type='object',properties={'actions':dict(type='array',minItems=1,maxItems=6,
            items={'anyOf':variants})},required=['actions'],additionalProperties=False))]


def validate_initial_plan(name,arguments):
    if name!='execute_initial_plan' or not isinstance(arguments,dict) or set(arguments)!={'actions'}:
        raise ValueError('Expected a complete initial-plan tool')
    actions=arguments['actions']
    if not isinstance(actions,list) or not 1<=len(actions)<=6:
        raise ValueError('Initial plan requires one to six bounded actions')
    for i,item in enumerate(actions):
        if not isinstance(item,dict) or set(item)!={'tool','arguments'}:
            raise ValueError('Malformed plan action')
        validate_action(item['tool'],item['arguments'])
        if item['tool']=='abstain' and i!=len(actions)-1:
            raise ValueError('Abstain must terminate the plan')
    return arguments


class InitialPlanExecutor:
    def __init__(self):
        self._actions=None
        self.next_step=None
        self.dispatched=0

    def accept(self,name,arguments,delivery_step):
        if self._actions is not None:raise ValueError('A single-observation plan cannot be replaced')
        if type(delivery_step) is not int or delivery_step<0:raise ValueError('Invalid delivery step')
        validate_initial_plan(name,arguments)
        self._actions=deque(copy.deepcopy(arguments['actions']))
        self.next_step=delivery_step

    def pop_due(self,step):
        if self._actions is None or not self._actions or step<self.next_step:return None
        item=self._actions.popleft();self.dispatched+=1
        self.next_step=step+item['arguments'].get('steps',30)
        return item
