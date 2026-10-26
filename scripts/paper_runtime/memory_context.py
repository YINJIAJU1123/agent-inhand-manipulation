"""Observed feedback adapter; preserves the fine-tuned model's dense output."""
import copy,hashlib,json
PUBLIC_KEYS={'remaining_calls','remaining_sim_seconds','actual_executed_skill','actual_executed_state','local_stop_proposal_pending','local_pose_verifier','learned_semantic_memory'}
FEEDBACK_KEYS={'step','image_change','proprio_change','visually_quiet_ticks'}

class ContextPredictor:
    def __init__(self,backend):
        self.backend=backend;self.context={};self.owner=None
    def capture(self,context):
        if set(context)-PUBLIC_KEYS:raise ValueError('Unexpected public context fields')
        if 'learned_semantic_memory' not in context:raise ValueError('Expected learned RGB memory')
        self.context=copy.deepcopy(context)
    def predict(self,rgb,instruction):
        if self.owner is None:raise RuntimeError('Missing agent owner')
        feedback=copy.deepcopy(self.owner.feedback_trace[-1])
        if set(feedback)-FEEDBACK_KEYS:raise ValueError('Unexpected measured feedback fields')
        feedback.update(previous_skill=self.owner.last_decision.color,previous_state=self.owner.last_decision.state)
        feedback.update(copy.deepcopy(self.context))
        # This is observed evidence, not a target-color lookup or an action override.
        augmented=instruction+'\n\nMeasured feedback and historical visual evidence (model estimates, not ground truth):\n'+json.dumps(feedback,ensure_ascii=False,sort_keys=True)
        result=dict(self.backend.predict(rgb,augmented))
        result.update(observed_context=feedback,conditioning_instruction=augmented,conditioning_sha256=hashlib.sha256(augmented.encode()).hexdigest(),context_scope='Same public feedback fields as Astra. Existing fine-tuned weights and continuous goal prediction are unchanged; no further context-specific fine-tuning.')
        return result
