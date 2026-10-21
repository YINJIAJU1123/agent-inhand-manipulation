"""Live Astra requests with controlled extra simulated response delay.

Only requests with an already delivered skill receive the extra delay. Cold
start is identical across arms. No sleep or Future wait on the physics thread.
"""
import copy
import math
import time


class AsyncAgent:
    def __init__(self, agent, pool, mode, hz=30, extra_delay_s=0.):
        self.agent, self.pool, self.mode, self.hz = agent, pool, mode, hz
        if extra_delay_s < 0:raise ValueError("Negative injected delay")
        self.extra_delay_s=float(extra_delay_s)
        self.future = None
        self.snapshot = dict(decision=dict(color=None,state='initial',reason=''), goal=None,
                             goal_symbol=None, goal_embedding=None, calls=0)
        self.due = 0
        self.events = []
        self.closed = False
        self.submitted = 0

    def submit(self, rgb, proprio, step):
        if self.closed or self.future is not None or step < self.due or self.submitted >= self.agent.budget:
            return False
        self.request_step, self.request_wall = step, time.monotonic()
        self.submitted += 1
        extra=self.extra_delay_s if self.snapshot["decision"]["color"] is not None else 0.
        self.event = dict(injected_delay_s=extra,sequence=self.submitted,request_step=step,status='pending',physics_steps=0)
        self.events.append(self.event)
        # Only this worker may touch agent state until its Future completes.
        def work():
            try:
                decision = self.agent.tick(rgb, proprio, step)
                result = dict(decision=vars(decision).copy(),goal=self.agent.goal,
                              goal_symbol=self.agent.goal_symbol,goal_embedding=self.agent.goal_embedding,
                              calls=self.agent.calls)
                return dict(snapshot=copy.deepcopy(result),finished=time.monotonic(),error=None)
            except Exception as error:
                return dict(snapshot=None,finished=time.monotonic(),error=type(error).__name__+': '+str(error))
        self.future = self.pool.submit(work)
        return True

    def poll(self, step):
        if self.future is None or not self.future.done():
            return False
        result = self.future.result()  # Always guarded by done(); never waits for inference.
        latency = result['finished'] - self.request_wall
        delivery_step = self.request_step + max(1, math.ceil((latency+self.event['injected_delay_s']) * self.hz))
        self.event.update(wall_latency_s=latency,effective_delay_s=latency+self.event['injected_delay_s'],earliest_delivery_step=delivery_step,error=result['error'])
        # A slow renderer must not turn 10 seconds of API latency into 2 seconds of physics.
        if not self.closed and step < delivery_step:
            return False
        self.future = None
        self.event.update(delivery_step=step,status='discarded_terminal' if self.closed else 'error' if result['error'] else 'delivered',sim_wait_s=(step-self.request_step)/self.hz)
        if self.closed:
            return False
        if result['error']:
            self.closed = True
            return False
        self.snapshot = result['snapshot']
        if self.mode == 'astra':
            duration = self.agent.next_step - self.request_step
            self.agent.next_step = step + duration
            self.due = self.agent.next_step
        else:
            self.agent.last_call = step
            self.agent.inspect_since = step
            self.due = step + 90
        return True

    def advanced(self):
        if self.future is not None:
            self.event['physics_steps'] += 1

    def close(self, step):
        self.closed = True
        if self.future is not None:
            self.event.update(status='terminal_pending',terminal_step=step)
            self.poll(step)
