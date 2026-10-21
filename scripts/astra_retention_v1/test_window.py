import unittest
from window import RetentionWindow

class WindowTests(unittest.TestCase):
 def call(self,w,**kw):
  args=dict(step=10,pending=True,grounded_command=True,goal=2,ready=True,fresh=True,selected_rate=0.);args.update(kw);return w.update(**args)
 def test_no_unknown_or_inspection_override(self):
  for kw in (dict(goal=None),dict(grounded_command=False),dict(pending=False),dict(fresh=False)):
   self.assertIsNone(self.call(RetentionWindow(),**kw))
 def test_reply_releases_before_old_rate_can_act(self):
  w=RetentionWindow();self.assertEqual(self.call(w),0.)
  self.assertIsNone(self.call(w,step=11,pending=False));self.assertIsNone(w.goal)
 def test_changed_goal_cannot_inherit_hold(self):
  w=RetentionWindow();self.call(w)
  self.assertIsNone(self.call(w,step=11,goal=4,ready=False));self.assertIsNone(w.rate)
 def test_window_does_not_extend_itself_while_waiting(self):
  w=RetentionWindow();self.call(w)
  self.assertEqual(self.call(w,step=189,ready=False),0.)
  self.assertIsNone(self.call(w,step=190));self.assertIsNone(self.call(w,step=191))
  self.call(w,step=192,pending=False)
  self.assertEqual(self.call(w,step=193,selected_rate=.25),.25)

if __name__=='__main__':unittest.main()
