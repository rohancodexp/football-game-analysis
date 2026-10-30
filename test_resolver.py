import unittest
import numpy as np
from run import resolve
class Tests(unittest.TestCase):
    def test_unknown_without_references(self):
        f={'fragment_id':'F1','numbers':[], 'embedding':np.array([1.,0.]),'start_s':0,'end_s':1,'start_frame':0,'end_frame':30}
        resolve([f],{'players':[]},{},.88,.08)
        self.assertEqual(f['player_id'],'UNK')
    def test_overlap_prevents_duplicate_identity(self):
        fs=[{'fragment_id':str(i),'numbers':[], 'embedding':np.array([1.,0.]),'start_s':0,'end_s':1,'start_frame':0,'end_frame':30} for i in range(2)]
        roster={'players':[{'player_id':'P1','team_id':'HOME','jersey':10}]}
        resolve(fs,roster,{'P1':np.array([1.,0.])},.88,.08)
        self.assertEqual([f['player_id'] for f in fs],['P1','UNK'])
if __name__=='__main__': unittest.main()
