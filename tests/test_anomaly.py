"""Boundary checks for causal anomaly windows and alert logic."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from predictive_maintenance.anomaly import windows, causal_mean, sustained_flags, validate_frame, lead_times


class AnomalyTests(unittest.TestCase):
    def test_windows_stop_at_current_cycle_and_never_cross_engines(self):
        frame = pd.DataFrame({'unit_number':[1]*31+[2]*30,'time_in_cycles':list(range(1,32))+list(range(1,31))})
        x=np.r_[np.arange(31),100+np.arange(30)].reshape(-1,1)
        matrix,ends=windows(frame,x,30)
        np.testing.assert_array_equal(ends,[29,30,60])
        np.testing.assert_array_equal(matrix[0],np.arange(30))
        np.testing.assert_array_equal(matrix[-1],100+np.arange(30))
        self.assertEqual(len(windows(frame.iloc[:29],x[:29],30)[0]),0)

    def test_smoothing_uses_only_past_and_resets_by_engine(self):
        smoothed=causal_mean([1,3,99,10],[1,1,1,2],2)
        np.testing.assert_allclose(smoothed,[1,2,51,10])
        np.testing.assert_array_equal(sustained_flags([1,1,1,1,1,0,1],[1,1,1,2,2,2,2],3),
                                      [False,False,True,False,False,False,False])

    def test_no_alert_is_missing_and_trigger_is_not_backdated(self):
        frame=pd.DataFrame({'unit_number':[1]*4+[2]*4,'time_in_cycles':[30,31,32,33]*2,
                            'autoencoder_score':[2,2,2,0,0,0,0,0]})
        result=lead_times(frame,[40,39,38,37,50,49,48,47],'autoencoder',1)
        self.assertEqual(result.iloc[0].first_alert_cycle,32)
        self.assertEqual(result.iloc[0].lead_time_cycles,38)
        self.assertTrue(np.isnan(result.iloc[1].lead_time_cycles))

    def test_gaps_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(pd.DataFrame({'unit_number':[1,1],'time_in_cycles':[1,3],'s_2':[1,2]}),['s_2'])


if __name__=='__main__':unittest.main()
