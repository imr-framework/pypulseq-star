from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pypulseq_star.plotting.plotter as pl

pytestmark = pytest.mark.unit


def test_plotter_identity_timing_and_block_helpers():
    assert pl._safe_plot_token('a-b c')=='a-b c'
    assert pl._safe_plot_token('a.b.c')=='a_b_c'
    assert pl._plot_node_within('a.b.c','a.b')
    assert not pl._plot_node_within('a.bc','a.b')
    assert pl._plot_node_depth('a.b.c')==3
    block=SimpleNamespace(name='b',role='readout',node='kernel.readout',metadata={'seqstar_node':'kernel.readout','seqstar_role':'readout'},events=[])
    assert pl._block_node_name(block,0)=='kernel.readout'
    assert pl._block_role(block)=='readout'
    assert list(pl._iter_events(block))==[]
    assert pl._looks_renderable_event(SimpleNamespace(type='rf'))
    assert not pl._looks_renderable_event(SimpleNamespace())
    assert pl._get_event_tstart(SimpleNamespace(delay=1e-3))==pytest.approx(1e-3)
    assert pl._get_event_duration(SimpleNamespace(duration=2e-3))==pytest.approx(2e-3)
    assert pl._get_event_duration(SimpleNamespace(shape=SimpleNamespace(duration=3e-3)))==pytest.approx(3e-3)
    assert pl._get_optional_float(SimpleNamespace(x='2.5'),'x')==pytest.approx(2.5)
    assert pl._get_optional_float(SimpleNamespace(),'x') is None


def test_plotter_adc_and_event_classification_helpers():
    rf=SimpleNamespace(type='rf',duration=1e-3)
    adc=SimpleNamespace(type='adc',delay=1e-4,num_samples=10,dwell=2e-6)
    grad=SimpleNamespace(type='trap',channel='x',rise_time=1e-4,flat_time=2e-4,fall_time=1e-4,amplitude=2)
    assert pl._is_rf_event(rf) and pl._is_adc_event(adc) and pl._is_gradient_event(grad)
    assert pl._get_adc_duration(adc)==pytest.approx(20e-6)
    windows=pl._adc_windows_for_plot(SimpleNamespace(type='adc',windows=[{'delay':0,'num_samples':4,'dwell':1e-6}]))
    assert windows[0]['num_samples']==4
    assert pl._extract_windows_from_candidate([{'delay':0,'num_samples':2,'dwell':1e-6}])
    assert pl._extract_windows_from_candidate(({'delay':0,'num_samples':2,'dwell':1e-6},)) == []
    assert pl._normalize_adc_window_for_plot(SimpleNamespace(delay=1e-3,num_samples=8,dwell=2e-6))['duration']==pytest.approx(16e-6)
    assert pl._normalize_axis('read')=='x'
    assert pl._normalize_axis('phase')=='y'
    assert pl._normalize_axis('slice')=='z'
    assert pl._normalize_axis('bad') is None
    assert pl._gradient_axis(grad, sequence=None)=='x'
    t,v=pl._gradient_waveform(grad,0.0)
    assert len(t)==4 and max(v)==2
    assert pl._gradient_active_duration(grad, t_relative=t)==pytest.approx(4e-4)


def test_plotter_array_render_and_range_helpers(capsys):
    assert pl._as_float_list(np.array([1,2]))==[1.0,2.0]
    assert pl._as_float_list(None)==[]
    payload={'tt':[0,1], 'waveform':[2,3]}
    assert pl._gradient_arrays_from_payload(payload)==([0.0,1.0],[2.0,3.0])
    assert pl._event_path(SimpleNamespace(name='rf'),1,2).endswith('rf')
    obj=SimpleNamespace(parameters={'n':'4','x':'2.5'},metadata={'timing':{'duration':3}})
    assert pl._get_int_from_parameters(obj,keys=('n',),default=0)==4
    assert pl._get_float_from_parameters(obj,keys=('x',),default=None)==pytest.approx(2.5)
    assert pl._get_timing_value(obj,'duration')==3
    assert pl._symmetric_or_positive_ylim([{'v':[-2,1]}], positive=False) == pytest.approx((-2.3,2.3))
    assert pl._symmetric_or_positive_ylim([{'v':[0,2]}], positive=True) == pytest.approx((-0.1,2.3))
    assert pl._first_event_unit([{'unit':'T/m'}])=='T/m'
    render={
        'rf':[{'t':[0.0,0.2]}],
        'adc':[{'span':(0.1,0.15)}],
        'gradients':{'x':[{'t':[0.0,0.3]}], 'y':[], 'z':[]},
    }
    assert pl._render_duration(render)==pytest.approx(0.3)
    assert pl._event_intersects_range({'span': (1.0, 1.0)}, 1.5, 3.0)
    assert pl._event_intersects_range({'t': [1.0, 2.0]}, 1.5, 3.0)
    assert not pl._event_intersects_range({'tstart': 1.0, 'duration': 1.0}, 1.5, 3.0)
    clipped={
        'rf':[{'t':[0.0,1.0]}],
        'adc':[],
        'gradients':{'x':[], 'y':[], 'z':[]},
    }
    result=pl._clip_render_to_time_range(clipped,(0.2,0.8))
    assert result is None
    assert clipped['rf']
    pl.debug_render_summary({'rf':[],'adc':[],'gradients':{'x':[],'y':[],'z':[]}})
    assert 'render' in capsys.readouterr().out.lower()
