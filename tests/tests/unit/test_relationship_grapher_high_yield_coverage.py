from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pypulseq_star.plotting.relationship_grapher as rg

pytestmark = pytest.mark.unit


def _sequence():
    rf = SimpleNamespace(name='rf_excitation_b001_e001', role='excitation', kind='rf')
    gx = SimpleNamespace(name='gx_readout_b001_e002', role='readout', kind='gradient')
    adc = SimpleNamespace(name='adc_readout_b001_e003', role='acquisition', kind='adc')
    block1 = SimpleNamespace(name='excitation', role='excitation', events=[rf], metadata={'seqstar_node':'kernel.excitation','seqstar_role':'excitation'})
    block2 = SimpleNamespace(name='readout', role='readout', events=[gx, adc], metadata={'seqstar_node':'kernel.readout','seqstar_role':'readout'})
    rel1 = {'name':'adc_center_after_rf','kind':'set_center_after','reference':rf,'target':adc,'offset':'TE','metadata':{}}
    rel2 = {'name':'kernel_repeat','kind':'repeat_every','period':'TR','events':[rf,gx,adc],'metadata':{}}
    system = SimpleNamespace(max_grad=28, max_slew=120, rf_dead_time=1e-4, adc_dead_time=2e-5)
    return SimpleNamespace(name='GRE', parameters={'sequence_type':'GRE','TE':0.005,'TR':0.02,'flip_angle':0.2,'repetitions':3,'extra':7}, system=system, timeline=SimpleNamespace(blocks=[block1,block2], nodes={'kernel':{'repeat_count':'repetitions'}}), relationships=[rel1,rel2], metadata={})


def test_graph_build_export_and_text(tmp_path: Path, monkeypatch):
    seq = _sequence()
    grapher = rg.RelationshipGrapher(seq, include_derived=True, show_all_protocol_parameters=True)
    graph = grapher.build()
    assert {'root','prot','sys','seqstar_loop','kernel'} <= set(graph.nodes)
    assert any(n.kind == 'relationship' for n in graph.nodes.values())
    assert any(e.kind == 'protocol_dependency' for e in graph.edges)
    mermaid = grapher.to_mermaid(graph)
    dot = grapher.to_dot(graph)
    assert 'flowchart TD' in mermaid and 'digraph' in dot
    assert grapher.write_mermaid(tmp_path/'g.mmd').exists()
    assert grapher.write_dot(tmp_path/'g.dot').exists()
    monkeypatch.setattr(rg.shutil, 'which', lambda _: None)
    assert grapher.write(tmp_path/'g.svg').suffix == '.dot'
    assert rg.write_relationship_graph_mermaid(seq, tmp_path/'h.mmd').exists()
    assert rg.write_relationship_graph_dot(seq, tmp_path/'h.dot').exists()
    text = rg.relationship_graph_text(seq)
    assert 'Hierarchy' in text and 'Relationships' in text and 'adc_center_after_rf' in text


def test_graph_container_and_normalization_helpers():
    graph = rg.RelationshipGraph()
    a = graph.add_node('1 bad id','A',kind='event')
    graph.add_node('1 bad id','A2',kind='rf')
    b = graph.add_node('b','B',kind='adc')
    graph.add_edge(a,b,kind='contains',label='x')
    graph.add_edge(a,b,kind='contains',label='x')
    assert len(graph.edges)==1 and graph.nodes[a].kind=='rf'
    assert rg._safe_id(' 1 a-b ') == 'n_1_a_b'
    assert rg._safe_id('***') == 'node'
    assert rg._spread(0,1,0)==[] and rg._spread(0,1,1)==[0.5]
    assert rg._spread(0,1,3)==[0.0,0.5,1.0]
    assert rg._compact_value(None)=='None' and rg._compact_value(0.0)=='0'
    assert 'e-' in rg._compact_value(1e-5)
    assert rg._wrap_text('a b c d',width=3)
    assert rg._mermaid_label('a\n"b"') == "a<br/>'b'"
    assert '\\"' in rg._escape_dot('a"b')
    assert rg._mermaid_node_shape(rg.GraphNode('x','x','root'))=='stadium'
    assert rg._mermaid_node_shape(rg.GraphNode('x','x','relationship'))=='subroutine'
    assert rg._mermaid_node_shape(rg.GraphNode('x','x','protocol'))=='database'
    assert rg._mermaid_node_shape(rg.GraphNode('x','x','derived_parameter'))=='circle'
    assert rg._node_style('rf')['facecolor'] and rg._node_style('unknown')['facecolor']
    for kind in ('contains','protocol_dependency','relationship_input','relationship_output','computes','other'):
        assert 'color' in rg._edge_style(kind)
    assert rg._icon_for_kind('rf') and rg._icon_for_kind('other')==''


def test_relationship_and_protocol_helpers():
    seq = _sequence()
    rels = rg._collect_relationships(seq)
    assert len(rels)==2
    assert rg._relationship_kind(rels[0])=='set_center_after'
    assert rg._relationship_name(rels[0],fallback='x')=='adc_center_after_rf'
    assert rg._relationship_field(rels[0],'offset')=='TE'
    assert rg._relationship_metadata(rels[0])=={}
    assert rg._get_parameters(seq)['TR']==0.02
    assert rg._important_protocol_keys(seq.parameters)
    assert rg._protocol_summary_lines(seq,max_items=2)
    assert rg._system_summary_lines(seq,max_items=3)
    assert rg._display_parameter_name('echo_time')=='Echo time (TE)'
    assert rg._resolve_protocol_alias('TE',{'echo_time':1})=='echo_time'
    assert rg._resolve_protocol_alias('missing',{}) is None
    assert rg._first_present({'TR':2},'tr','TR')==2
    assert rg._repeat_count_from_sequence(seq,seq.parameters)==3
    assert rg._compact_visual_event_signature(SimpleNamespace(role='readout',kind='adc'),'adc_b001_e002')
    assert rg._compact_visual_event_label('adc_b001_e002')=='adc'
    assert rg._object_name('x',fallback='y')=='x'
    assert rg._resolve_object_node_id(SimpleNamespace(name='RF 1'))=='RF_1'
    assert rg._event_kind(SimpleNamespace(role='excitation',kind='rf'))=='rf'
    assert 'RF pulse' in rg._event_label('rf','rf')
    assert rg._sequence_title(seq)=='GRE: GRE'
    assert rg._find_first_node_id_by_kind(rg.RelationshipGrapher(seq).build(),'block') is not None
    assert rg._compact_relationship_label('a_after_b','set_center_after')
    assert rg._tooltip_for_node(rg.GraphNode('x','X','rf',{'value':1}))
    assert rg._format_attrs({'a':'b'})=='a="b"'
