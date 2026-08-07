from __future__ import annotations

from types import SimpleNamespace

import pytest

from pypulseq_star.relationships.expression import SeqStarExpression
from pypulseq_star.relationships.lua_backend import relationship_to_lua_payload
from pypulseq_star.relationships.protocol import (
    MissingProtocolParameterError,
    is_protocol_reference,
    require_protocol_parameters,
)
from pypulseq_star.relationships.relationship import SeqStarRelationship
from pypulseq_star.relationships.sympy_backend import (
    SymbolicDependencyGraph,
    graph_from_gammastar_parameters,
    parameter_is_literal,
)
from pypulseq_star.writers.gammastar_literal_validator import (
    GammaStarLiteralValidationError,
    GammaStarLiteralValidator,
    LiteralIssue,
    LiteralValidationReport,
)

pytestmark=pytest.mark.unit


def test_literal_models_and_error_message():
    issue=LiteralIssue('root.x.duration',1.0,'derived','bad')
    report=LiteralValidationReport([issue],[])
    assert not report.ok and report.unresolved==[issue]
    assert report.to_dict()['unresolved_count']==1
    err=GammaStarLiteralValidationError(report)
    assert 'literal validation failed' in str(err)


def test_literal_validator_classification_and_repairs():
    v=GammaStarLiteralValidator(fail_on_unresolved=False)
    assert v._is_literal_parameter({'inputs':{},'script':'return 3'})
    assert not v._is_literal_parameter({'inputs':{'x':'root.prot.x'},'script':'return x'})
    assert v._literal_value({'inputs':{},'script':'return 3'})==3
    assert v._is_number(1) and not v._is_number(True)
    assert v._is_executable_path('root.kernel.rf.duration')
    assert not v._is_executable_path('root.prot.TR')
    assert v._classify('root.prot.TR',1) is None
    assert v._classify('root.kernel.rf.duration',1) is not None
    assert v._allowed_zero_tstart('root.tstart',0)
    assert v._message('derived_event_property') and v._expected_source('derived_event_property')
    params={'root.prot.TR':{'inputs':{},'script':'return 1'},'root.kernel.rf.duration':{'inputs':{},'script':'return 0.001'}}
    report=v.validate_and_fix(params,relationship_replacements={'root.kernel.rf.duration':{'inputs':{'d':'root.prot.TR'},'script':'return d'}})
    assert 'root.kernel.rf.duration' in report.fixed_paths


def test_protocol_lua_and_symbolic_backends():
    seq=SimpleNamespace(parameters={'TE':0.01})
    require_protocol_parameters(seq,['TE'])
    with pytest.raises(MissingProtocolParameterError): 
        require_protocol_parameters(seq,['TR'])
    with pytest.raises(MissingProtocolParameterError): 
        require_protocol_parameters(SimpleNamespace(),['TE'])
    assert is_protocol_reference('TE') and not is_protocol_reference(1)
    expr=SeqStarExpression(canonical='x+1',lua='return x+1',inputs={'x':'root.prot.x'})
    rel=SeqStarRelationship(name='r', relation_type='custom', description='test', expression=expr, resolved={'y':2})
    payload=relationship_to_lua_payload(rel)
    assert payload['script']=='return x+1'
    assert relationship_to_lua_payload(SeqStarRelationship(name='n', relation_type='custom', description='test'))=={}
    graph=SymbolicDependencyGraph()
    graph.add_expression('b',inputs={'a':'a'},script='return a')
    graph.add_expression('c',inputs={'b':'b'},script='return b')
    assert graph.dependencies_of('c',transitive=False)=={'b'}
    assert graph.depends_on('c','a')
    assert graph.to_dict()['expressions']['c']['target']=='c'
    assert parameter_is_literal(3)
    assert parameter_is_literal({'inputs':{},'script':'return 1'})
    assert not parameter_is_literal({'inputs':{'x':'a'},'script':'return x'})
    rebuilt=graph_from_gammastar_parameters({'x':{'inputs':{'a':'root.a'},'script':'return a'},'y':3})
    assert rebuilt.depends_on('x','root.a')
