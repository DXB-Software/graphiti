from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.nodes import EpisodicNode


@pytest.mark.asyncio
async def test_get_by_saga_names_filters_groups_and_returns_episodes():
    records = [
        {'episode': 'record-1'},
        {'episode': 'record-2'},
    ]

    executor = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=(records, None, None)
        )
    )

    episodes = [
        EpisodicNode.model_construct(uuid='episode-1'),
        EpisodicNode.model_construct(uuid='episode-2'),
    ]

    operations = Neo4jEpisodeNodeOperations()

    with patch(
        'graphiti_core.driver.neo4j.operations.episode_node_ops.episodic_node_from_record',
        side_effect=episodes,
    ) as parser:
        results = await operations.get_by_saga_names(
            executor,
            ['saga-1', 'saga-2'],
            ['group-1', 'group-2'],
        )

    assert [episode.uuid for episode in results] == [
        'episode-1',
        'episode-2',
    ]

    parser.assert_any_call(records[0])
    parser.assert_any_call(records[1])

    executor.execute_query.assert_awaited_once()

    query = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs

    assert 'MATCH (s:Saga)-[:HAS_EPISODE]->(e:Episodic)' in query
    assert 'WHERE s.name IN $saga_names' in query
    assert 'AND s.group_id IN $group_ids' in query
    assert 'RETURN DISTINCT' in query
    assert 'ORDER BY e.valid_at ASC, e.created_at ASC, e.uuid ASC' in query

    assert kwargs['saga_names'] == [
        'saga-1',
        'saga-2',
    ]
    assert kwargs['group_ids'] == [
        'group-1',
        'group-2',
    ]
    assert kwargs['routing_'] == 'r'


@pytest.mark.asyncio
async def test_get_by_saga_names_omits_group_filter_when_not_supplied():
    executor = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=([], None, None)
        )
    )

    operations = Neo4jEpisodeNodeOperations()

    results = await operations.get_by_saga_names(
        executor,
        ['saga-1'],
    )

    assert results == []

    executor.execute_query.assert_awaited_once()

    query = executor.execute_query.await_args.args[0]
    kwargs = executor.execute_query.await_args.kwargs

    assert 'MATCH (s:Saga)-[:HAS_EPISODE]->(e:Episodic)' in query
    assert 'WHERE s.name IN $saga_names' in query
    assert 's.group_id IN $group_ids' not in query

    assert kwargs['saga_names'] == ['saga-1']
    assert kwargs['group_ids'] is None
    assert kwargs['routing_'] == 'r'
