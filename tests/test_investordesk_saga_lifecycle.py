from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graphiti_core.driver.neo4j.operations.entity_edge_ops import (
    Neo4jEntityEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.saga_node_ops import (
    Neo4jSagaNodeOperations,
    _saga_node_from_record,
)
from graphiti_core.edges import Edge, EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.models.edges.edge_db_queries import (
    HAS_EPISODE_EDGE_SAVE,
    NEXT_EPISODE_EDGE_SAVE,
)
from graphiti_core.nodes import EpisodeType, EpisodicNode, Node, SagaNode


@pytest.mark.asyncio
async def test_neo4j_saga_save_preserves_lifecycle_fields():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    saga = SagaNode(
        uuid='saga-1',
        name='artifact-1',
        group_id='group-1',
        created_at=timestamp,
        summary='summary',
        first_episode_uuid='episode-1',
        last_episode_uuid='episode-3',
        last_summarized_at=timestamp,
        last_summarized_episode_valid_at=timestamp,
    )

    executor = SimpleNamespace(
        execute_query=AsyncMock(return_value=([], None, None))
    )

    operations = Neo4jSagaNodeOperations()

    await operations.save(executor, saga)

    kwargs = executor.execute_query.await_args.kwargs

    assert kwargs['uuid'] == 'saga-1'
    assert kwargs['summary'] == 'summary'
    assert kwargs['first_episode_uuid'] == 'episode-1'
    assert kwargs['last_episode_uuid'] == 'episode-3'
    assert kwargs['last_summarized_at'] == timestamp
    assert kwargs['last_summarized_episode_valid_at'] == timestamp


def test_neo4j_saga_hydration_preserves_lifecycle_fields():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    saga = _saga_node_from_record(
        {
            'uuid': 'saga-1',
            'name': 'artifact-1',
            'group_id': 'group-1',
            'created_at': timestamp,
            'summary': 'summary',
            'first_episode_uuid': 'episode-1',
            'last_episode_uuid': 'episode-3',
            'last_summarized_at': timestamp,
            'last_summarized_episode_valid_at': timestamp,
        }
    )

    assert saga.summary == 'summary'
    assert saga.first_episode_uuid == 'episode-1'
    assert saga.last_episode_uuid == 'episode-3'
    assert saga.last_summarized_at == timestamp
    assert saga.last_summarized_episode_valid_at == timestamp


def test_saga_structure_queries_are_idempotent():
    assert 'MERGE (saga)-[e:HAS_EPISODE]->(episode)' in HAS_EPISODE_EDGE_SAVE
    assert 'MERGE (saga)-[e:HAS_EPISODE {uuid: $uuid}]' not in HAS_EPISODE_EDGE_SAVE
    assert 'ON CREATE SET' in HAS_EPISODE_EDGE_SAVE

    assert (
        'MERGE (source_episode)-[e:NEXT_EPISODE]->(target_episode)'
        in NEXT_EPISODE_EDGE_SAVE
    )
    assert (
        'MERGE (source_episode)-[e:NEXT_EPISODE {uuid: $uuid}]'
        not in NEXT_EPISODE_EDGE_SAVE
    )
    assert 'ON CREATE SET' in NEXT_EPISODE_EDGE_SAVE


@pytest.mark.asyncio
async def test_reprocessing_attached_episode_does_not_mutate_saga_structure():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = SimpleNamespace(
        execute_query=AsyncMock(
            return_value=([{'already_attached': True}], None, None)
        )
    )
    graphiti.embedder = object()
    graphiti.store_raw_episode_content = True

    graphiti._get_or_create_saga = AsyncMock(
        return_value=SagaNode(
            uuid='saga-1',
            name='artifact-1',
            group_id='group-1',
            created_at=timestamp,
            first_episode_uuid='episode-1',
            last_episode_uuid='episode-3',
        )
    )
    graphiti._saga_get_previous_episode_uuid = AsyncMock()

    episode = EpisodicNode(
        uuid='episode-2',
        name='chunk-2',
        group_id='group-1',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        created_at=timestamp,
        valid_at=timestamp,
    )

    with (
        patch(
            'graphiti_core.graphiti.build_episodic_edges',
            return_value=[],
        ),
        patch(
            'graphiti_core.graphiti.add_nodes_and_edges_bulk',
            new=AsyncMock(),
        ),
        patch.object(
            SagaNode,
            'save',
            new=AsyncMock(),
        ) as saga_save,
    ):
        await graphiti._process_episode_data(
            episode=episode,
            nodes=[],
            entity_edges=[],
            now=timestamp,
            group_id='group-1',
            saga='artifact-1',
        )

    graphiti._saga_get_previous_episode_uuid.assert_not_awaited()
    saga_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_remove_episode_preserves_shared_fact_provenance():
    removed_created_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    removed_valid_at = datetime(2026, 7, 1, tzinfo=timezone.utc)

    replacement_created_at = datetime(2026, 8, 15, tzinfo=timezone.utc)
    replacement_valid_at = datetime(2026, 7, 15, tzinfo=timezone.utc)

    removed_episode = EpisodicNode.model_construct(
        uuid='episode-a',
        entity_edges=['edge-1'],
        created_at=removed_created_at,
        valid_at=removed_valid_at,
    )

    replacement_episode = EpisodicNode.model_construct(
        uuid='episode-b',
        entity_edges=['edge-1'],
        created_at=replacement_created_at,
        valid_at=replacement_valid_at,
    )

    edge = EntityEdge.model_construct(
        uuid='edge-1',
        episodes=['episode-a', 'episode-b'],
        created_at=removed_created_at,
        reference_time=removed_valid_at,
    )

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = object()

    with (
        patch.object(
            EpisodicNode,
            'get_by_uuid',
            new=AsyncMock(
                side_effect=[
                    removed_episode,
                    replacement_episode,
                ]
            ),
        ),
        patch.object(
            EntityEdge,
            'get_by_uuids',
            new=AsyncMock(return_value=[edge]),
        ),
        patch.object(
            EntityEdge,
            'load_fact_embedding',
            new=AsyncMock(),
        ) as load_embedding,
        patch.object(
            EntityEdge,
            'save',
            new=AsyncMock(),
        ) as save_edge,
        patch(
            'graphiti_core.graphiti.get_mentioned_nodes',
            new=AsyncMock(return_value=[]),
        ),
        patch.object(
            Edge,
            'delete_by_uuids',
            new=AsyncMock(),
        ) as delete_edges,
        patch.object(
            Node,
            'delete_by_uuids',
            new=AsyncMock(),
        ),
        patch.object(
            EpisodicNode,
            'delete',
            new=AsyncMock(),
        ),
    ):
        await graphiti.remove_episode('episode-a')

    assert edge.episodes == ['episode-b']
    assert edge.created_at == replacement_created_at
    assert edge.reference_time == replacement_valid_at

    load_embedding.assert_awaited_once()
    save_edge.assert_awaited_once()
    delete_edges.assert_awaited_once_with(graphiti.driver, [])


@pytest.mark.asyncio
async def test_remove_episode_deletes_fact_without_remaining_provenance():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    episode = EpisodicNode.model_construct(
        uuid='episode-a',
        entity_edges=['edge-1'],
        created_at=timestamp,
        valid_at=timestamp,
    )

    edge = EntityEdge.model_construct(
        uuid='edge-1',
        episodes=['episode-a'],
    )

    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = object()

    with (
        patch.object(
            EpisodicNode,
            'get_by_uuid',
            new=AsyncMock(return_value=episode),
        ),
        patch.object(
            EntityEdge,
            'get_by_uuids',
            new=AsyncMock(return_value=[edge]),
        ),
        patch.object(
            EntityEdge,
            'save',
            new=AsyncMock(),
        ) as save_edge,
        patch(
            'graphiti_core.graphiti.get_mentioned_nodes',
            new=AsyncMock(return_value=[]),
        ),
        patch.object(
            Edge,
            'delete_by_uuids',
            new=AsyncMock(),
        ) as delete_edges,
        patch.object(
            Node,
            'delete_by_uuids',
            new=AsyncMock(),
        ),
        patch.object(
            EpisodicNode,
            'delete',
            new=AsyncMock(),
        ),
    ):
        await graphiti.remove_episode('episode-a')

    save_edge.assert_not_awaited()
    delete_edges.assert_awaited_once_with(
        graphiti.driver,
        ['edge-1'],
    )


@pytest.mark.asyncio
async def test_neo4j_entity_edge_save_preserves_reference_time():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    edge = EntityEdge.model_construct(
        uuid='edge-1',
        source_node_uuid='node-1',
        target_node_uuid='node-2',
        name='OWNS',
        fact='A owns B',
        fact_embedding=[0.1, 0.2],
        group_id='group-1',
        episodes=['episode-1'],
        created_at=timestamp,
        expired_at=None,
        valid_at=timestamp,
        invalid_at=None,
        reference_time=timestamp,
        attributes={},
    )

    executor = SimpleNamespace(
        execute_query=AsyncMock(return_value=([], None, None))
    )

    operations = Neo4jEntityEdgeOperations()

    await operations.save(executor, edge)

    edge_data = executor.execute_query.await_args.kwargs['edge_data']

    assert edge_data['reference_time'] == timestamp


@pytest.mark.asyncio
async def test_neo4j_entity_edge_bulk_save_preserves_reference_time():
    timestamp = datetime(2026, 8, 31, tzinfo=timezone.utc)

    edge = EntityEdge.model_construct(
        uuid='edge-1',
        source_node_uuid='node-1',
        target_node_uuid='node-2',
        name='OWNS',
        fact='A owns B',
        fact_embedding=[0.1, 0.2],
        group_id='group-1',
        episodes=['episode-1'],
        created_at=timestamp,
        expired_at=None,
        valid_at=timestamp,
        invalid_at=None,
        reference_time=timestamp,
        attributes={},
    )

    executor = SimpleNamespace(
        execute_query=AsyncMock(return_value=([], None, None))
    )

    operations = Neo4jEntityEdgeOperations()

    await operations.save_bulk(executor, [edge])

    prepared = executor.execute_query.await_args.kwargs['entity_edges']

    assert prepared[0]['reference_time'] == timestamp
