"""Unit tests for FalkorDB per-group-id query routing in the operations layer.

These cover the *namespace/ops* read path (e.g. ``graphiti.nodes.episode.
get_by_group_ids``), which delegates to the Falkor ``*Operations`` classes with
the base driver (pointed at ``default_db``). FalkorDB stores each ``group_id`` in
its own physical graph, so a group-scoped read must clone the driver to that
group's graph or it queries an empty default database and returns nothing.

The tests use a ``spec=GraphDriver`` mock, so they need no live FalkorDB and run
in CI regardless of whether the ``falkordb`` package is installed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.entity_edge_ops import (
    FalkorEntityEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.entity_node_ops import (
    FalkorEntityNodeOperations,
)
from graphiti_core.driver.falkordb.operations.episode_node_ops import (
    FalkorEpisodeNodeOperations,
)

pytestmark = pytest.mark.asyncio


def _make_falkor_driver(execute_return=None):
    """A GraphDriver-shaped mock whose ``clone`` returns a routed child driver."""
    child = MagicMock(spec=GraphDriver)
    child.provider = GraphProvider.FALKORDB
    child.execute_query = AsyncMock(return_value=execute_return or ([], None, None))

    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=execute_return or ([], None, None))
    base.clone = MagicMock(return_value=child)
    return base, child


async def test_episode_get_by_group_ids_single_group_routes_to_its_graph():
    """The single-group case must clone the base driver to the group's graph."""
    ops = FalkorEpisodeNodeOperations()
    base, child = _make_falkor_driver()

    result = await ops.get_by_group_ids(base, ['group-a'])

    # Routed: cloned to the group's database, query ran against the clone,
    # never against the base (default_db) driver.
    base.clone.assert_called_once_with(database='group-a')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()
    assert result == []


async def test_entity_node_get_by_group_ids_single_group_routes_to_its_graph():
    ops = FalkorEntityNodeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, ['tenant-1'])

    base.clone.assert_called_once_with(database='tenant-1')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()


async def test_entity_edge_get_by_group_ids_single_group_routes_to_its_graph():
    ops = FalkorEntityEdgeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, ['tenant-1'])

    base.clone.assert_called_once_with(database='tenant-1')
    child.execute_query.assert_awaited_once()
    base.execute_query.assert_not_called()


async def test_episode_get_by_group_ids_multi_group_fans_out_per_graph():
    """The multi-group case must clone once per group_id and aggregate."""
    ops = FalkorEpisodeNodeOperations()
    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=([], None, None))

    clones = {}

    def _clone(database):
        child = MagicMock(spec=GraphDriver)
        child.provider = GraphProvider.FALKORDB
        child.execute_query = AsyncMock(return_value=([], None, None))
        # Mirror FalkorDriver.clone: re-cloning to the same database returns self,
        # so the per-group recursion's single-group clone is a no-op.
        child.clone = MagicMock(return_value=child)
        clones[database] = child
        return child

    base.clone = MagicMock(side_effect=_clone)

    await ops.get_by_group_ids(base, ['g1', 'g2', 'g3'])

    # One clone per group; the base driver itself never runs the query.
    assert sorted(clones) == ['g1', 'g2', 'g3']
    base.execute_query.assert_not_called()
    for child in clones.values():
        child.execute_query.assert_awaited_once()


async def test_get_by_group_ids_empty_group_ids_does_not_route():
    """No group_ids => no cloning; falls through to the normal (base) query."""
    ops = FalkorEpisodeNodeOperations()
    base, child = _make_falkor_driver()

    await ops.get_by_group_ids(base, [])

    base.clone.assert_not_called()
    base.execute_query.assert_awaited_once()


async def test_episode_get_by_saga_names_multi_group_fans_out_per_graph():
    """Saga lookup must query each requested group graph and aggregate the results."""

    ops = FalkorEpisodeNodeOperations()

    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=([], None, None))

    clones = {}

    def _clone(database):
        child = MagicMock(spec=GraphDriver)
        child.provider = GraphProvider.FALKORDB
        child.execute_query = AsyncMock(return_value=([], None, None))
        clones[database] = child
        return child

    base.clone = MagicMock(side_effect=_clone)

    result = await ops.get_by_saga_names(
        base,
        ['saga-1', 'saga-2'],
        ['group-a', 'group-b'],
    )

    assert result == []

    assert sorted(clones) == [
        'group-a',
        'group-b',
    ]

    base.execute_query.assert_not_called()

    for child in clones.values():
        child.execute_query.assert_awaited_once()

        query = child.execute_query.await_args.args[0]
        kwargs = child.execute_query.await_args.kwargs

        assert 'MATCH (s:Saga)-[:HAS_EPISODE]->(e:Episodic)' in query
        assert 'WHERE s.name IN $saga_names' in query
        assert 's.group_id IN $group_ids' not in query
        assert kwargs['saga_names'] == [
            'saga-1',
            'saga-2',
        ]


async def test_episode_get_by_saga_names_deduplicates_and_orders_across_groups(monkeypatch):
    """Episodes returned from multiple group graphs must be deduplicated and ordered."""

    ops = FalkorEpisodeNodeOperations()

    base = MagicMock(spec=GraphDriver)
    base.provider = GraphProvider.FALKORDB
    base.execute_query = AsyncMock(return_value=([], None, None))

    records_by_database = {
        'group-a': [
            {
                'uuid': 'episode-3',
                'valid_at': 3,
                'created_at': 3,
            },
            {
                'uuid': 'episode-1',
                'valid_at': 1,
                'created_at': 1,
            },
        ],
        'group-b': [
            {
                'uuid': 'episode-2',
                'valid_at': 2,
                'created_at': 2,
            },
            {
                'uuid': 'episode-1',
                'valid_at': 1,
                'created_at': 1,
            },
        ],
    }

    def _clone(database):
        child = MagicMock(spec=GraphDriver)
        child.provider = GraphProvider.FALKORDB
        child.execute_query = AsyncMock(
            return_value=(
                records_by_database[database],
                None,
                None,
            )
        )
        return child

    base.clone = MagicMock(side_effect=_clone)

    def _parse_episode(record):
        episode = MagicMock()
        episode.uuid = record['uuid']
        episode.valid_at = record['valid_at']
        episode.created_at = record['created_at']
        return episode

    monkeypatch.setattr(
        'graphiti_core.driver.falkordb.operations.episode_node_ops.episodic_node_from_record',
        _parse_episode,
    )

    results = await ops.get_by_saga_names(
        base,
        ['saga-1'],
        ['group-a', 'group-b'],
    )

    assert [episode.uuid for episode in results] == [
        'episode-1',
        'episode-2',
        'episode-3',
    ]

    assert len(results) == 3
    base.execute_query.assert_not_called()
