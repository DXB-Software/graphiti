from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.namespaces.nodes import EpisodeNodeNamespace


@pytest.mark.asyncio
async def test_get_nodes_and_edges_by_saga_deduplicates_semantic_objects():
    graphiti = Graphiti.__new__(Graphiti)

    episodes = [
        EpisodicNode.model_construct(uuid='episode-1'),
        EpisodicNode.model_construct(uuid='episode-2'),
    ]

    nodes = [
        EntityNode.model_construct(uuid='node-1'),
        EntityNode.model_construct(uuid='node-1'),
        EntityNode.model_construct(uuid='node-2'),
    ]

    edges = [
        EntityEdge.model_construct(uuid='edge-1'),
        EntityEdge.model_construct(uuid='edge-1'),
        EntityEdge.model_construct(uuid='edge-2'),
    ]

    get_by_saga_names = AsyncMock(return_value=episodes)

    graphiti.nodes = SimpleNamespace(
        episode=SimpleNamespace(
            get_by_saga_names=get_by_saga_names,
        )
    )

    graphiti.get_nodes_and_edges_by_episode = AsyncMock(
        return_value=SearchResults(
            nodes=nodes,
            edges=edges,
        )
    )

    results = await graphiti.get_nodes_and_edges_by_saga(
        ['saga-1', 'saga-2'],
        ['group-1'],
    )

    assert [episode.uuid for episode in results.episodes] == [
        'episode-1',
        'episode-2',
    ]

    assert [node.uuid for node in results.nodes] == [
        'node-1',
        'node-2',
    ]

    assert [edge.uuid for edge in results.edges] == [
        'edge-1',
        'edge-2',
    ]

    get_by_saga_names.assert_awaited_once_with(
        ['saga-1', 'saga-2'],
        ['group-1'],
    )

    graphiti.get_nodes_and_edges_by_episode.assert_awaited_once_with(
        ['episode-1', 'episode-2']
    )


@pytest.mark.asyncio
async def test_get_nodes_and_edges_by_saga_returns_empty_for_no_saga_names():
    graphiti = Graphiti.__new__(Graphiti)

    get_by_saga_names = AsyncMock()

    graphiti.nodes = SimpleNamespace(
        episode=SimpleNamespace(
            get_by_saga_names=get_by_saga_names,
        )
    )

    graphiti.get_nodes_and_edges_by_episode = AsyncMock()

    results = await graphiti.get_nodes_and_edges_by_saga([])

    assert results == SearchResults()

    get_by_saga_names.assert_not_awaited()
    graphiti.get_nodes_and_edges_by_episode.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_nodes_and_edges_by_saga_returns_empty_when_no_episodes_found():
    graphiti = Graphiti.__new__(Graphiti)

    get_by_saga_names = AsyncMock(return_value=[])

    graphiti.nodes = SimpleNamespace(
        episode=SimpleNamespace(
            get_by_saga_names=get_by_saga_names,
        )
    )

    graphiti.get_nodes_and_edges_by_episode = AsyncMock()

    results = await graphiti.get_nodes_and_edges_by_saga(
        ['missing-saga'],
        ['group-1'],
    )

    assert results == SearchResults()

    get_by_saga_names.assert_awaited_once_with(
        ['missing-saga'],
        ['group-1'],
    )

    graphiti.get_nodes_and_edges_by_episode.assert_not_awaited()


@pytest.mark.asyncio
async def test_episode_namespace_get_by_saga_names_forwards_to_operations():
    driver = object()

    episodes = [
        EpisodicNode.model_construct(uuid='episode-1'),
        EpisodicNode.model_construct(uuid='episode-2'),
    ]

    operations = SimpleNamespace(
        get_by_saga_names=AsyncMock(return_value=episodes)
    )

    namespace = EpisodeNodeNamespace(
        driver,
        operations,
    )

    results = await namespace.get_by_saga_names(
        ['saga-1', 'saga-2'],
        ['group-1'],
    )

    assert results == episodes

    operations.get_by_saga_names.assert_awaited_once_with(
        driver,
        ['saga-1', 'saga-2'],
        ['group-1'],
    )
