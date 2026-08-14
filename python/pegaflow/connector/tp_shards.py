"""Query and lease coordination for node-local TP shards."""

from dataclasses import dataclass

from pegaflow.connector.common import RecurrentLoadHold, logger
from pegaflow.pegaflow import EngineRpcClient, QueryLoading, QueryReady


@dataclass(frozen=True, slots=True)
class ShardedQueryReady:
    num_hit_blocks: int
    leases: tuple[bytes, ...]
    # HMA only: per recurrent group, per shard membership leases and their
    # hit positions (see RecurrentLoadHold for the wire/load contract).
    recurrent_hold: RecurrentLoadHold | None = None
    # HMA only: sorted query positions the hit may legally end at (every
    # recurrent group holds a checkpoint there on every shard, below the
    # attention prefix). Drives boundary re-derivation under token clamps.
    usable_positions: tuple[int, ...] = ()


class TpShardQueryClient:
    def __init__(self, clients: tuple[EngineRpcClient, ...]):
        self._clients = clients

    def query(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        wait_for_full_prefix: bool,
    ) -> ShardedQueryReady | None:
        results: list[QueryReady] = []
        try:
            for client in self._clients:
                result = client.query_prefetch(
                    instance_id,
                    block_hashes,
                    req_id=req_id,
                    wait_for_full_prefix=wait_for_full_prefix,
                )
                if isinstance(result, QueryLoading):
                    self.release(tuple(ready.lease for ready in results), req_id)
                    return None
                if not isinstance(result, QueryReady):
                    raise TypeError(f"query_prefetch returned unexpected outcome {type(result)!r}")
                results.append(result)
                self._validate_ready(result, len(block_hashes), len(results) - 1)
        except Exception:
            self.release(tuple(ready.lease for ready in results), req_id)
            raise

        common_blocks = min(result.num_hit_blocks for result in results)
        leases = [result.lease for result in results]
        if common_blocks == 0:
            self.release(tuple(leases), req_id)
            return ShardedQueryReady(0, tuple(b"" for _ in results))

        exact_hashes = block_hashes[:common_blocks]
        try:
            for index, (client, result) in enumerate(zip(self._clients, results, strict=True)):
                if result.num_hit_blocks == common_blocks:
                    continue
                exact = client.query_prefetch(
                    instance_id,
                    exact_hashes,
                    req_id=f"{req_id}:tp-common-{common_blocks}",
                    wait_for_full_prefix=False,
                )
                if not isinstance(exact, QueryReady):
                    raise RuntimeError(
                        f"TP shard {index} could not lease the common {common_blocks}-block prefix"
                    )
                try:
                    self._validate_ready(exact, common_blocks, index)
                    if exact.num_hit_blocks != common_blocks:
                        raise RuntimeError(
                            f"TP shard {index} could not lease the common "
                            f"{common_blocks}-block prefix"
                        )
                except Exception:
                    self._release_one(client, exact.lease, req_id)
                    raise
                old_lease = leases[index]
                leases[index] = exact.lease
                self._release_one(client, old_lease, req_id)
        except Exception:
            self.release(tuple(leases), req_id)
            raise

        return ShardedQueryReady(common_blocks, tuple(leases))

    def query_group_membership(
        self,
        instance_id: str,
        block_hashes: list[bytes],
        req_id: str,
        group_id: int,
    ) -> list[tuple[tuple[int, ...], bytes]]:
        """Per-shard membership query over one hybrid storage group.

        Returns ``(hit_positions, lease)`` per shard; the lease pins exactly
        the hit blocks in positions order. Membership queries are local-only,
        so every shard answers Ready immediately (never Loading).
        """
        results: list[tuple[tuple[int, ...], bytes]] = []
        try:
            for shard_index, client in enumerate(self._clients):
                result = client.query_prefetch(
                    instance_id,
                    block_hashes,
                    req_id=req_id,
                    group_id=group_id,
                )
                if isinstance(result, QueryLoading):
                    raise RuntimeError(
                        f"TP shard {shard_index} membership query for group {group_id} "
                        "returned Loading; membership queries are local-only"
                    )
                if not isinstance(result, QueryReady):
                    raise TypeError(f"query_prefetch returned unexpected outcome {type(result)!r}")
                positions = tuple(result.hit_positions)
                if len(positions) != result.num_hit_blocks:
                    raise RuntimeError(
                        f"TP shard {shard_index} reported {result.num_hit_blocks} hits "
                        f"but returned {len(positions)} positions"
                    )
                if any(position >= len(block_hashes) for position in positions):
                    raise RuntimeError(
                        f"TP shard {shard_index} returned hit positions outside "
                        f"a {len(block_hashes)}-hash query"
                    )
                if positions and not result.lease:
                    raise RuntimeError(
                        f"TP shard {shard_index} returned {len(positions)} hits without a lease"
                    )
                results.append((positions, result.lease))
        except Exception:
            self.release(tuple(lease for _, lease in results), req_id)
            raise
        return results

    def release(self, leases: tuple[bytes, ...], req_id: str) -> bool:
        released = True
        for client, lease in zip(self._clients, leases, strict=False):
            released = self._release_one(client, lease, req_id) and released
        return released

    @staticmethod
    def _validate_ready(result: QueryReady, queried_blocks: int, shard_index: int) -> None:
        if result.num_hit_blocks > queried_blocks:
            raise RuntimeError(
                f"TP shard {shard_index} reported {result.num_hit_blocks} hits for "
                f"a {queried_blocks}-block query"
            )
        if result.num_hit_blocks and not result.lease:
            raise RuntimeError(
                f"TP shard {shard_index} returned {result.num_hit_blocks} hits without a lease"
            )

    @staticmethod
    def _release_one(client: EngineRpcClient, lease: bytes, req_id: str) -> bool:
        if not lease:
            return True
        try:
            client.release(lease)
        except Exception:
            logger.exception(
                "[PegaKVConnector] query lease release exception: req=%s",
                req_id,
            )
            return False
        return True


__all__ = ["ShardedQueryReady", "TpShardQueryClient"]
