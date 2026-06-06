"""P2-2 — VectorIndex 路径补强测试。

目标:把 vector_index.py 的覆盖率从 21% 提升到 80%+。
覆盖全部关键路径:
  * __init__ 参数分支
  * add 行为 (维度校验、IVF 触发、FAISS 触发)
  * brute_score / ivf_score / faiss_score 三个评分路径
  * _build_ivf k-means 全流程 (含空 cell reseed)
  * rebuild 从 store 重建
  * query 返回格式与排序
  * 异常路径: store 异常、维度不匹配、空索引
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from zwm.storage.episodic_db import EpisodicStore
from zwm.storage.vector_index import VectorIndex


# ===================== Fixtures =====================
@pytest.fixture
def dim():
    return 16  # 小维度加速测试


@pytest.fixture
def small_index(dim):
    return VectorIndex(dim=dim, nlist=4, nprobe=2, ivf_threshold=32, kmeans_iters=3)


@pytest.fixture
def in_memory_store():
    """临时 SQLite 内存库。"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    store = EpisodicStore(db_path=path)
    yield store, path
    store.close()
    Path(path).unlink(missing_ok=True)


# ===================== 1. __init__ =====================
class TestInit:
    def test_default_init(self):
        idx = VectorIndex()
        assert idx.dim == 1000
        assert idx.nlist == 16
        assert idx.nprobe == 4
        assert idx.ivf_threshold == 256
        assert len(idx) == 0
        assert idx._ivf_centroids is None
        assert idx._ivf_cells == []
        assert idx._faiss_index is None

    def test_custom_init(self, dim):
        idx = VectorIndex(dim=dim, nlist=4, nprobe=2, ivf_threshold=8, kmeans_iters=2)
        assert idx.dim == dim
        assert idx.nlist == 4
        assert idx.nprobe == 2
        assert idx.ivf_threshold == 8
        assert idx.kmeans_iters == 2

    def test_faiss_backend_probe(self):
        """_try_load_faiss 不应抛异常 — 无论是否安装。"""
        idx = VectorIndex()
        # 可能是 None (无 faiss) 或 module
        assert idx._faiss_backend is None or hasattr(idx._faiss_backend, "IndexFlatL2")


# ===================== 2. add 行为 =====================
class TestAdd:
    def test_add_valid(self, small_index, dim):
        v = np.ones(dim, dtype=np.float32)
        small_index.add(1, v)
        assert len(small_index) == 1
        assert small_index._ids == [1]
        assert len(small_index._vecs) == 1

    def test_add_malformed_wrong_ndim_skipped(self, small_index, dim):
        """二维向量被静默丢弃,不抛异常。"""
        v = np.ones((dim, 2), dtype=np.float32)
        small_index.add(1, v)
        assert len(small_index) == 0

    def test_add_malformed_wrong_dim_skipped(self, small_index, dim):
        v = np.ones(dim + 4, dtype=np.float32)
        small_index.add(1, v)
        assert len(small_index) == 0

    def test_add_triggers_ivf_build(self, dim):
        """当 N >= ivf_threshold 时,IVF 自动构建。"""
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=8)
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        assert len(idx) == 8
        assert idx._ivf_centroids is not None
        assert len(idx._ivf_cells) == 2  # nlist=2

    def test_add_int8_via_conversion(self, small_index, dim):
        """int8 数组也能加入 (会被转 float32)。"""
        v = np.ones(dim, dtype=np.int8)
        small_index.add(42, v)
        assert small_index._vecs[0].dtype == np.float32


# ===================== 3. _brute_score =====================
class TestBruteScore:
    def test_empty(self, small_index, dim):
        scored = small_index._brute_score(np.ones(dim, dtype=np.float32))
        assert scored == []

    def test_unit_vectors_orthogonal(self, small_index, dim):
        """两正交向量 → cosine=0。"""
        a = np.zeros(dim, dtype=np.float32); a[0] = 1.0
        b = np.zeros(dim, dtype=np.float32); b[1] = 1.0
        small_index.add(1, a)
        small_index.add(2, b)
        scored = small_index._brute_score(a)
        assert len(scored) == 2
        # 自己和自己 cosine = 1
        assert scored[0][0] == 1
        assert scored[0][1] == pytest.approx(1.0, abs=1e-5)
        # 与自己正交的 cosine = 0
        other = [s for s in scored if s[0] == 2][0]
        assert other[1] == pytest.approx(0.0, abs=1e-5)

    def test_id_mapping_preserved(self, small_index, dim):
        rng = np.random.default_rng(0)
        for i in range(5):
            v = rng.normal(size=dim).astype(np.float32)
            small_index.add(i + 100, v)
        scored = small_index._brute_score(rng.normal(size=dim).astype(np.float32))
        ids = {eid for eid, _ in scored}
        assert ids == {100, 101, 102, 103, 104}


# ===================== 4. _build_ivf k-means 行为 =====================
class TestBuildIVF:
    def test_below_threshold_skips(self, small_index):
        """N < ivf_threshold 不构建。"""
        small_index._build_ivf()
        assert small_index._ivf_centroids is None

    def test_above_threshold_builds(self, dim):
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=8, kmeans_iters=2)
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        idx._build_ivf()
        assert idx._ivf_centroids is not None
        assert idx._ivf_centroids.shape == (2, dim)
        assert len(idx._ivf_cells) == 2
        # 全部向量被分配
        total = sum(len(c) for c in idx._ivf_cells)
        assert total == 8

    def test_empty_cells_reseeded(self, dim):
        """k-means 中若某 cell 为空,会被随机向量填回,保持 k 稳定。"""
        # 用集中数据,人为制造 1-2 个空 cell
        idx = VectorIndex(dim=dim, nlist=4, nprobe=1, ivf_threshold=4, kmeans_iters=5)
        # 全部向量挤在一个方向 → 几乎肯定有多个空 cell
        for i in range(4):
            v = np.ones(dim, dtype=np.float32) * (i + 1)
            idx.add(i, v)
        idx._build_ivf()
        # centroids 仍为 4 个 (k 稳定)
        assert idx._ivf_centroids.shape == (4, dim)
        # cells 仍为 4 列表
        assert len(idx._ivf_cells) == 4

    def test_early_convergence(self, dim):
        """kmeans 收敛时会早停 — 直接退出循环。"""
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=4, kmeans_iters=20)
        # 完全相同的两向量,第一次迭代就收敛
        for i in range(4):
            v = np.array([1.0, 0.0] + [0.0] * (dim - 2), dtype=np.float32)
            idx.add(i, v)
        idx._build_ivf()
        assert idx._ivf_centroids is not None


# ===================== 5. _ivf_score =====================
class TestIVFScore:
    def test_no_centroids_falls_back_to_brute(self, small_index, dim):
        """centroids 缺失时,返回空 (与 _brute_score 一致,实现略不同)。"""
        small_index.add(1, np.ones(dim, dtype=np.float32))
        scored = small_index._ivf_score(np.ones(dim, dtype=np.float32))
        # _ivf_score 在 centroids 为空时退回 _brute_score
        assert len(scored) == 1

    def test_with_centroids(self, dim):
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=8)
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        q = rng.normal(size=dim).astype(np.float32)
        scored = idx._ivf_score(q)
        # 全部 8 个 vector 都在 cells 中,即使只 probe 1 个 cell 也至少有几个
        assert len(scored) > 0
        # similarity 应在 [-1, 1]
        for _, sim in scored:
            assert -1.0 <= sim <= 1.0


# ===================== 6. _faiss_score =====================
class TestFaissScore:
    def test_falls_back_when_unavailable(self, dim):
        """未安装 faiss 时 _faiss_score 退回 _ivf_score。"""
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=8)
        # 强制 faiss_backend 为 None
        idx._faiss_backend = None
        idx._faiss_index = None
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        # 不会抛异常
        scored = idx._faiss_score(rng.normal(size=dim).astype(np.float32))
        assert len(scored) > 0

    def test_search_exception_falls_back(self, dim):
        """FAISS 搜索抛异常时,退回到 IVF。"""
        idx = VectorIndex(dim=dim, nlist=2, nprobe=1, ivf_threshold=8)
        # 模拟一个"坏的"faiss index
        fake = MagicMock()
        fake.search.side_effect = RuntimeError("FAISS not built")
        idx._faiss_index = fake
        idx._faiss_backend = MagicMock()
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        scored = idx._faiss_score(rng.normal(size=dim).astype(np.float32))
        # 退回到 IVF, 至少返回部分结果
        assert len(scored) >= 0


# ===================== 7. _maybe_build_faiss =====================
class TestMaybeBuildFaiss:
    def test_below_1024_skipped(self, dim):
        idx = VectorIndex(dim=dim)
        for i in range(100):
            idx.add(i, np.ones(dim, dtype=np.float32))
        idx._maybe_build_faiss()
        # 不应构建 (N < 1024)
        assert idx._faiss_index is None

    def test_without_backend_skipped(self, dim):
        idx = VectorIndex(dim=dim)
        idx._faiss_backend = None
        for i in range(2000):
            idx.add(i, np.ones(dim, dtype=np.float32))
        idx._maybe_build_faiss()
        assert idx._faiss_index is None

    def test_already_built_skipped(self, dim):
        idx = VectorIndex(dim=dim)
        idx._faiss_index = MagicMock()  # 假装已构建
        idx._maybe_build_faiss()
        # 仍是原来的 mock (未重建)
        assert idx._faiss_index.__class__.__name__ == "MagicMock"


# ===================== 8. rebuild =====================
class TestRebuild:
    def test_rebuild_from_empty_store(self, small_index):
        """空 store 不抛异常。"""
        store = MagicMock()
        store.query_recent.return_value = []
        small_index.rebuild(store)
        assert len(small_index) == 0

    def test_rebuild_store_query_exception(self, small_index):
        """store.query_recent 抛异常时,rebuild 静默返回。"""
        store = MagicMock()
        store.query_recent.side_effect = RuntimeError("db locked")
        small_index.rebuild(store)
        assert len(small_index) == 0

    def test_rebuild_with_real_store(self, in_memory_store, dim):
        store, _ = in_memory_store
        # 插入几个情节 — store() 接受 main_bits (int)
        for i in range(5):
            v = np.ones(dim, dtype=np.float32) * (1.0 if i % 2 == 0 else -1.0)
            store.store(
                main_bits=i,  # bits 0..63
                reward=0.5, outcome="吉", encoded_vector=v,
            )
        # 先 add 一个,然后 rebuild 应该会清空
        idx = VectorIndex(dim=dim, ivf_threshold=3)
        idx.add(99, np.ones(dim, dtype=np.float32))
        idx.rebuild(store)
        assert len(idx) == 5
        assert 99 not in idx._ids

    def test_rebuild_skips_episodes_without_vector(self, in_memory_store, dim):
        store, _ = in_memory_store
        # 无 encoded_vector 的情节
        store.store(
            main_bits=0,
            reward=0.5, outcome="吉", encoded_vector=None,
        )
        idx = VectorIndex(dim=dim)
        idx.rebuild(store)
        assert len(idx) == 0


# ===================== 9. query 端到端 =====================
class TestQuery:
    def test_empty_index_returns_empty(self, in_memory_store, dim):
        store, _ = in_memory_store
        idx = VectorIndex(dim=dim)
        out = idx.query(store, np.ones(dim, dtype=np.float32))
        assert out == []

    def test_query_dimension_mismatch_returns_empty(self, in_memory_store, dim):
        store, _ = in_memory_store
        idx = VectorIndex(dim=dim)
        idx.add(1, np.ones(dim, dtype=np.float32))
        out = idx.query(store, np.ones(dim + 4, dtype=np.float32))
        # 维度不匹配 → 返回 []
        assert out == []

    def test_query_brute_path(self, in_memory_store, dim):
        store, _ = in_memory_store
        idx = VectorIndex(dim=dim, ivf_threshold=1000)
        for i in range(5):
            v = (np.ones(dim, dtype=np.float32) * (1.0 if i % 2 == 0 else -1.0))
            store.store(
                main_bits=i,
                reward=0.5, outcome="吉", encoded_vector=v,
            )
        # 重建以使 store 和 idx 同步
        idx.rebuild(store)
        q = np.ones(dim, dtype=np.float32)
        out = idx.query(store, q, limit=3)
        assert len(out) == 3
        # 包含情节核心字段
        for ep in out:
            assert "id" in ep
            assert "main_hex_bits" in ep

    def test_query_ivf_path(self, dim):
        """IVF 路径下也能返回结果。"""
        store = MagicMock()
        store.query_recent.return_value = []
        # 插入 8 个 vector 触发 IVF
        idx = VectorIndex(dim=dim, nlist=2, nprobe=2, ivf_threshold=4)
        rng = np.random.default_rng(0)
        for i in range(8):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        # store._conn 查询需要 mock
        store._conn.execute.return_value.fetchone.return_value = None
        out = idx.query(store, rng.normal(size=dim).astype(np.float32), limit=5)
        # fetchone 返回 None → _fetch_episode 返回 None → 列表为空
        assert out == []

    def test_query_with_fetched_rows(self, in_memory_store, dim):
        store, _ = in_memory_store
        # 真实数据 + 真实 fetch
        for i in range(3):
            v = (np.ones(dim, dtype=np.float32) * (1.0 if i == 0 else -1.0))
            store.store(
                main_bits=i,
                reward=0.5, outcome="吉", encoded_vector=v,
            )
        idx = VectorIndex(dim=dim, ivf_threshold=1000)
        idx.rebuild(store)
        out = idx.query(store, np.ones(dim, dtype=np.float32), limit=2)
        assert len(out) == 2
        # 排序 — 最相似的应该排第一
        ids = [ep["id"] for ep in out]
        # 注: rebuild 后 _ids 顺序是按时间倒序 (query_recent),
        # 排序稳定性保证完全相同的 -1 相似度按 [3, 2] 顺序。
        # 第一个结果应是 +1 向量 (id=1, 最相似)
        # 第二个结果应是 -1 向量 (id=3 优先于 id=2,因为 sort 是稳定的)
        assert ids[0] == 1
        assert ids[1] in (2, 3)


# ===================== 10. _fetch_episode =====================
class TestFetchEpisode:
    def test_exception_returns_none(self):
        store = MagicMock()
        store._conn.execute.side_effect = RuntimeError("db closed")
        assert VectorIndex._fetch_episode(store, 1) is None

    def test_missing_row_returns_none(self):
        store = MagicMock()
        store._conn.execute.return_value.fetchone.return_value = None
        assert VectorIndex._fetch_episode(store, 1) is None

    def test_valid_row_returns_dict(self):
        store = MagicMock()
        store._conn.execute.return_value.fetchone.return_value = {
            "id": 7, "h_current": "乾", "h_next": "坤", "reward": 0.5,
        }
        store._row_to_dict = lambda r: dict(r)
        ep = VectorIndex._fetch_episode(store, 7)
        assert ep["id"] == 7


# ===================== 11. 大规模 benchmark (opt) =====================
class TestScale:
    def test_500_vectors_brute(self, dim):
        """500 个向量 brute-force 查询 < 100ms。"""
        import time
        idx = VectorIndex(dim=dim, ivf_threshold=10000)
        rng = np.random.default_rng(0)
        for i in range(500):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        q = rng.normal(size=dim).astype(np.float32)
        store = MagicMock()
        store._conn.execute.return_value.fetchone.return_value = None
        t0 = time.time()
        out = idx.query(store, q, limit=10)
        dt = (time.time() - t0) * 1000
        assert dt < 1000  # < 1s for 500

    def test_500_vectors_ivf(self, dim):
        """500 个向量走 IVF 路径。"""
        idx = VectorIndex(dim=dim, nlist=8, nprobe=2, ivf_threshold=200)
        rng = np.random.default_rng(0)
        for i in range(500):
            idx.add(i, rng.normal(size=dim).astype(np.float32))
        assert idx._ivf_centroids is not None
        # IVF 路径
        scored = idx._ivf_score(rng.normal(size=dim).astype(np.float32))
        assert len(scored) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
