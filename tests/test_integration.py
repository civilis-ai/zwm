import math

from zwm.core.hexagram import all_hexagrams, hexagram_from_bits, hexagram_from_name
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.five_hexagrams import FiveHexagramChain
from zwm.scene_field.wuxing import element_force, hexagram_element_profile
from zwm.scene_field.liuqin import determine_six_relations, social_field_vector
from zwm.scene_field.calendar import GanzhiTime, MultiScaleCalendar
from zwm.scene_field.unified_field import UnifiedField
from zwm.planner.mutations import (
    all_mutations,
    all_successors,
    apply_mutation,
    classify_mutation,
    mutation_path,
    single_yao_mutations,
)
from zwm.planner.codon import codon_amino_acid, hexagram_to_codon
from zwm.planner.loop import TrinityPlanner
from zwm.moe.sparse_activation import SparseMoE
from zwm.langevin.sampler import LangevinSampler
from zwm.langevin.score import score_surface, total_score_gradient
from zwm.learning.online import CuriosityScheduler, GrowthManager, OnlineLearner
from zwm.learning.hebbian import HebbianAssociator
from zwm.storage.episodic_db import EpisodicStore
from zwm.encoder.base import RuleBasedEncoder


class TestFiveHexagramChain:
    def test_from_current(self):
        qian = hexagram_from_name("乾为天")
        chain = FiveHexagramChain.from_current(qian)
        assert chain.main.name == "乾为天"
        assert chain.inter.name == "乾为天"
        assert chain.complement.name == "坤为地"

    def test_with_evolution(self):
        qian = hexagram_from_name("乾为天")
        chain = FiveHexagramChain.with_evolution(qian, 0b000001)
        assert chain.evolved.name == "天风姤"
        assert chain.main.name == "乾为天"

    def test_narrative_coherence(self):
        qian = hexagram_from_name("乾为天")
        chain = FiveHexagramChain.from_current(qian)
        coherence = chain.narrative_coherence()
        assert 0.0 <= coherence <= 1.0


class TestWuxing:
    def test_element_profile(self):
        qian = hexagram_from_name("乾为天")
        profile = hexagram_element_profile(qian)
        assert profile["金"] > 0.4
        assert profile["火"] < 0.1

    def test_element_force_same(self):
        qian = hexagram_from_name("乾为天")
        force = element_force(qian, qian)
        assert force > 0

    def test_element_force_complement(self):
        qian = hexagram_from_name("乾为天")
        kun = hexagram_from_name("坤为地")
        force = element_force(qian, kun)
        assert -1.0 <= force <= 1.0


class TestLiuqin:
    def test_six_relations(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        relations = determine_six_relations(qian, grid, self_element="金")
        assert relations[grid.self_position] == "我"
        assert all(
            role in ("我", "父母", "兄弟", "妻财", "官鬼", "子孙")
            for role in relations.values()
        )

    def test_social_field(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        field = social_field_vector(qian, grid)
        assert field["我"] == [5]


class TestCalendar:
    def test_ganzhi_time(self):
        gt = GanzhiTime(0, 0, 0, 0)
        assert gt.year_ganzhi == "甲子"
        assert gt.hour_ganzhi == "甲子"

    def test_multi_scale_calendar(self):
        cal = MultiScaleCalendar()
        layers = cal.time_layers(2026)
        assert len(layers) == 3
        assert all(0 <= v <= 2 * math.pi for v in layers.values())


class TestMutations:
    def test_all_64_masks(self):
        masks = all_mutations()
        assert len(masks) == 64
        assert 0 in masks
        assert 63 in masks

    def test_single_yao_mutations(self):
        singles = single_yao_mutations()
        assert len(singles) == 6
        for s in singles:
            assert s.bit_count() == 1

    def test_classify(self):
        assert classify_mutation(0) == "不变"
        assert classify_mutation(0x01) == "初爻变"
        assert "2爻变" in classify_mutation(0x03)

    def test_mutation_path(self):
        qian = hexagram_from_name("乾为天")
        path = mutation_path(qian, [0x01, 0x02])
        assert len(path) == 3
        assert path[0] == qian

    def test_all_successors(self):
        qian = hexagram_from_name("乾为天")
        successors = all_successors(qian)
        assert len(successors) == 64
        assert all(isinstance(h, type(qian)) for h in successors.values())


class TestCodon:
    def test_hexagram_to_codon(self):
        codon = hexagram_to_codon(0)
        assert codon == "UUU"

    def test_codon_amino_acid(self):
        assert codon_amino_acid("AUG") == "Met"
        assert codon_amino_acid("UAA") == "STOP"


class TestUnifiedField:
    def test_snapshot(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        uf = UnifiedField.snapshot(qian, grid)
        assert uf.hexagram == qian
        assert uf.grid == grid
        assert len(uf.element_profile) == 5

    def test_evolve(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        uf = UnifiedField.snapshot(qian, grid)
        uf2 = uf.evolve(mutation_mask=0b000001)
        assert uf2.hexagram.name == "天风姤"

    def test_to_tensor(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        uf = UnifiedField.snapshot(qian, grid)
        tensor = uf.to_tensor()
        assert len(tensor) > 6


class TestPlanner:
    def test_plan(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        planner = TrinityPlanner()
        result = planner.plan(qian, grid)
        assert result.top_score >= 0
        assert result.top_mutation > 0
        assert len(result.chain.to_dict()) == 5

    def test_hexagram_scores_populated(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        planner = TrinityPlanner()
        result = planner.plan(qian, grid, top_k=5)
        assert len(result.hexagram_scores) > 0


class TestMoE:
    def test_sparse_moe(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        moe = SparseMoE(top_k=3)
        score = moe.evaluate(qian, grid, time_phase=0.0, target_palace=1)
        assert 0.0 <= score <= 1.0

    def test_active_experts(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        moe = SparseMoE(top_k=3)
        active = moe.active_experts(qian, grid, time_phase=0.0)
        assert 1 <= len(active) <= 6


class TestLangevin:
    def test_score_surface(self):
        qian = hexagram_from_name("乾为天")
        score = score_surface(qian)
        assert 0.0 <= score <= 1.0

    def test_gradient(self):
        qian = hexagram_from_name("乾为天")
        grad = total_score_gradient(qian)
        assert grad.shape == (6,)

    def test_sampler(self):
        qian = hexagram_from_name("乾为天")
        sampler = LangevinSampler(num_steps=20)
        trajectory = sampler.sample(qian)
        assert len(trajectory) >= 1

    def test_top_k_mutations(self):
        qian = hexagram_from_name("乾为天")
        sampler = LangevinSampler()
        results = sampler.top_k_mutations(qian, k=5)
        assert len(results) <= 5
        assert all(isinstance(s, float) for _, _, s in results)


class TestLearning:
    def test_online_learner(self):
        learner = OnlineLearner()
        qian = hexagram_from_name("乾为天")
        learner.update_from_outcome(qian, reward=0.8)
        assert learner.total_visits == 1

    def test_curiosity_decay(self):
        curiosity = CuriosityScheduler(beta_initial=0.5, beta_final=0.05)
        beta_start = curiosity.beta
        for _ in range(100):
            curiosity.step()
        assert curiosity.beta < beta_start

    def test_growth_manager(self):
        gm = GrowthManager()
        assert gm.phase == "explore"
        for _ in range(500):
            gm.advance()
        assert gm.phase == "expert"

    def test_hebbian(self):
        hebb = HebbianAssociator()
        hebb.strengthen(0, 1, 0.8)
        assert hebb.get_strength(0, 1) > 0.0


class TestStorage:
    def test_episodic_store(self):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            store = EpisodicStore(db_path=path)
            store.store(main_bits=63, reward=0.8, outcome="吉")
            assert store.count() == 1
            recent = store.query_recent(10)
            assert len(recent) == 1
            assert recent[0]["main_hex_bits"] == 63
            store.close()
        finally:
            os.unlink(path)


class TestEncoder:
    def test_rule_based_encoder(self):
        encoder = RuleBasedEncoder()
        sensor = {
            "temperature": 25.0,
            "terrain": 0.8,
            "social_proximity": 0.7,
            "resource_level": 0.5,
            "momentum": 0.3,
            "overall_favorability": 0.6,
        }
        h = encoder.encode(sensor)
        assert h.normal_order >= 0
        assert h.normal_order <= 63


class TestEndToEnd:
    def test_full_planning_loop(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        planner = TrinityPlanner()
        moe = SparseMoE(top_k=3)
        learner = OnlineLearner()
        calendar = MultiScaleCalendar()

        layers = calendar.time_layers(2026)
        time_phase = layers["年"]

        result = planner.plan(qian, grid, time_phase)

        moe_score = moe.evaluate(
            result.chain.evolved, grid,
            time_phase=time_phase,
            target_palace=1,
        )

        learner.update_from_outcome(
            result.chain.evolved,
            reward=moe_score,
        )

        chain = result.chain
        assert chain.main.name == "乾为天"
        assert chain.evolved.name != "乾为天" or result.top_mutation == 0
        assert learner.total_visits == 1
        assert 0.0 <= moe_score <= 1.0

    def test_full_five_hexagram_flow(self):
        qian = hexagram_from_name("乾为天")
        chain = FiveHexagramChain.from_current(qian)
        sampler = LangevinSampler(num_steps=10)

        results = sampler.top_k_mutations(qian, k=3)
        best_mask = results[0][1]

        chain_evolved = FiveHexagramChain.with_evolution(qian, best_mask)
        assert chain_evolved.evolved.name != chain.main.name or best_mask == 0
        assert chain_evolved.narrative_coherence() >= 0.0

    def test_self_localization(self):
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        relations = determine_six_relations(qian, grid, self_element="金")

        assert grid.self_position == 5
        assert relations[5] == "我"

        supporters = [p for p, r in relations.items() if r == "父母"]
        constraints = [p for p, r in relations.items() if r == "官鬼"]
        resources = [p for p, r in relations.items() if r == "妻财"]

        assert len(supporters) >= 0
        assert len(constraints) >= 0
        assert len(resources) >= 0


class TestTopology:
    def test_expand_topology(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=2)
        assert topo.max_depth == 2
        assert topo.total_nodes() == 1 + 9 + 81

    def test_root_is_center(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=1)
        assert topo.root.palace_position == 5
        assert topo.root.bagua == "中"

    def test_depth0_only_root(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=0)
        assert topo.total_nodes() == 1
        assert len(topo.root.children) == 0

    def test_find_by_path(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=2)
        node = topo.find_by_path((5, 9))
        assert node is not None
        assert node.palace_position == 9
        assert node.depth == 2

    def test_generation_pairs(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=1)
        pairs = topo.generation_pairs_at(1)
        assert len(pairs) >= 4  # (1,6), (2,7), (3,8), (4,9)

    def test_conflict_pairs(self):
        from zwm.topology import expand_topology

        topo = expand_topology(max_depth=1)
        pairs = topo.conflict_pairs_at(1)
        assert len(pairs) >= 4  # (1,9), (2,8), (3,7), (4,6)


class TestDayGanLiuqin:
    def test_day_gan_jia_is_wood(self):
        from zwm.scene_field.liuqin import self_element_from_day_gan

        assert self_element_from_day_gan("甲") == "木"
        assert self_element_from_day_gan("乙") == "木"

    def test_day_gan_geng_is_metal(self):
        from zwm.scene_field.liuqin import self_element_from_day_gan

        assert self_element_from_day_gan("庚") == "金"
        assert self_element_from_day_gan("辛") == "金"

    def test_day_gan_ren_is_water(self):
        from zwm.scene_field.liuqin import self_element_from_day_gan

        assert self_element_from_day_gan("壬") == "水"
        assert self_element_from_day_gan("癸") == "水"

    def test_liuqin_with_day_gan(self):
        from zwm.scene_field.liuqin import determine_six_relations
        from zwm.core.hexagram import hexagram_from_name
        from zwm.self_field.palace_graph import LuoshuGrid

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        relations = determine_six_relations(qian, grid, day_gan="甲")
        assert relations[grid.self_position] == "我"
        assert len(relations) == 9

    def test_social_field_with_day_gan(self):
        from zwm.scene_field.liuqin import social_field_vector
        from zwm.core.hexagram import hexagram_from_name
        from zwm.self_field.palace_graph import LuoshuGrid

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        field = social_field_vector(qian, grid, day_gan="庚")
        assert field["我"] == [5]
        total = sum(len(v) for v in field.values())
        assert total == 9

    def test_hexagram_palace_mapping(self):
        from zwm.scene_field.liuqin import (
            hexagram_palace_index,
            hexagram_palace_element,
        )
        from zwm.core.hexagram import hexagram_from_bits

        # 乾为天(63) → 乾宫(7) → 金
        qian = hexagram_from_bits(63)
        assert hexagram_palace_index(qian) == 7
        assert hexagram_palace_element(qian) == "金"

        # 坤为地(0) → 坤宫(0) → 土
        kun = hexagram_from_bits(0)
        assert hexagram_palace_index(kun) == 0
        assert hexagram_palace_element(kun) == "土"

        # All 64 valid
        for no in range(64):
            h = hexagram_from_bits(no)
            pid = hexagram_palace_index(h)
            pe = hexagram_palace_element(h)
            assert 0 <= pid <= 7
            assert pe in ("金", "木", "水", "火", "土")


class TestUnifiedFieldCalendar:
    def test_snapshot_with_calendar_context(self):
        from zwm.scene_field.unified_field import UnifiedField
        from zwm.core.hexagram import hexagram_from_name
        from zwm.self_field.palace_graph import LuoshuGrid

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        cal_ctx = {"年": 0.5, "月": 1.2, "日": 2.0}

        uf = UnifiedField.snapshot(qian, grid, calendar_context=cal_ctx)
        assert uf.calendar_context == cal_ctx
        assert len(uf.luoshu_field) == 9

    def test_snapshot_with_day_gan(self):
        from zwm.scene_field.unified_field import UnifiedField
        from zwm.core.hexagram import hexagram_from_name
        from zwm.self_field.palace_graph import LuoshuGrid

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()

        uf = UnifiedField.snapshot(qian, grid, day_gan="戊")
        assert uf.six_relations[grid.self_position] == "我"

    def test_evolve_with_day_gan(self):
        from zwm.scene_field.unified_field import UnifiedField
        from zwm.core.hexagram import hexagram_from_name
        from zwm.self_field.palace_graph import LuoshuGrid

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        uf = UnifiedField.snapshot(qian, grid, day_gan="甲")
        uf2 = uf.evolve(mutation_mask=0b000001, day_gan="甲")
        assert uf2.six_relations[grid.self_position] == "我"
