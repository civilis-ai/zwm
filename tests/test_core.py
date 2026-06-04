from zwm.core.yao import YANG, YIN, YaoLine
from zwm.core.trigram import (
    Trigram,
    trigram_from_index,
    trigram_from_lines,
)
from zwm.core.hexagram import (
    Hexagram,
    all_hexagrams,
    fuxi_square_hexagram,
    hexagram_from_bits,
    hexagram_from_name,
    hexagram_from_phase_vector,
    hexagram_from_trigrams,
)


class TestYaoLine:
    def test_yang_is_yang(self):
        assert YANG.is_yang
        assert not YANG.is_yin

    def test_yin_is_yin(self):
        assert YIN.is_yin
        assert not YIN.is_yang

    def test_flip_yang_to_yin(self):
        assert YANG.flip() == YIN

    def test_flip_yin_to_yang(self):
        assert YIN.flip() == YANG

    def test_flip_is_idempotent(self):
        assert YANG.flip().flip() == YANG
        assert YIN.flip().flip() == YIN

    def test_phase_yang_is_0(self):
        assert YANG.phase == 0

    def test_phase_yin_is_1(self):
        assert YIN.phase == 1

    def test_immutable(self):
        try:
            YANG.is_yang = False
            assert False, "Should not be mutable"
        except AttributeError:
            pass

    def test_hash_consistent(self):
        assert hash(YANG) == hash(YANG)
        assert hash(YIN) == hash(YIN)
        assert hash(YANG) != hash(YIN)

    def test_order(self):
        assert YIN < YANG


class TestTrigram:
    def test_qian_all_yang(self):
        assert Trigram.QIAN.lower.is_yang
        assert Trigram.QIAN.middle.is_yang
        assert Trigram.QIAN.upper.is_yang

    def test_kun_all_yin(self):
        assert Trigram.KUN.lower.is_yin
        assert Trigram.KUN.middle.is_yin
        assert Trigram.KUN.upper.is_yin

    def test_eight_unique(self):
        all_tri = [
            Trigram.QIAN, Trigram.DUI, Trigram.LI, Trigram.ZHEN,
            Trigram.XUN, Trigram.KAN, Trigram.GEN, Trigram.KUN,
        ]
        assert len(set(all_tri)) == 8

    def test_index_range(self):
        for i in range(8):
            t = trigram_from_index(i)
            assert t.index == i

    def test_qian_element_is_metal(self):
        assert Trigram.QIAN.element == "金"

    def test_liquid_element_is_fire(self):
        assert Trigram.LI.element == "火"

    def test_kun_element_is_earth(self):
        assert Trigram.KUN.element == "土"

    def test_qian_pre_heaven_order(self):
        assert Trigram.QIAN.pre_heaven_order == 1

    def test_kun_pre_heaven_order(self):
        assert Trigram.KUN.pre_heaven_order == 8


class TestHexagram:
    def test_64_unique(self):
        all_h = all_hexagrams()
        assert len(all_h) == 64
        assert len(set(all_h)) == 64

    def test_qian_all_yang(self):
        qian = hexagram_from_bits(0b111111)
        assert qian.name == "乾为天"
        for line in qian.lines:
            assert line.is_yang

    def test_kun_all_yin(self):
        kun = hexagram_from_bits(0b000000)
        assert kun.name == "坤为地"
        for line in kun.lines:
            assert line.is_yin

    def test_mutate_single_yao(self):
        qian = hexagram_from_bits(0b111111)
        gou = qian.mutate(0b000001)
        assert gou.name == "天风姤"

    def test_mutate_all_yao(self):
        qian = hexagram_from_bits(0b111111)
        kun = qian.mutate(0b111111)
        assert kun.name == "坤为地"

    def test_mutate_is_closed(self):
        for h in all_hexagrams():
            for mask in range(1, 64):
                mutated = h.mutate(mask)
                assert 0 <= mutated.normal_order <= 63

    def test_interlock(self):
        qian = hexagram_from_bits(0b111111)
        inter = qian.interlock()
        assert inter.name == "乾为天"

    def test_reverse(self):
        qian = hexagram_from_bits(0b111111)
        rev = qian.reverse()
        assert rev.name == "乾为天"

        tai = hexagram_from_name("地天泰")
        pi = hexagram_from_name("天地否")
        assert tai.reverse() == pi
        assert pi.reverse() == tai

    def test_complement(self):
        qian = hexagram_from_bits(0b111111)
        comp = qian.complement()
        assert comp.name == "坤为地"

        kun = hexagram_from_bits(0b000000)
        assert kun.complement().name == "乾为天"

    def test_complement_is_involution(self):
        for h in all_hexagrams():
            assert h.complement().complement() == h

    def test_reverse_is_involution(self):
        for h in all_hexagrams():
            assert h.reverse().reverse() == h

    def test_square_diagonal_pure_hexagrams(self):
        for i in range(8):
            h = fuxi_square_hexagram(i, i)
            assert h.lower_trigram == h.upper_trigram

    def test_binary_str_length(self):
        for h in all_hexagrams():
            assert len(h.binary_str) == 6
            assert all(c in "01" for c in h.binary_str)

    def test_from_name_roundtrip(self):
        for h in all_hexagrams():
            assert hexagram_from_name(h.name) == h

    def test_phase_vector_roundtrip(self):
        for h in all_hexagrams():
            pv = h.phase_vector
            restored = hexagram_from_phase_vector(pv)
            assert restored == h

    def test_hamming_distance_mutation(self):
        qian = hexagram_from_bits(0b111111)
        assert qian.hamming_distance(qian) == 0
        assert qian.hamming_distance(qian.mutate(0b000001)) == 1
        assert qian.hamming_distance(qian.mutate(0b000011)) == 2
        assert qian.hamming_distance(qian.complement()) == 6


class TestVSA:
    def test_codebook_covers_all_64(self):
        from zwm.hexaembed.vsa import VSACodebook
        cb = VSACodebook(dim=1000, seed=42)
        for bits in range(64):
            vec = cb.encode_hexagram(bits)
            assert vec.shape == (1000,)
            decoded = cb.decode_to_hexagram(vec)
            assert decoded == bits

    def test_bind_unbind(self):
        from zwm.hexaembed.vsa import bind, unbind
        import numpy as np
        a = np.array([1, -1, 1, -1], dtype=np.int8)
        b = np.array([-1, 1, 1, -1], dtype=np.int8)
        c = bind(a, b)
        assert np.array_equal(unbind(c, b), a)


class TestSpectrum:
    def test_resonance_qian_max(self):
        from zwm.spectrum import FrequencySpectrum, HexagramPhaseVector
        pv = HexagramPhaseVector.from_bits(0b111111)
        spectrum = FrequencySpectrum(pv)
        r = spectrum.resonance()
        assert r > 0, f"Qian should have positive resonance, got {r}"

    def test_resonance_kun_positive(self):
        from zwm.spectrum import FrequencySpectrum, HexagramPhaseVector
        pv = HexagramPhaseVector.from_bits(0b000000)
        spectrum = FrequencySpectrum(pv)
        r = spectrum.resonance()
        assert r > 0, f"Kun should have positive resonance, got {r}"

    def test_compute_interference(self):
        from zwm.spectrum import (
            FrequencySpectrum,
            HexagramPhaseVector,
            compute_interference,
        )
        pv = HexagramPhaseVector.from_bits(0b111111)
        spectrum = FrequencySpectrum(pv)
        result = compute_interference(spectrum)
        assert 0.0 <= result.fortune_index <= 1.0
        assert result.is_harmonious

    def test_scene_spectrum(self):
        from zwm.spectrum import FrequencySpectrum, HexagramPhaseVector, SceneSpectrum
        pv = HexagramPhaseVector.from_bits(0b111111)
        main = FrequencySpectrum(pv)
        inter = FrequencySpectrum(pv.mutate(0b000110))
        evolved = FrequencySpectrum(pv.mutate(0b000001))
        reversed_ = FrequencySpectrum(pv.reverse())
        complement = FrequencySpectrum(pv.complement())
        scene = SceneSpectrum(main, inter, evolved, reversed_, complement)
        coherence = scene.narrative_coherence()
        assert 0.0 <= coherence <= 1.0
