import unittest

from llm2jev.backend.mlx.prefix_cache import PrefixCache, shared_prefixes


class FakeCache:
    def __init__(self, tokens, *, nbytes=None, trimmable=True, trim_limit=None):
        self.tokens = list(tokens)
        self.nbytes = len(self.tokens) if nbytes is None else nbytes
        self.trimmable = trimmable
        self.trim_limit = trim_limit

    def is_trimmable(self):
        return self.trimmable

    def trim(self, count):
        count = min(count, len(self.tokens))
        if self.trim_limit is not None:
            count = min(count, self.trim_limit)
        if count:
            del self.tokens[-count:]
        return count


class OpaqueCache:
    """A model-specific state that supports copying but has no trim API."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.nbytes = len(self.tokens)


class RejectingCache(FakeCache):
    def trim(self, count):
        raise NotImplementedError("cannot trim this state")


class PrefixCacheTests(unittest.TestCase):
    def test_accepts_positive_limits_and_accounts_for_every_layer(self):
        cache = PrefixCache(max_entries=1, max_bytes=9)
        cache.put([1, 2], [FakeCache([1, 2], nbytes=4), FakeCache([1, 2], nbytes=5)])

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 9)
        layers, reused = cache.fetch([1, 2, 3])
        self.assertEqual(reused, 2)
        self.assertEqual([layer.tokens for layer in layers], [[1, 2], [1, 2]])

    def test_rejects_boolean_noninteger_and_negative_limits(self):
        for name in ("max_entries", "max_bytes"):
            for value in (True, False, -1, 1.5, "2", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        PrefixCache(**{name: value})

    def test_either_zero_limit_disables_storage(self):
        for name in ("max_entries", "max_bytes"):
            with self.subTest(name=name):
                cache = PrefixCache(**{name: 0})
                cache.put([1], [FakeCache([1], nbytes=0)])
                self.assertEqual(len(cache), 0)
                self.assertEqual(cache.nbytes, 0)
                self.assertEqual(cache.fetch([1]), (None, 0))

    def test_ignores_empty_tokens_and_missing_layer_states(self):
        cache = PrefixCache()
        cache.put([], [FakeCache([])])
        cache.put([1], [])

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.fetch([]), (None, 0))

    def test_put_and_each_fetch_create_independent_snapshots(self):
        cache = PrefixCache()
        original = FakeCache([1, 2])
        cache.put([1, 2], [original])
        original.tokens.append(3)
        original.nbytes = 100
        first, reused = cache.fetch([1, 2])
        first[0].tokens.append(4)
        second, _ = cache.fetch([1, 2])

        self.assertEqual(reused, 2)
        self.assertEqual(second[0].tokens, [1, 2])
        self.assertEqual(cache.nbytes, 2)

    def test_selects_longest_prefix_independent_of_insertion_order(self):
        cache = PrefixCache()
        cache.put([1, 2, 3], [FakeCache([1, 2, 3])])
        cache.put([1], [FakeCache([1])])
        cache.put([1, 2], [FakeCache([1, 2])])

        layers, reused = cache.fetch([1, 2, 3, 4])

        self.assertEqual(reused, 3)
        self.assertEqual(layers[0].tokens, [1, 2, 3])

    def test_trims_longer_cache_to_full_query_without_mutating_entry(self):
        cache = PrefixCache()
        cache.put([1, 2, 3, 4], [FakeCache([1, 2, 3, 4])])

        layers, reused = cache.fetch([1, 2])
        intact, _ = cache.fetch([1, 2, 3, 4])

        self.assertEqual(reused, 2)
        self.assertEqual(layers[0].tokens, [1, 2])
        self.assertEqual(intact[0].tokens, [1, 2, 3, 4])
        self.assertEqual(cache.nbytes, 4)

    def test_trims_diverging_sequence_to_longest_common_prefix(self):
        cache = PrefixCache()
        cache.put([1, 2, 3, 4], [FakeCache([1, 2, 3, 4])])

        layers, reused = cache.fetch([1, 2, 5, 6, 7])

        self.assertEqual(reused, 2)
        self.assertEqual(layers[0].tokens, [1, 2])

    def test_untrimmable_and_opaque_layers_support_exact_and_strict_prefix_hits(self):
        for layer in (FakeCache([1, 2], trimmable=False), OpaqueCache([1, 2])):
            with self.subTest(layer=type(layer).__name__):
                cache = PrefixCache()
                cache.put([1, 2], [layer])
                for tokens in ([1, 2], [1, 2, 3]):
                    result, reused = cache.fetch(tokens)
                    self.assertEqual(reused, 2)
                    self.assertEqual(result[0].tokens, [1, 2])
                self.assertEqual(cache.fetch([1]), (None, 0))
                self.assertEqual(cache.fetch([1, 3]), (None, 0))

    def test_all_layers_must_support_complete_trimming(self):
        for final_layer in (
            FakeCache([1, 2, 3], trimmable=False),
            FakeCache([1, 2, 3], trim_limit=1),
            OpaqueCache([1, 2, 3]),
            RejectingCache([1, 2, 3]),
        ):
            with self.subTest(layer=type(final_layer).__name__):
                cache = PrefixCache()
                cache.put([1, 2, 3], [FakeCache([1, 2, 3]), final_layer])
                self.assertEqual(cache.fetch([1, 4]), (None, 0))
                intact, reused = cache.fetch([1, 2, 3])
                self.assertEqual(reused, 3)
                self.assertEqual([layer.tokens for layer in intact], [[1, 2, 3]] * 2)

    def test_falls_back_to_shorter_entry_when_longest_match_cannot_trim(self):
        cache = PrefixCache()
        cache.put([1], [OpaqueCache([1])])
        cache.put([1, 2, 3], [OpaqueCache([1, 2, 3])])

        layers, reused = cache.fetch([1, 2, 4])

        self.assertEqual(reused, 1)
        self.assertEqual(layers[0].tokens, [1])

    def test_namespaces_isolate_other_evidence_and_share_byte_limit(self):
        cache = PrefixCache()
        cache.put([1, 2], [FakeCache([10, 20])], namespace=b"image-a")
        cache.put([1, 2], [FakeCache([30, 40])], namespace=b"image-b")

        first, _ = cache.fetch([1, 2], namespace=b"image-a")
        second, _ = cache.fetch([1, 2], namespace=b"image-b")

        self.assertEqual(first[0].tokens, [10, 20])
        self.assertEqual(second[0].tokens, [30, 40])
        self.assertEqual(cache.fetch([1, 2]), (None, 0))
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 4)

    def test_entry_limit_evicts_least_recently_used_after_exact_hit(self):
        cache = PrefixCache(max_entries=2)
        cache.put([1], [FakeCache([1])])
        cache.put([2], [FakeCache([2])])
        cache.fetch([1])
        cache.put([3], [FakeCache([3])])

        self.assertEqual(cache.fetch([2]), (None, 0))
        self.assertEqual(cache.fetch([1])[1], 1)
        self.assertEqual(cache.fetch([3])[1], 1)
        self.assertEqual(cache.nbytes, 2)

    def test_partial_hit_also_promotes_entry_before_eviction(self):
        cache = PrefixCache(max_entries=2)
        cache.put([1, 2], [FakeCache([1, 2])])
        cache.put([3], [FakeCache([3])])
        cache.fetch([1, 4])
        cache.put([5], [FakeCache([5])])

        self.assertEqual(cache.fetch([3]), (None, 0))
        self.assertEqual(cache.fetch([1, 2])[1], 2)
        self.assertEqual(cache.nbytes, 3)

    def test_byte_limit_can_evict_multiple_entries(self):
        cache = PrefixCache(max_bytes=5)
        for token in (1, 2):
            cache.put([token], [FakeCache([token], nbytes=2)])
        cache.put([3], [FakeCache([3], nbytes=4)])

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)
        self.assertEqual(cache.fetch([1]), (None, 0))
        self.assertEqual(cache.fetch([2]), (None, 0))

    def test_oversized_entry_is_ignored_without_evicting_useful_state(self):
        cache = PrefixCache(max_bytes=3)
        cache.put([1], [FakeCache([1], nbytes=2)])
        cache.put([2], [FakeCache([2], nbytes=4)])

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)
        self.assertEqual(cache.fetch([1])[1], 1)
        self.assertEqual(cache.fetch([2]), (None, 0))

    def test_replacing_entry_updates_memory_and_recency(self):
        cache = PrefixCache(max_entries=2)
        cache.put([1], [FakeCache([1], nbytes=2)])
        cache.put([2], [FakeCache([2], nbytes=2)])
        cache.put([1], [FakeCache([1], nbytes=3)])
        self.assertEqual(cache.nbytes, 5)
        cache.put([3], [FakeCache([3], nbytes=1)])

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 4)
        self.assertEqual(cache.fetch([2]), (None, 0))

    def test_clear_releases_all_entries_and_accounting(self):
        cache = PrefixCache()
        cache.put([1], [FakeCache([1])])
        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(cache.fetch([1]), (None, 0))


class SharedPrefixesTests(unittest.TestCase):
    def test_finds_each_branch_shortest_first_without_all_intermediate_prefixes(self):
        sequences = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 9], [2, 8, 1], [2, 8, 2]]

        self.assertEqual(shared_prefixes(sequences), ((1,), (2, 8), (1, 2, 3)))
        self.assertEqual(sequences[0], [1, 2, 3, 4])

    def test_duplicate_and_strict_prefix_sequences_keep_complete_shared_path(self):
        sequences = [[1, 2, 3], [1, 2], [1, 2, 3], [1, 2]]

        self.assertEqual(shared_prefixes(sequences), ((1, 2), (1, 2, 3)))

    def test_no_common_path_produces_no_empty_prefix(self):
        for sequences in ([], [[]], [[1]], [[1], [2]], [[], [], [1]]):
            with self.subTest(sequences=sequences):
                self.assertEqual(shared_prefixes(sequences), ())

    def test_same_length_prefixes_have_deterministic_token_order(self):
        sequences = [[3, 1], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2]]

        self.assertEqual(shared_prefixes(sequences), ((1,), (2,), (3,)))


if __name__ == "__main__":
    unittest.main()
