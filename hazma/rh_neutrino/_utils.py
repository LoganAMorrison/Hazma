def three_lepton_fs_generations(gen_n: int, unique: bool = False):
    assert gen_n in [1, 2, 3], f"Invalid generation {gen_n}"
    gen1 = gen_n
    gen2, gen3 = {1, 2, 3}.difference({gen_n})
    gens = [
        (gen1, gen1, gen1),
        (gen1, gen2, gen2),
        (gen1, gen3, gen3),
    ]

    if not unique:
        gens.append((gen2, gen1, gen2))
        gens.append((gen2, gen2, gen1))
        gens.append((gen3, gen1, gen3))
        gens.append((gen3, gen3, gen1))

    return gens


def three_lepton_fs_strings(gen_n: int, unique: bool = False):
    strs = ["e", "mu", "tau"]

    def gen_tup_to_str_tup(tup):
        return tuple(map(lambda gen: strs[gen - 1], tup))

    gen_tups = three_lepton_fs_generations(gen_n, unique)
    return list(map(gen_tup_to_str_tup, gen_tups))
