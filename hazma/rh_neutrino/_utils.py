from ._proto import Generation


def three_lepton_fs_generations(gen_n: Generation, unique: bool = False):
    gen1 = gen_n
    gen2, gen3 = {Generation.Fst, Generation.Snd, Generation.Trd}.difference({gen_n})
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


def three_lepton_fs_strings(gen_n: Generation, unique: bool = False):
    strs = ["e", "mu", "tau"]

    def gen_tup_to_str_tup(tup):
        return tuple(map(lambda gen: strs[gen], tup))

    gen_tups = three_lepton_fs_generations(gen_n, unique)
    return list(map(gen_tup_to_str_tup, gen_tups))
