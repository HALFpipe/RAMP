from gwas.meta.index import Index


def test_index() -> None:
    index = Index()

    phenotype = "sub-0003_age-child_and_adolescent_task-emotionalconflict_epi_norm_rpt"
    index.put(phenotype)

    tags = index.get_tags(phenotype)
    assert isinstance(tags, dict)
    assert len(tags) == 4

    assert tags["sub"] == "0003"
    assert tags["age"] == "child_and_adolescent"
    assert tags["task"] == "emotionalconflict"
    assert tags["suffix"] == "epi_norm_rpt"

    index.recode("sub", "0003", "0004")
    tags = index.get_tags(phenotype)
    assert isinstance(tags, dict)
    assert len(tags) == 4

    assert tags["sub"] == "0004"
    assert tags["age"] == "child_and_adolescent"
    assert tags["task"] == "emotionalconflict"
    assert tags["suffix"] == "epi_norm_rpt"

    index.alternative("age", "child_and_adolescent", "mixed")
    matches = index.get_phenotypes(age="mixed")
    assert len(matches) == 1

    matches = index.get_phenotypes(**tags)
    assert len(matches) == 1
    index.remove(phenotype)
    matches = index.get_phenotypes(**tags)
    assert len(matches) == 0

    phenotype = "sub-0194_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm"
    index.put(phenotype)

    tags = index.get_tags(phenotype)
    assert isinstance(tags, dict)
    assert len(tags) == 5

    assert tags["sub"] == "0194"
    assert tags["from"] == "T1w"
    assert tags["to"] == "MNI152NLin2009cAsym"
    assert tags["mode"] == "image"
    assert tags["suffix"] == "xfm"

    matches = index.get_phenotypes(**tags)
    assert len(matches) == 1

    matches = index.get_phenotypes(sub="0*")
    assert len(matches) == 1

    assert len(index.get_tag_groups(["sub", "from"])) == 1


def test_index_alternative() -> None:
    index = Index()

    index.put("study-a_age-adult_population-EUR_label-wmLoadVsControlPeak13_stat-z")
    a = "study-a_age-adult_population-mixed_label-wmLoadVsControlPeak13_stat-z"
    index.put(a)

    index.put("study-b_age-adult_population-EUR_label-wmLoadVsControlPeak13_stat-z")
    b = "study-b_age-mixed_population-EUR_label-wmLoadVsControlPeak13_stat-z"
    index.put(b)

    c = "study-c_age-mixed_population-mixed_label-wmLoadVsControlPeak13_stat-z"
    index.put(c)

    d = "study-d_age-adult_population-EUR_label-wmLoadVsControlPeak13_stat-z"
    index.put(d)

    index.alternative("age", "adult", "mixed")
    index.alternative("population", "EUR", "mixed")

    phenotypes = index.get_phenotypes(
        age="mixed",
        population="mixed",
        label="wmLoadVsControlPeak13",
        stat="z",
    )
    assert phenotypes == {a, b, c, d}
