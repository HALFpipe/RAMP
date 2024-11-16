from shutil import which

from .utils.shutil import unwrap_which

bcftools: list[str] = [unwrap_which("bcftools")]
gcta64: list[str] = [unwrap_which("gcta64")]
ldsc_munge_sumstats: list[str] = [unwrap_which("munge_sumstats.py")]
ldsc: list[str] = [unwrap_which("ldsc.py")]
metal: list[str] = [unwrap_which("metal")]
plink2: list[str] = [unwrap_which("plink2")]
raremetalworker: list[str] = [unwrap_which("raremetalworker")]
tabix: list[str] = [unwrap_which("tabix")]

micromamba: str | None = which("micromamba")
conda: str | None = which("conda")

sha256sum: list[str] = [unwrap_which("sha256sum")]


def conda_run(environment: str) -> list[str]:
    command: list[str] = list()
    if micromamba is not None:
        command.extend((micromamba, "run"))
    elif conda is not None:
        command.extend((conda, "run", "--no-capture-output"))
    else:
        raise ValueError('Could not find either "conda" or "micromamba"')
    command.extend(("--name", environment))
    return command


regenie: list[str] = [*conda_run("regenie"), "regenie"]

saige_create_sparse_grm: list[str] = [*conda_run("r-saige"), "createSparseGRM.R"]
saige_step1_fit_null_glmm: list[str] = [*conda_run("r-saige"), "step1_fitNULLGLMM.R"]
saige_step2_spa_tests: list[str] = [*conda_run("r-saige"), "step2_SPAtests.R"]

bgenix: list[str] = [*conda_run("bgenix"), "bgenix"]
