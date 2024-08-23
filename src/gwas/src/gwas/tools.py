from .utils.shutil import unwrap_which

bcftools: list[str] = [unwrap_which("bcftools")]
gcta64: list[str] = [unwrap_which("gcta64")]
plink2: list[str] = [unwrap_which("plink2")]
raremetalworker: list[str] = [unwrap_which("raremetalworker")]
tabix: list[str] = [unwrap_which("tabix")]

conda: list[str] = [unwrap_which("conda")]


def conda_run(environment: str) -> list[str]:
    return [*conda, "run", "--no-capture-output", "--name", environment]


regenie: list[str] = [*conda_run("regenie"), "regenie"]

saige_createsparsegrm: list[str] = [*conda_run("r-saige"), "createSparseGRM.R"]
saige_step1_fitnullglmm: list[str] = [*conda_run("r-saige"), "step1_fitNULLGLMM.R"]
saige_step2_spatests: list[str] = [*conda_run("r-saige"), "step2_SPAtests.R"]

bgenix: list[str] = [*conda_run("bgenix"), "bgenix"]
