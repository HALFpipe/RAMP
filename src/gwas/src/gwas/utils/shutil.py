from shutil import which


def unwrap_which(command: str) -> str:
    executable = which(command)
    if executable is None:
        raise ValueError(f"Could not find executable for {command}")
    return executable
