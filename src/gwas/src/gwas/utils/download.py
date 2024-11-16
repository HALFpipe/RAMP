from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import IO

import requests
import requests.adapters
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


@dataclass
class Session(AbstractContextManager):
    session: requests.Session = field(default_factory=requests.session)

    def __post_init__(self):
        max_retries = Retry(
            total=8,
            backoff_factor=10,
            status_forcelist=tuple(range(400, 600)),
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
        for protocol in ["http", "https"]:
            self.session.mount(f"{protocol}://", adapter)

    def __enter__(self) -> requests.Session:
        return self.session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.session.close()


def download(source: str, destination: IO[bytes]) -> None:
    with Session() as session, session.get(source, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        import pdb

        pdb.set_trace()
        block_size = 256 * 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            for block in response.iter_content(block_size):
                if block:  # Filter out keep-alive new chunks
                    progress_bar.update(len(block))
                    destination.write(block)
