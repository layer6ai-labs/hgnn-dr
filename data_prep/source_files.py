import time
import sys
from omegaconf import DictConfig, OmegaConf
from os import path, makedirs
from typing import Union
from urllib import request

__DIR__ = path.dirname(path.realpath(__file__))


class SourceFiles:
    def __init__(self,
        config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml")
    ) -> None:

        if isinstance(config, str):
            config = OmegaConf.load(config)

        self.directory = path.join(
            config.data_directory.format(__dir__=__DIR__),
            config.source_files.subdirectory,
        )

        self.organisms = config.organisms

        source_keys = list(config.source_files.keys())
        source_keys.remove("subdirectory")
        self.sources = OmegaConf.masked_copy(config.source_files, source_keys)


    def download(self, source: str,
                       organism: int = 5664,
                       force: bool = False) -> None:

        file = self.sources[source].file.format(org=organism)
        url = self.sources[source].url.format(org=organism)
        local_file = path.join(self.directory, file)

        if not path.isfile(local_file):
            makedirs(self.directory, exist_ok=True)
            reporthook = ReportHook(f"Downloading\t{file}")
            request.urlretrieve(url, local_file, reporthook.update)
        elif force:
            reporthook = ReportHook(f"Redownloading\t{file}")
            request.urlretrieve(url, local_file, reporthook.update)
        else:
            print(f"Found\t\t{file}")


    def download_all(self, force: bool = False) -> None:
        print("Downloading all source files (this may take a while)")
        non_org_sources = [source for source, info in self.sources.items()
                           if "{org}" not in info.file]
        for source in non_org_sources:
            self.download(source, force=force)

        org_sources = [source for source, info in self.sources.items()
                       if "{org}" in info.file]
        for organism in self.organisms:
            for source in org_sources:
                self.download(source, organism=organism, force=force)
        print("Finished downloading all source files.")


class ReportHook:
    """https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html"""

    def __init__(self, desc: str):
        self.start_time = time.time()
        self.desc = desc
        self.finished = False

    def update(self, count, block_size, total_size):
        if not self.finished:
            duration = time.time() - self.start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r%s\t%d%%, %d/%d MB, %d KB/s, %ds elapsed" % (
                             self.desc,
                             percent,
                             progress_size/1024**2,
                             total_size/1024**2,
                             speed,
                             duration))
            sys.stdout.flush()

            if percent >= 100:
                self.finished = True
                print()
