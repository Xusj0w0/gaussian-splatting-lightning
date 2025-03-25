from typing import List, Optional, Union

from jsonargparse import ArgumentParser

from large_scene.utils.partition import Partition

if __name__ == "__main__":
    parser = ArgumentParser()
    Partition.start(parser)
