#!/usr/bin/env python3
from p3_ssl.snr_embedding_impact import build_parser, run

if __name__ == "__main__":
    run(build_parser().parse_args())
