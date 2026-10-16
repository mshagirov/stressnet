"""Deprecated entry point kept for backwards compatibility.

``main.py`` is a thin shim around :mod:`predict`, which now owns all prediction
logic. The old flags below are mapped onto ``predict.py``'s raw-data mode
(``-w -i -l -c``) so existing command lines keep working unchanged, including
the ``<output>/<weights filename>/<phase>_*`` artefact names.

The old ``val`` phase is the held-out split that ``predict.py`` calls
``prediction``; it is forwarded verbatim so that ``-p val`` keeps writing
``val_scatter.html`` / ``val_preds.csv``. Prefer::

    uv run python predict.py -w WEIGHTS -i ROOT -l LABELS -c SUFFIX -p prediction
"""

from __future__ import annotations

import argparse
import sys

import predict

DEPRECATION_NOTICE = ('DEPRECATED: main.py is deprecated - use predict.py instead '
                      '(uv run python predict.py --help).')


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Deprecated alias for predict.py (StressNet predictions)')

    parser.add_argument('-o', '--output', help='directory path for output files', required=True)
    parser.add_argument('-w', '--weights', help='path to the model weights', required=True)
    parser.add_argument('-i', '--input', help='root directory for the input dataset',
                        required=True)
    parser.add_argument('-l', '--labels',
                        help='name of the spreadsheet file with the labels inside the root '
                             'directory', required=True)
    parser.add_argument('-c', '--chsuffix',
                        help='Channel directory suffix (for dir-s inside the root dir-y)',
                        required=True)
    parser.add_argument('-p', '--phase', help='Training or validation phase',
                        choices=['train', 'val'], required=True)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    print(DEPRECATION_NOTICE)

    predict.run(argparse.Namespace(
        weights=args.weights,
        dataset=None,
        input=args.input,
        labels=args.labels,
        chsuffix=args.chsuffix,
        phase=args.phase,
        output=args.output,
        task='auto',
        dataset_class='ageloc',
        batch_size=2,
    ))


if __name__ == '__main__':
    main()
