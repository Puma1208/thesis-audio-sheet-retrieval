"""
Unit tests for thesis.modules.MSMD_Dataset.

The tests build a tiny fake dataset in a temporary folder with the same layout
as the snippets created by thesis/scripts/create_dataset.py:

    <data_dir>/<piece>/sheet_snippets/<performance>/<performance>_<i>.png
    <data_dir>/<piece>/performances/<performance>/<performance>_<i>.png

Every image is filled with a single grey value (10 * i) so the tests can check
that images come back in the right order.

Run from the repository root with:
    pytest tests/
"""

import os

import pytest
import torch
import torchvision.io as io

from thesis.modules.MSMD_Dataset import MSMD_Dataset

SHEET_SHAPE = (160, 180)   # sheet snippet height x width used for training
AUDIO_SHAPE = (92, 20)     # spectrogram excerpt height x width used for training
SEQUENCE_LENGTH = 10       # MSMD_Dataset default snippets_batch_size


def write_image(path, value, shape):
    image = torch.full((1, *shape), value, dtype=torch.uint8)
    io.write_png(image, path)


def make_piece(data_dir, piece, performance, n_images):
    """Create one piece with one performance of n_images aligned snippet/excerpt pairs."""
    for subfolder, shape in (('sheet_snippets', SHEET_SHAPE), ('performances', AUDIO_SHAPE)):
        folder = os.path.join(data_dir, piece, subfolder, performance)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_images):
            write_image(os.path.join(folder, f'{performance}_{i}.png'), 10 * i, shape)


@pytest.fixture
def data_dir(tmp_path):
    make_piece(tmp_path, 'Piece_A', 'Piece_A_tempo-1000_Piano', n_images=15)
    return str(tmp_path)


def test_number_of_sequences(data_dir):
    # 15 images and sequences of 10 -> a sequence can start at image 0..5
    dataset = MSMD_Dataset(dir=data_dir, pieces=['Piece_A'], n_performances=1)
    assert len(dataset) == 15 - SEQUENCE_LENGTH + 1


def test_item_shapes(data_dir):
    dataset = MSMD_Dataset(dir=data_dir, pieces=['Piece_A'], n_performances=1)
    snippets, excerpts = dataset[0]
    assert tuple(snippets.shape) == (SEQUENCE_LENGTH, *SHEET_SHAPE)
    assert tuple(excerpts.shape) == (SEQUENCE_LENGTH, *AUDIO_SHAPE)


def test_images_are_in_numerical_order(data_dir):
    # sorting must be numerical (_2 before _10), not alphabetical (_10 before _2)
    dataset = MSMD_Dataset(dir=data_dir, pieces=['Piece_A'], n_performances=1)
    start = 5
    snippets, excerpts = dataset[start]
    expected = [10 * (start + k) for k in range(SEQUENCE_LENGTH)]
    assert [int(s[0, 0]) for s in snippets] == expected
    assert [int(e[0, 0]) for e in excerpts] == expected


def test_hidden_files_are_ignored(data_dir):
    # macOS creates .DS_Store files in folders opened in Finder
    piece = os.path.join(data_dir, 'Piece_A')
    for folder in (piece,
                   os.path.join(piece, 'sheet_snippets'),
                   os.path.join(piece, 'sheet_snippets', 'Piece_A_tempo-1000_Piano'),
                   os.path.join(piece, 'performances', 'Piece_A_tempo-1000_Piano')):
        open(os.path.join(folder, '.DS_Store'), 'w').close()

    dataset = MSMD_Dataset(dir=data_dir, pieces=['Piece_A'], n_performances=1)
    assert len(dataset) == 15 - SEQUENCE_LENGTH + 1
    snippets, excerpts = dataset[len(dataset) - 1]
    assert snippets.shape[0] == SEQUENCE_LENGTH


def test_short_performances_are_skipped(data_dir):
    # a performance with fewer images than one sequence cannot be used
    make_piece(data_dir, 'Piece_B', 'Piece_B_tempo-1000_Piano', n_images=SEQUENCE_LENGTH - 1)
    dataset = MSMD_Dataset(dir=data_dir, pieces=['Piece_A', 'Piece_B'], n_performances=1)
    assert len(dataset) == 15 - SEQUENCE_LENGTH + 1
