import torch
from torch.utils.data import Dataset
from . import dataset_utils
import numpy as np
import torchvision.transforms
import random

#############################################
############ SUDOKU DATASET #################
#############################################
def define_relative_encoding():
    colind = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )

    rowind = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
    )

    blockind = np.array(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
        ]
    )

    colenc = np.zeros((81, 9))
    rowenc = np.zeros((81, 9))
    blockenc = np.zeros((81, 9))
    colenc[np.arange(81), colind.flatten()] = 1
    rowenc[np.arange(81), rowind.flatten()] = 1
    blockenc[np.arange(81), blockind.flatten()] = 1
    allenc = np.concatenate([colenc, rowenc, blockenc], axis=1)
    return torch.FloatTensor(allenc[:, None, :] == allenc[None, :, :])


def construct_puzzle_solution():
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle = [[0] * 9 for i in range(9)]  # start with blank puzzle
            rows = [set(range(1, 10)) for i in range(9)]  # set of available
            columns = [set(range(1, 10)) for i in range(9)]  # numbers for each
            squares = [set(range(1, 10)) for i in range(9)]  # row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = (
                        rows[i]
                        .intersection(columns[j])
                        .intersection(squares[(i // 3) * 3 + j // 3])
                    )
                    choice = random.choice(list(choices))

                    puzzle[i][j] = choice

                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i // 3) * 3 + j // 3].discard(choice)

            # success! every cell is filled.
            return puzzle

        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass


def gen_sudoku(num):
    """
    Generates `num` games of Sudoku.
    """
    solutions = np.zeros((num, 9, 9), np.int32)
    for i in range(num):
        solutions[i] = construct_puzzle_solution()

    return solutions


@dataset_utils.register_dataset
class SudokuDataset(Dataset):
    def __init__(self, cfg, device, _):
        self.limit = cfg.data.limit
        self.device = device

        sudokus = gen_sudoku(self.limit).astype(np.float32)

        if cfg.data.powered:
            sudokus = (2 ** sudokus) / (2**9)
        else:
            sudokus = sudokus / 9.0

        self.sudokus = torch.from_numpy(sudokus).unsqueeze(1)

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        return self.sudokus[idx]


def sudoku_acc(cfg, sample, return_acc=False):
    if cfg.data.powered:
        sample = np.rint(np.log2(sample * (2**9))).astype(int)

    else:
        sample = sample * 9.0
        sample = sample.round().astype(int)

    numbers_1_N = np.arange(1, 10)

    correct = 0
    total = sample.shape[0]
    corrects = []

    for board in sample:
        if np.all(np.sort(board, axis=1) == numbers_1_N) and np.all(
            np.sort(board.T, axis=1) == numbers_1_N
        ):
            # Check 3x3 blocks
            blocks = board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
            if np.all(np.sort(blocks, axis=1) == numbers_1_N):
                correct += 1
                corrects.append(True)
            else:
                corrects.append(False)
        else:
            corrects.append(False)

    print(correct, total)
    acc = correct / total
    if return_acc:
        return acc
    print(f"Acc: {100 * acc}%, correct: {correct}, total: {total}")
