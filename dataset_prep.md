# Dataset Preparation for Hangman AI

This document explains the process of creating the training dataset for the Hangman AI model

## Overview

The goal of this notebook is to generate a comprehensive dataset for training a model to play Hangman. The fundamental idea is to take a large list of English words and, for each word, generate all possible "game states." A game state consists of the word with some letters revealed and some hidden. The dataset maps these game states to the set of correct, un-guessed letters.

- **Input:** A plain text file (`words_250000_train.txt`) containing a list of words, one per line.
- **Output:** A single Parquet file (`fd.parquet`) containing the structured training data.

## Data Generation Logic

The core of the dataset creation lies within the `dataframemaker` function, which is designed to be executed in parallel on chunks of the dictionary.

### For each word in the dictionary:

1.  **Generate Letter Combinations**: The script first identifies the unique letters in the word. Then, it generates every possible combination of these unique letters (from single letters up to all unique letters). Each combination represents a set of letters that have been "correctly guessed" in a game of Hangman.

2.  **Create Training Samples**: For each of these letter combinations, a training sample (a single row in the final dataset) is created. This sample consists of a feature vector and a target vector.

    -   **Feature Vector (`X`):** This is an 80-element integer array that represents the current state of the word being guessed.
        -   The word is encoded twice within this vector: once left-aligned and once right-aligned. This helps the model learn regardless of word length.
        -   If a letter in the word has been "guessed" (i.e., it's in the current letter combination), its position in the vector is filled with its corresponding numeric value (`a`=1, `b`=2, ..., `z`=26).
        -   If a letter has *not* been guessed, its position is filled with `0`.
        -   All unused elements in the 80-element vector are filled with `-1` as padding.

    -   **Target Vector (`y`):** This is a 26-element one-hot encoded array, where each element corresponds to a letter of the alphabet.
        -   A `1` at a specific position indicates that the corresponding letter is present in the original word but has *not* been guessed yet in the current game state. These are the correct letters for the model to predict.
        -   A `0` indicates that the letter has either already been guessed or is not in the word at all.

### Example

- **Word**: `cat`
- **Unique letters**: `{c, a, t}`
- **A possible combination (guessed letters)**: `{a}`
- **Feature Vector**:
    - `c` is unguessed -> `0`
    - `a` is guessed -> `1` (value of 'a')
    - `t` is unguessed -> `0`
    - The resulting 80-element vector would encode `[0, 1, 0]` at the start and end.
- **Target Vector**:
    - `c` and `t` are in the word but not yet guessed.
    - The 26-element vector would have a `1` at the positions for `c` and `t`, and `0`s everywhere else.

## Parallel Processing

Generating all these combinations for over 250,000 words is computationally intensive. To manage this, the script uses Python's `multiprocessing` library to parallelize the workload.

1.  **Major Chunks**: The entire word dictionary is first split into 10 large, manageable chunks.
2.  **Worker Pools**: The script iterates through these 10 chunks one by one. For each chunk, it creates a pool of worker processes (equal to the number of CPU cores available).
3.  **Sub-chunks**: The major chunk is further divided into 128 smaller sub-chunks.
4.  **Distribution**: The `pool.map()` function distributes these 128 sub-chunks among the available worker processes. Each process runs the `dataframemaker` function on its assigned sub-chunks.
5.  **Aggregation**: After all processes in the pool have finished, their results (a list of pandas DataFrames) are collected and concatenated into a single DataFrame for that major chunk.
6.  **Final Concatenation**: Finally, the DataFrames from all 10 major chunks are concatenated to form the complete dataset.

## Final Output

The resulting DataFrame is saved as `fd.parquet`.

### Schema

| Column Name(s)      | Description                                                                                              | Data Type |
| :------------------ | :------------------------------------------------------------------------------------------------------- | :-------- |
| `0` through `79`    | The 80-element feature vector representing the current, partially revealed word.                         | `int8`    |
| `a` through `z`     | The 26-element one-hot encoded target vector, indicating the correct letters remaining to be guessed.      | `int8`    |

## Usage in Model Training (`model_training.ipynb`)

The generated dataset is designed for a multi-label classification task, where the model must predict which of the 26 letters are correct guesses for a given game state. The training script implements a **one-vs-rest** strategy.

1.  **Loading the Data**: The `fd.parquet` file is loaded into a pandas DataFrame.

2.  **Feature and Label Splitting**:
    -   The **feature set (`X`)** consists of the first 80 columns (`'0'` to `'79'`), representing the encoded state of the hangman word.
    -   The **label set (`y`)** consists of the last 26 columns (`'a'` to `'z'`). Each of these columns is treated as an independent binary classification problem.

3.  **The Multi-Label Model**: The `MultiLabelCatBoostClassifier` class wraps 26 individual `catboost.CatBoostClassifier` instances. Each classifier is responsible for a single letter of the alphabet.

4.  **Training Process**:
    -   The script iterates through each of the 26 labels (from `a` to `z`).
    -   For each label (e.g., `'c'`), it trains the corresponding CatBoost model. The entire feature set `X` is used as input, and the single column for that label (e.g., `y['c']`) is used as the target.
    -   This means the first model learns to predict whether 'a' is a correct un-guessed letter, the second model learns to predict for 'b', and so on.

### Training Example Revisited

Let's use the same example:
- **Word**: `cat`
- **Guessed letters**: `{a}`
- **Feature Vector (`X`)**: The 80-element vector encoding `[0, 1, 0]...`
- **Target Vector (`y`)**: The 26-element vector where `y['c']` is 1, `y['t']` is 1, and all others (including `y['a']`) are 0.

When this sample is used for training:
- The **'a' classifier** is trained on this `X` with a target label of **0**.
- The **'b' classifier** is trained on this `X` with a target label of **0**.
- The **'c' classifier** is trained on this `X` with a target label of **1**.
- ...
- The **'t' classifier** is trained on this `X` with a target label of **1**.
- ...
- The **'z' classifier** is trained on this `X` with a target label of **0**.

This approach allows each classifier to become a specialist in determining if its assigned letter is a good guess, given the current state of the word. 
