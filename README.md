# Hangman AI Solver - Project Submission

## Overview

This project implements a high-performance Hangman solving agent. The core of the solution is a **Mixture of Experts (MoE) machine learning model**, where 26 individual `CatBoostClassifier` models are trained to predict the probability of each letter being in the secret word. This architecture avoids the complexity of multi-label classification and allows each model to become a specialist for its target letter.

The agent employs a sophisticated, multi-stage guessing strategy that combines the predictive power of the ML model with robust heuristics, achieving a high success rate on an unseen test dictionary.

---

## Core Methodology

The agent's strength comes from a synergy of three key components:

1.  **Rich Feature Engineering:** The model does not just look at the word's current state. For each guess, a **134-dimensional feature vector** is generated to provide the model with maximum context about the game state. This vector includes:
    *   **Word State (80 features):** A fixed-size representation of the known letters and their positions, padded to handle variable word lengths.
    *   **Guessed Letters (26 features):** A one-hot encoded vector indicating which letters of the alphabet have already been guessed.
    *   **Game State Counters (2 features):**
        *   The number of remaining blank spaces (`_`).
        *   A count of incorrect guesses made so far.
    *   **Visible Letter Frequencies (26 features):** A vector counting the occurrences of each revealed letter (e.g., for `_pp_e`, the vector would show two 'p's and one 'e'). This helps the model understand letter distributions within the specific word.

2.  **Mixture of Experts (MoE) Model:**
    *   Instead of a single complex model, we use 26 simpler, specialized `CatBoostClassifier` models. The 'A' model learns to predict the probability of 'a' being a missing letter, the 'B' model predicts for 'b', and so on.
    *   **Training:** The models are trained on a large dataset generated from the `words_250000_train.txt` file. For each word, we create numerous partial game states to simulate real gameplay. The training process is parallelized across multiple CPU cores for efficiency and uses `warm_start` to enable incremental training on large datasets that don't fit into memory at once.
    *   **Inference:** During a game, the 134-feature vector is fed to all 26 models simultaneously. The output is a probability score for each of the 26 letters, representing the model's collective confidence.

3.  **Advanced Hybrid Guessing Strategy:** A simple ML model is not enough. The agent uses a dynamic, state-aware guessing strategy:
    *   **Stage 1: The Starter Strategy:** At the beginning of a game (a blank word), the model has no information and is at its weakest. To overcome this, the agent first guesses from a pre-computed list of high-frequency vowels and consonants (e.g., 'e', 'a', 's', 'r', 't'). This "probes" the word to reveal initial letters. The moment a correct letter is found, the agent immediately switches to its primary model-based strategy.
    *   **Stage 2: The Model-Driven Core:** Once letters are visible, the agent relies on the MoE model's predictions.
    *   **Stage 3: The Dictionary Pattern Search:** In the endgame (when 5 or fewer blanks remain), the model's pattern recognition can sometimes fail on uncommon words. To counter this, a **Dictionary Pattern Search** is activated as a "second opinion." This search analyzes sub-word patterns (e.g., `_ p p l e`) against the training dictionary to find the most likely letter for the blank.
    *   **Stage 4: The Confidence-Based Decision:** This is the most critical part of the strategy. The agent does not blindly trust its models.
        *   If either the Model or the Dictionary Pattern Search is **highly confident** (probability > 60%), its guess is chosen immediately.
        *   If **both are uncertain** (probability < 25%), the agent refuses to make a risky guess. Instead, it falls back to the safest possible move: guessing the most common letter in the entire English language that hasn't been tried yet. This conserves lives for more certain guesses later.
        *   If both are moderately confident, the agent simply chooses the guess with the higher probability.

---

## Why This Approach is Effective

*   **Robustness:** The multi-stage strategy ensures the agent performs well across all phases of the game. It uses safe heuristics when information is sparse and sophisticated models when information is rich.
*   **Adaptability:** The feature engineering allows the model to learn complex relationships. For example, it can learn that the letter 's' is highly probable at the end of a word, but less so if five incorrect guesses have already been made (indicating an unusual word).
*   **Resilience to Out-of-Dictionary Words:** The problem states that the test words are not in the training dictionary. My approach is designed for this. The ML model learns generalized patterns of the English language (e.g., 'q' is almost always followed by 'u') rather than memorizing specific words. The Dictionary Pattern Search further helps by matching sub-word components, which are often shared between training and test words even if the full words are different.
*   **Skepticism:** The confidence threshold prevents the agent from wasting tries on low-probability guesses, a common failure mode for simpler models. This "intelligent patience" is key to achieving a high win rate.

## How to Run the Code

1.  **Setup:**
    *   Ensure all required libraries are installed: `pip install -r requirements.txt`.
    *   Create a `.env` file in the root directory with your `TOKEN`.
    *   Place the `words_250000_train.txt` file in the root directory.

2.  **Training (Optional - Pre-trained model provided):**
    *   To generate the training data, run `python MOE.py` after uncommenting the `build_dataset(api)` line.
    *   To train the models, run `python MOE.py` after uncommenting the `train(api)` line. This will populate the `temp_catboost_models/` directory.

3.  **Final Model Assembly:**
    *   Run `python assembly.py`. This script loads all intermediate models from `temp_catboost_models/`, assembles them into a single `MixtureOfExperts` object, and saves the final `models/model.pkl`.

4.  **Running the Final Submission:**
    *   Ensure the final `models/model.pkl` is present.
    *   To run the 1000 recorded games for the submission, execute `python MOE.py` with the `submit(api)` line uncommented in the `if __name__ == "__main__"` block.