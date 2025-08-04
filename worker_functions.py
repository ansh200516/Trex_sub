import numpy as np
import pandas as pd
import random
import collections

value = {
    letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz", start=1)
}
ALPHA = "abcdefghijklmnopqrstuvwxyz"

def dataframemaker(dicc_chunk):
    """
    Worker function to create training data with enhanced features.
    - Generates a RANDOM SAMPLE of game states to prevent overfitting.
    - Creates a 134-element feature vector:
        - 80 for word state (as before)
        - 26 for guessed letters (as before)
        - 1 for number of blanks
        - 1 for number of wrong guesses
        - 26 for frequency counts of visible letters
    """
    dfs_rows = []
    MAX_STATES_PER_WORD = 15 

    for word in dicc_chunk:
        unique_letters = list(set(word))
        if len(unique_letters) < 2:
            continue

        num_possible_states = 2**len(unique_letters) - 2 
        num_states_to_generate = min(MAX_STATES_PER_WORD, num_possible_states)
        if num_states_to_generate <= 0:
            continue

        for _ in range(num_states_to_generate):
            num_to_reveal = random.randint(1, len(unique_letters) - 1)
            revealed_letters = random.sample(unique_letters, num_to_reveal)
            
            
            num_wrong_to_add = random.randint(0, 5)
            potential_wrong_letters = [l for l in ALPHA if l not in unique_letters]
            wrong_letters = random.sample(potential_wrong_letters, min(num_wrong_to_add, len(potential_wrong_letters)))
            
            all_guessed_letters = set(revealed_letters + wrong_letters)
            
            word_state_vec = np.full(80, -1, dtype=np.int8)
            guessed_vec = np.full(26, 0, dtype=np.int8)
            target_vec = np.full(26, 0, dtype=np.int8)
            word_len = len(word)
            visible_letter_counts = collections.Counter()
            
            for char_code, letter in enumerate(ALPHA):
                if letter in all_guessed_letters:
                    guessed_vec[char_code] = 1

            blanks_count = 0
            for i in range(word_len):
                char = word[i]
                if char in revealed_letters:
                    word_state_vec[i] = value[char]
                    word_state_vec[80 - word_len + i] = value[char]
                    visible_letter_counts[char] += 1
                else:
                    word_state_vec[i] = 0
                    word_state_vec[80 - word_len + i] = 0
                    target_vec[value[char] - 1] = 1
                    blanks_count += 1
            
            blanks_vec = np.array([blanks_count], dtype=np.int8)
            wrong_guesses_vec = np.array([len(wrong_letters)], dtype=np.int8)
            
            freq_vec = np.zeros(26, dtype=np.int8)
            for letter, count in visible_letter_counts.items():
                if letter in value:
                    freq_vec[value[letter] - 1] = count

            full_row = np.concatenate([
                word_state_vec, 
                guessed_vec, 
                blanks_vec, 
                wrong_guesses_vec,
                freq_vec,
                target_vec
            ])
            dfs_rows.append(full_row)

    if not dfs_rows:
        return pd.DataFrame()

    return pd.DataFrame(dfs_rows, dtype=np.int8)