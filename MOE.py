import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import json
import requests
import random
import string
import secrets
import time
import re
import collections
import os
import pickle
import logging
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,balanced_accuracy_score
from dotenv import load_dotenv
from worker_functions import dataframemaker
import multiprocessing as mp
import pyarrow.parquet as pq
load_dotenv()


import multiprocessing as mp

X_data = None
y_data = None
full_dictionary_global = []

def init_worker(X_chunk, y_chunk):
    global X_data, y_data
    X_data = X_chunk
    y_data = y_chunk

def _train_classifier_wrapper(args):
    idx, model_path, alpha_char, iterations, thread_count = args
    global X_data, y_data
    
    target_column = y_data[alpha_char]
    if len(target_column.unique()) < 2:
        logging.warning(f"--- Skipping letter '{alpha_char}' for this chunk: Only one class present. ---")
        return True

    classifier = CatBoostClassifier(
        iterations=iterations, 
        task_type='CPU', 
        thread_count=thread_count,
        l2_leaf_reg=30,
        learning_rate=0.03,
    )

    init_model = None
    if os.path.exists(model_path):
        logging.info(f"[Letter {alpha_char}] Loading existing model from {model_path}")
        init_model = CatBoostClassifier()
        init_model.load_model(model_path)
        logging.info(f"[Letter {alpha_char}] Previous model iterations: {init_model.tree_count_}")
    else:
        logging.info(f"[Letter {alpha_char}] No existing model found, training from scratch")

    classifier.fit(
        X_data,
        y_data[alpha_char],
        init_model=init_model,
        verbose=100
    )

    logging.info(f"[Letter {alpha_char}] Final model iterations: {classifier.tree_count_}")
    logging.info(f"[Letter {alpha_char}] Saving model to {model_path}")
    classifier.save_model(model_path)
    return True

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


value = {
    letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz", start=1)
}


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.vowels='aeiou'
        self.strategy='vowel' 
        
        self.value_map = {letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz", start=1)}
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.vowel_probabilities = self._calculate_vowel_probability()
        
        self.current_dictionary = []

        try:
            self.model = self.MixtureOfExperts.load("models/model_134_features_15.pkl") 
            logging.info("Model loaded successfully.")
        except FileNotFoundError:
            self.model = None
            logging.warning("Model not found, will use fallback guessing strategy.")
        
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com']
        data = {link: 0 for link in links}
        for link in links:
            requests.get(link)
            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s
        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    class MixtureOfExperts:
        def __init__(self, numclasses=26):
            self.iterations = 2000
            self.classifiers = [None] * numclasses
            self.alpha = "abcdefghijklmnopqrstuvwxyz"
            self.model_dir ="temp_catboost_models"
        
        def fit(self, X, y, warm_start=False):
            if not warm_start and os.path.exists(self.model_dir):
                shutil.rmtree(self.model_dir)
            
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            processes = max(1, mp.cpu_count() - 2)
            thread_count_per_worker = 1

            tasks = []
            for i in range(len(self.alpha)):
                model_path = os.path.join(self.model_dir, f"model_{self.alpha[i]}.cbm")
                tasks.append((i, model_path, self.alpha[i], self.iterations, thread_count_per_worker))

            with mp.Pool(processes=processes, initializer=init_worker, initargs=(X, y)) as pool:
                logging.info(f"\nDistributing training for {len(tasks)} models across {processes} workers...")
                pool.map(_train_classifier_wrapper, tasks)
            
            logging.info(f"--- Finished training chunk. Model states saved to disk. ---")

        def load_trained_models(self):
            logging.info("--- Loading final trained models from disk ---")
            for i in range(len(self.alpha)):
                model_path = os.path.join(self.model_dir, f"model_{self.alpha[i]}.cbm")
                if os.path.exists(model_path):
                    self.classifiers[i] = CatBoostClassifier().load_model(model_path)
                else:
                    logging.warning(f"Warning: Model file not found for letter '{self.alpha[i]}'")

        def cleanup(self):
            if os.path.exists(self.model_dir):
                logging.info("--- Cleaning up temporary model files ---")
                shutil.rmtree(self.model_dir)

        def predict(self, X):
            if not any(self.classifiers):
                self.load_trained_models()

            predictions = np.zeros((len(X), len(self.classifiers)))
            for i, clf in enumerate(self.classifiers):
                if clf is not None:
                    predictions[:, i] = clf.predict_proba(X)[:, 1]
            return predictions
        
        def save(self,filename):
            with open(filename,'wb') as file:
                pickle.dump(self,file)
        
        @classmethod
        def load(cls,filename):
            with open(filename,'rb') as file:
                return pickle.load(file)

        @staticmethod
        def accuracy(y_pred, y_test):
            cols = y_test.columns
            x = []
            for i in range(len(cols)):
                if cols[i] in y_test:
                    logging.info(f"{cols[i]} : {confusion_matrix(np.round(y_pred[:,i]),y_test[cols[i]])}")
                    logging.info(f"{cols[i]} : {balanced_accuracy_score(np.round(y_pred[:,i]),y_test[cols[i]])}")
                    x.append(balanced_accuracy_score(np.round(y_pred[:,i]),y_test[cols[i]]))
            return np.mean(x) if x else 0.0
        
    def _get_dictionary_pattern_guess(self, clean_word, window_size=5):
        letter_frequencies = collections.Counter()
        blank_indices = [i for i, char in enumerate(clean_word) if char == '_']
        if not blank_indices:
            return None, 0

        for i in blank_indices:
            start = max(0, i - window_size + 1)
            end = min(len(clean_word), i + window_size)
            for j in range(start, i + 1):
                if j + window_size <= end:
                    sub_pattern = clean_word[j : j + window_size]
                    if sub_pattern.count('_') == 1:
                        regex_pattern = sub_pattern.replace('_', '.')
                        blank_pos_in_sub = sub_pattern.find('_')
                        try:
                            for word_dict in self.full_dictionary:
                                if len(word_dict) == len(clean_word):
                                    for match in re.finditer(regex_pattern, word_dict):
                                        letter = word_dict[match.start() + blank_pos_in_sub]
                                        if letter not in self.guessed_letters:
                                            letter_frequencies[letter] += 1
                        except re.error:
                            continue

        if not letter_frequencies:
            return None, 0
        
        most_common_letter, count = letter_frequencies.most_common(1)[0]
        probability = count / sum(letter_frequencies.values())
        return most_common_letter, probability

    def guess(self, word):
        ALPHA = "abcdefghijklmnopqrstuvwxyz"
        clean_word = word[::2]
        logging.info(f"\n----- NEW GUESS | Word: {clean_word} | Guessed: {''.join(sorted(self.guessed_letters))} -----")

        if self.strategy == 'vowel':
            logging.info("Strategy: Starter List")
            starter_letters = self.vowel_probabilities.get(len(clean_word), list('eaiou'))
            starter_letters.extend(['s', 'r', 't', 'n', 'l', 'c'])
            for letter in starter_letters:
                if letter not in self.guessed_letters:
                    if any(c != '_' for c in clean_word):
                        logging.info("Hit found! Switching to model strategy.")
                        self.strategy = 'model'
                        break
                    return letter
            logging.info("Starter list exhausted. Switching to model strategy.")
            self.strategy = 'model'

        if self.strategy == 'model' and self.model:
            logging.info("Strategy: Model + Dictionary Pattern Search + Confidence Check")
            
            word_state_vec = np.full(80, -1, dtype=np.int8)
            k = len(clean_word)
            visible_letter_counts = collections.Counter()
            correct_guesses = set()
            for i in range(k):
                char = clean_word[i]
                if char != '_':
                    word_state_vec[i] = self.value_map.get(char, 0)
                    word_state_vec[80 - k + i] = self.value_map.get(char, 0)
                    visible_letter_counts[char] += 1
                    correct_guesses.add(char)
                else:
                    word_state_vec[i] = 0
                    word_state_vec[80 - k + i] = 0
            
            guessed_vec = np.full(26, 0, dtype=np.int8)
            for i, letter in enumerate(ALPHA):
                if letter in self.guessed_letters:
                    guessed_vec[i] = 1

            blanks_count = np.array([clean_word.count('_')], dtype=np.int8)
            wrong_guesses_count = np.array([len(set(self.guessed_letters) - correct_guesses)], dtype=np.int8)
            freq_vec = np.zeros(26, dtype=np.int8)
            for letter, count in visible_letter_counts.items():
                if letter in self.value_map:
                    freq_vec[self.value_map[letter] - 1] = count
            
            full_feature_vector = np.concatenate([word_state_vec, guessed_vec, blanks_count, wrong_guesses_count, freq_vec])
            df = pd.DataFrame([full_feature_vector], columns=[str(x) for x in range(134)])

            predictions = self.model.predict(df)[0]
            prob_map = {ALPHA[i]: predictions[i] for i in range(len(ALPHA))}
            ml_letter, ml_prob = None, 0.0
            sorted_probs = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
            for letter, prob in sorted_probs:
                if letter not in self.guessed_letters:
                    ml_letter, ml_prob = letter, prob
                    break
            logging.info(f"Model Top Guess: '{ml_letter}' (Prob: {ml_prob:.4f})")

            pattern_letter, pattern_prob = None, 0.0
            if clean_word.count('_') <= 2:
                pattern_letter, pattern_prob = self._get_dictionary_pattern_guess(clean_word)
                if pattern_letter:
                    logging.info(f"Dictionary Pattern Top Guess: '{pattern_letter}' (Prob: {pattern_prob:.4f})")

            if pattern_letter is not None:
                logging.info(f"DECISION: Dictionary Pattern is highly confident. Trusting it: '{pattern_letter}'")
                return pattern_letter
            if ml_prob > 0.6:
                logging.info(f"DECISION: Model is highly confident. Trusting it: '{ml_letter}'")
                return ml_letter
            
            CONFIDENCE_THRESHOLD = 0.25
            if ml_prob < CONFIDENCE_THRESHOLD and pattern_prob < CONFIDENCE_THRESHOLD:
                logging.warning(f"CONFIDENCE LOW. Falling back to global frequency.")
                for letter, _ in self.full_dictionary_common_letter_sorted:
                    if letter not in self.guessed_letters:
                        logging.info(f"DECISION: Safe fallback guess: '{letter}'")
                        return letter
            
            if pattern_prob > ml_prob:
                logging.info(f"DECISION: Dictionary Pattern prob > Model prob. Choosing Dictionary Pattern: '{pattern_letter}'")
                return pattern_letter
            else:
                logging.info(f"DECISION: Model prob >= Dictionary Pattern prob. Choosing Model: '{ml_letter}'")
                return ml_letter

        logging.warning("Fallback: Model not found or failed. Using basic frequency.")
        for letter, _ in self.full_dictionary_common_letter_sorted:
            if letter not in self.guessed_letters:
                return letter
        return 'a' 

    def _calculate_vowel_probability(self):
        vowel_counts_by_length=collections.defaultdict(lambda: collections.Counter())
        for word in self.full_dictionary:
            length=len(word)
            for char in word:
                if char in self.vowels:
                    vowel_counts_by_length[length][char]+=1
        vowel_probabilities={}
        for length, counts in vowel_counts_by_length.items():
            vowel_probabilities[length]=[v for v,c in counts.most_common()]
        return vowel_probabilities

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        self.strategy='vowel' 
        
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                logging.info("="*40)
                logging.info(f"New Game Started! ID: {game_id}, Tries: {tries_remains}, Word: {word}")
                logging.info("="*40)

            while tries_remains > 0:
                guess_letter = self.guess(word)
                self.guessed_letters.append(guess_letter)

                if verbose:
                    logging.info(f"Guessing letter: '{guess_letter}'")
                    
                try:    
                    res = self.request("/guess_letter", {"game_id":game_id, "letter":guess_letter})
                except HangmanAPIError as e:
                    logging.error(f'HangmanAPIError on guess: {e.result}')
                    continue
                except Exception as e:
                    logging.error(f'Unknown exception on guess: {e}')
                    raise e
               
                if verbose:
                    logging.info(f"Server response: {res}")
                
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                
                if status=="success":
                    if verbose:
                        logging.info(f"SUCCESS! Game {game_id} won.")
                    return True
                elif status=="failed":
                    reason = res.get('reason', 'Number of tries exceeded!')
                    if verbose:
                        logging.warning(f"FAILED. Game {game_id} lost. Reason: {reason}")
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                logging.error("Failed to start a new game.")
        return False
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        if self.access_token:
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        try:
            response = self.session.request(
                method or "GET",
                self.hangman_url + path,
                timeout=self.timeout,
                params=args,
                data=post_args,
                verify=False
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"A network error occurred: {e}")
            raise HangmanAPIError({"error": "Network Error", "error_description": str(e)})

        if 'json' in response.headers.get('content-type', ''):
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            result = {"access_token": query_str["access_token"][0]}
            if "expires" in query_str:
                result["expires"] = query_str.get("expires", [None])[0]
        else:
            raise HangmanAPIError({'error': 'Unexpected Content Type', 'error_description': response.text})

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)
        
def build_dataset(api):
    logging.info("Starting build_dataset...")
    logging.info("Loading dictionary...")
    words = api.build_dictionary("words_250000_train.txt")
    logging.info(f"Dictionary loaded with {len(words)} words")
    
    PROCESSES = max(1, mp.cpu_count() - 2)
    logging.info(f"Using {PROCESSES} processes for dataset creation")
    word_chunks = np.array_split(words, PROCESSES * 5) 
    logging.info(f"Split words into {len(word_chunks)} chunks")

    logging.info("Starting parallel processing...")
    with mp.Pool(PROCESSES) as pool:
        logging.info("Pool created, mapping dataframemaker...")
        results = pool.map(dataframemaker, word_chunks)
        logging.info("Mapping complete")

    logging.info("Concatenating results...")
    final_df = pd.concat([df for df in results if not df.empty]).reset_index(drop=True)
    logging.info(f"Concatenation complete. Got {len([df for df in results if not df.empty])} valid chunks")
    
    feature_count = 134
    feature_cols = [str(x) for x in range(feature_count)]
    target_cols = list("abcdefghijklmnopqrstuvwxyz")
    final_df.columns = feature_cols + target_cols
    
    dataset_filename = "dataset_134_features_15.parquet"
    logging.info(f"Dataset created with {len(final_df)} rows. Saving to {dataset_filename}...")
    final_df.to_parquet(dataset_filename)
    logging.info("Dataset saved successfully.")
    logging.info("build_dataset complete!")
    

def train(api):
    dataset_filename = "dataset_134_features_15.parquet"
    try:
        parquet_file = pq.ParquetFile(dataset_filename)
    except FileNotFoundError:
        logging.error(f"Error: {dataset_filename} not found. Please run build_dataset(api) first.")
        return

    num_row_groups = parquet_file.num_row_groups
    logging.info(f"Dataset contains {num_row_groups} row groups.")
    
    if num_row_groups < 2:
        logging.error("Not enough row groups to create a train/test split. Need at least 2.")
        return
    test_group_index = num_row_groups - 1
    train_groups = list(range(num_row_groups - 1))
        
    logging.info(f"Reading test data from row group {test_group_index}...")
    test_df = parquet_file.read_row_group(test_group_index).to_pandas()
    
    feature_count = 134
    X_test = test_df[[str(x) for x in range(feature_count)]]
    y_test = test_df[[x for x in "abcdefghijklmnopqrstuvwxyz"]]
    del test_df

    model = api.MixtureOfExperts()
    model.iterations = 2000 # Set desired iterations here
    logging.info("--- Starting Incremental Training ---")

    for i in train_groups:
        logging.info(f"\n--- Training on row group {i + 1}/{len(train_groups)} ---")
        train_chunk = parquet_file.read_row_group(i).to_pandas()
        X_train_chunk = train_chunk[[str(x) for x in range(feature_count)]]
        y_train_chunk = train_chunk[[x for x in "abcdefghijklmnopqrstuvwxyz"]]
        
        is_first_batch = (i == train_groups[0])
        model.fit(X_train_chunk, y_train_chunk, warm_start=not is_first_batch)
        
        del train_chunk, X_train_chunk, y_train_chunk

    logging.info("\n--- Training Complete ---")
    logging.info("\n--- Loading final models for evaluation ---")
    model.load_trained_models() 
    
    logging.info("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test)
    accuracy = model.accuracy(y_pred, y_test)
    logging.info(f"\nOverall Balanced Accuracy on Test Set: {accuracy:.4f}")

    logging.info("\n--- Saving Model ---")
    model.save("models/model_134_features_15.pkl")
    logging.info("Model saved to models/model_134_features.pkl")
    model.cleanup()


def play(api,game_count):
    logging.info("Getting baseline statistics...")
    try:
        [initial_runs,_,_,initial_successes] = api.my_status()
        logging.info(f"Initial stats: {initial_successes} wins in {initial_runs} runs")
    except Exception as e:
        initial_runs,initial_successes = 0,0
        logging.info(f"No prior games found. Starting fresh")

    logging.info(f"\nRunning a batch of {game_count} practice games")

    for i in range(game_count):
        logging.info(f"Running game {i+1}")
        api.start_game(practice=1,verbose=True)
        time.sleep(0.1)
        
    logging.info("\n Getting updated statistics...")
    [final_runs,_,_,final_successes] = api.my_status()
    logging.info(f"Final stats: {final_successes} wins in {final_runs} runs")

    batch_runs=final_runs-initial_runs
    batch_successes=final_successes-initial_successes

    if batch_runs>0:
        batch_success_rate=batch_successes/batch_runs
        logging.info("\n---Performance of this Batch ---")
        logging.info(f"Games played in this test: {batch_runs}")
        logging.info(f"Wins in this test: {batch_successes}")
        logging.info(f"Success rate for this batch: {batch_success_rate:.3f}({batch_successes}/{batch_runs})")
    else:
        logging.info("\n---No new games played in this batch---")

    cumilative_success_rate=final_successes/final_runs if final_runs>0 else 0
    logging.info(f"\nCumilative Success Rate: (All Time): {cumilative_success_rate:.3f}({final_successes}/{final_runs})")


def submit(api):
    logging.info("Getting baseline statistics...")
    try:
        [initial_runs,_,_,initial_successes] = api.my_status()
        logging.info(f"Initial stats: {initial_successes} wins in {initial_runs} runs")
    except Exception as e:
        initial_runs,initial_successes = 0,0
        logging.info(f"No prior games found. Starting fresh")


    for i in range(1000):
        logging.info(f"Running game {i+1}")
        api.start_game(practice=0,verbose=False)
        time.sleep(0.1)
        
    logging.info("\n Getting updated statistics...")
    [final_runs,_,_,final_successes] = api.my_status()
    logging.info(f"Final stats: {final_successes} wins in {final_runs} runs")

    batch_runs=final_runs-initial_runs
    batch_successes=final_successes-initial_successes

    if batch_runs>0:
        batch_success_rate=batch_successes/batch_runs
        logging.info("\n---Performance of this Batch ---")
        logging.info(f"Games played in this test: {batch_runs}")
        logging.info(f"Wins in this test: {batch_successes}")
        logging.info(f"Success rate for this batch: {batch_success_rate:.3f}({batch_successes}/{batch_runs})")
    else:
        logging.info("\n---No new games played in this batch---")

    cumilative_success_rate=final_successes/final_runs if final_runs>0 else 0
    logging.info(f"\nCumilative Success Rate: (All Time): {cumilative_success_rate:.3f}({final_successes}/{final_runs})")

if __name__=="__main__":
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = datetime.now().strftime('hangman_run_%Y-%m-%d_%H-%M-%S.log')
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    logging.info("--- Starting Hangman AI ---")
    token=os.getenv("TOKEN")
    api = HangmanAPI(access_token=token, timeout=2000)
    
    # build_dataset(api)
    
    # train(api)
    
    # play(api, 50)
    submit(api)
    
    logging.info("--- Run Finished ---")