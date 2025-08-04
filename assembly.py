import os
import shutil
from dotenv import load_dotenv
from MOE import HangmanAPI

load_dotenv()

print("--- Starting Final Model Assembly ---")

token = os.getenv("TOKEN")
api = HangmanAPI(access_token=token, timeout=2000)
model = HangmanAPI.MixtureOfExperts()
model.iterations = 1500

print(f"Looking for intermediate models in: {model.model_dir}")
model.load_trained_models()
print("\n--- Saving Final Assembled Model ---")
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/model.pkl")
print("Successfully saved final model to 'models/model.pkl'")
model.cleanup()

print("\n--- Assembly Complete ---")
print("Your submission file 'models/model.pkl' is ready.")