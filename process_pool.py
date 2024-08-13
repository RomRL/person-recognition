import multiprocessing
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for the pool and model initialization function
process_pool = None

def initialize_model(device):
    global mtcnn, model
    mtcnn = MTCNN(device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("Model and MTCNN initialized in process.")

# Function to initialize the pool, called once in your main application
def initialize_pool(num_processes=4):
    global process_pool
    if process_pool is None:
        process_pool = multiprocessing.Pool(
            processes=num_processes, initializer=initialize_model, initargs=(device,)
        )
        print("Process pool initialized.")
