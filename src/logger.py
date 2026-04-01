#Logger is used to record what the program is doing and helps in debugging and tracking errors.
#Saves messages to a file 
#Used in real projects (industry standard)
#It tracks your program step-by-step
#instead of writing print("Data loaded"),we use logger.info("Data loaded successfully")
#why we do not use print-->
                           #Temporary
                           #Not saved anywhere
                           #Hard to debug big projects
#Logger is like an advanced version of print()
#

import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(

    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,

)


