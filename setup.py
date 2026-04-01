from setuptools import find_packages,setup
from typing import List


hypen_e_dot = "-e ."

def get_requirements(file_path:str)->list[str]:
    '''
    this function will return the list of requirements
    
    '''
    requirements=[]
    with open('requirements.txt') as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]


        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    
    return requirements




        

setup(
    name = 'ml2.project',
    version = '0.0.1',
    author = 'RITHESH',
    author_email = "ritheshrmr@gamil.com",
    packages = find_packages(),
    install_requires=get_requirements("requirements.txt")

)