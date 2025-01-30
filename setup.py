from setuptools import find_packages,setup 
from typing import List 

def get_requirements(file_path:str)->List[str]:
    """this will return the list of requiremnts."""
    requiremetns=[]
    with open(file_path) as file_obj:
        requiremetns=file_obj.readlines()
        requiremetns=[req.replace('\n','') for req in requiremetns]

        if '-e .' in requiremetns:
            requiremetns.remove('-e .')


setup(
    name='ml_preoject',
    packages=find_packages(),
    install_requires=get_requirements('./reqment.txt'),
    version='0.0.1'
)