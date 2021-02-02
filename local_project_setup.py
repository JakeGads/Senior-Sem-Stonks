import os
import platform

if not os.path.exists('env'):
    command = input("what is the command linked to python 3 (python, python3, py, py3):\t").strip()
    
    os.system(f'{command} -m pip install virtualenv')
    os.system(f'{command} -m venv env')

    if platform.system() == 'Windows':
        # todo I need one of you windows kids to verify this path
        os.system(f'env\\scripts\\activate')
    else:
        os.platform(f'source env/bin/activate')

    os.system('pip install --upgrade pip')
    os.system('pip install -r requirements.pip')