import os
import platform

def brr():
    if not os.path.exists('env'):
        print('goal is to aim at python 3.7.x, this may be difficult on windows, do not under any circumstances point this at python 3.9.x')
        command = input("what is the command linked to python 3 (python, python3, py, py3):\t").strip()
        
        if command == '':
            return

        os.system(f'{command} -m pip install virtualenv')
        os.system(f'{command} -m venv env')

        if platform.system() == 'Windows':
            # todo I need one of you windows kids to verify this path
            os.system(f'env\\scripts\\activate')
        else:
            os.platform(f'source env/bin/activate')

        os.system('pip install --upgrade pip')
        os.system('pip install -r requirements.pip')

brr()