# Slitherlink Solver
## Installation Instructions
### Operating System
Windows
### Required Software
* Python 3.7 or above
* Minizinc IDE(Bundled)
  * Downloaded from: https://www.minizinc.org/ide/
### Required Python Packages

`pip install -r requirements.txt`

OR

`pip install numpy`

`pip install beautifulsoup4==4.10.0`

`pip install minizinc`

`pip install z3-solver`

`pip install requests==2.23.0`

### Configuration of Environment Variables
Right click on 'This PC' -> Properties -> Advanced system settings -> Environment Variables -> 'Path' (under System variables) ->
Add the path of the installed Minizinc IDE to the 'Path'

### Running the program
* Open the Command Prompt
* Go to the folder where the main.py file is located
* Using  `python main.py` to run the program (Tips: The program will automatically execute 6 algorithms 10 times before stopping. Each time it will show the solution and time cost.)

Note: When using IDEs such as pycharm or vscode, there is a certain chance that the Gecode solver cannot be run, causing the program to interrupt and report an error, so it is recommended to use the command prompt to run this program.
