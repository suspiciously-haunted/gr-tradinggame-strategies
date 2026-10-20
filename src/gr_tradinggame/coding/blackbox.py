from contextlib import redirect_stderr, redirect_stdout
import cloudpickle as pickle
import base64
from IPython.terminal.interactiveshell import TerminalInteractiveShell
import io


class Blackbox:
    def __init__(self, source, name='play'):
        self.name = name
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        with redirect_stderr(self.stderr), redirect_stdout(self.stdout):
            self.shell = TerminalInteractiveShell()
            out = self.shell.run_cell(source)
            if out.error_before_exec is not None:
                raise out.error_before_exec
            if out.error_in_exec is not None:
                raise out.error_in_exec
        if not isinstance(name, str) or name not in self.shell.user_ns:
            raise Exception(f"Your source code must define a function called `{self.name}`, but only has `{self.shell.user_ns.keys()}`")

    def __call__(self, *args, **kwargs):
        input_variable_name = '__magic_input__'
        with redirect_stderr(self.stderr), redirect_stdout(self.stdout):
            self.shell.user_ns[input_variable_name] = args, kwargs
            out = self.shell.run_cell(f'{self.name}(*{input_variable_name}[0], **{input_variable_name}[1])')
            if out.error_before_exec is not None:
                raise out.error_before_exec
            if out.error_in_exec is not None:
                raise out.error_in_exec
            return out.result


class Plackbox:
    def __init__(self, source):
        self.foo = pickle.loads(source)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def __call__(self, *args, **kwargs):
        with redirect_stderr(self.stderr), redirect_stdout(self.stdout):
            return self.foo(*args, **kwargs)


def generate_function(data):
    if data[0] == 'p':
        return Plackbox(base64.b64decode(data[1:]))
    else:
        return Blackbox(base64.b64decode(data[1:]).decode('utf-8'))
