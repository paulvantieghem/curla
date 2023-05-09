import multiprocessing as mp
import os
import psutil
import subprocess
import time
import pkg_resources
import carla

class CarlaClient(carla.Client):
    LOAD_WORLD_ATTEMPTS = 5

    def test_connection(self):
        # Test the connection with the CARLA server by retrieving the world data.
        # This is a blocking call until the server responds or a RuntimeError occurs.
        self.get_world()

    def load_world(self, *args, **kwargs):
        # Safely load a new map by retrying it a couple of times.
        world = None
        attempt = 0
        while world is None and attempt < self.LOAD_WORLD_ATTEMPTS:
            try:
                world = super().load_world(*args, **kwargs)
            except RuntimeError:
                world = None
                attempt += 1
        if world is None:
            raise RuntimeError("Could not load world")
        else:
            return world


class CarlaServer:
    CARLA_ROOT = os.environ.get('CARLA_ROOT') 
    carla_version_string = pkg_resources.get_distribution('carla').version
    x, y, z = carla_version_string.split('.')
    CARLA_VERSION = (int(x), int(y), int(z))

    def __init__(self, port=2000, offscreen=True, sound=False, launch_delay=30, launch_retries=3,
                 connect_timeout=10, connect_retries=3):
        self._port = port
        self._offscreen = offscreen
        self._sound = sound
        self._launch_delay = launch_delay
        self._launch_retries = launch_retries
        self._connect_timeout = connect_timeout
        self._connect_retries = connect_retries
        self._server = None

    def launch(self, delay=None, retries=None):

        print(f'CARLA_VERSION: {self.CARLA_VERSION}')

        # Check if the server is already running
        if self.is_active:
            print("CARLA server is already running.")
            return

        # Set delays and retries
        if delay is None:
            delay = self._launch_delay
        if retries is None:
            retries = self._launch_retries

        # Setup arguments and environment variables
        args = [f'-carla-port={self._port}']
        env = os.environ.copy()
        if self._offscreen:
            # Offscreen rendering (see https://carla.readthedocs.io/en/latest/adv_rendering_options/#off-screen-mode)
            if self.CARLA_VERSION >= (0, 9, 12):
                args.append('-RenderOffScreen')
            else:
                args.append('-opengl')
                env['DISPLAY'] = ''
        if not self._sound:
            args.append('-nosound')
        carla_path = os.path.join(self.CARLA_ROOT, 'CarlaUE4.exe' if os.name == 'nt' else 'CarlaUE4.sh')
        cmd = [carla_path, *args]

        # Try launching the server
        attempt = 0
        self._server = None
        while self._server is None and attempt < retries:
            attempt += 1

            # Try to launch server and wait for delay seconds before attempting the first connection.
            print(f"Launching CARLA server (attempt {attempt}/{retries})")
            self._server = subprocess.Popen(cmd, env=env)
            time.sleep(delay)
            try:
                if self.is_active:
                    # Try to run _test_target in a client session with small timeout and many retries.
                    # This ensures the CARLA server is completely ready to handle further client connections.
                    self.run_client(self._test_target, worker_threads=1, timeout=5, retries=20, verbose=False)
            except RuntimeError:
                # Server didn't respond to client connections, either it crashed or is unresponsive.
                # Kill server process if it didn't crash already and try again.
                self.kill()
                self._server = None

            if not self.is_active:
                # Server process terminated, retry launch
                self._server = None
                print("Launching CARLA server failed.")

        # If the server is still not active after all retries, give up.
        if not self.is_active:
            raise RuntimeError("Could not launch CARLA server.")
        else:
            print("CARLA server ready")

    def kill(self):
        if self.is_active:
            # CARLA server spawns child processes, make sure to kill them too (otherwise the simulator keeps running)
            children = psutil.Process(self._server.pid).children(recursive=True)
            for child in children:
                child.kill()
            self._server.kill()
            self._server = None

    @property
    def is_active(self):
        return self._server is not None and self._server.poll() is None
    

    def run_client(self, target, worker_threads=0, timeout=None, retries=None, verbose=True):
        # To prevent issues when clients are closed, it is recommended to launch the client in a separate process.
        # See https://github.com/carla-simulator/carla/issues/2789#issuecomment-689619998
        if timeout is None:
            timeout = self._connect_timeout
        if retries is None:
            retries = self._connect_retries

        err_out, err_in = mp.Pipe(False)
        p = mp.Process(target=self._client_proc, args=(self._port, worker_threads, timeout, retries, verbose, target, err_in))
        try:
            p.start()
            # Receive exception or `False` from child process
            e = err_out.recv()
            if e:
                # Raise it again in main process
                raise e
        finally:
            p.join()

    @staticmethod
    def _client_proc(port, worker_threads, timeout, retries, verbose=True, target=None, err=None):
        try:
            client = None
            attempt = 0
            while client is None and attempt < retries:
                attempt += 1
                if verbose:
                    print(f"Connecting to CARLA server (attempt {attempt}/{retries})")
                try:
                    client = CarlaClient('localhost', port, worker_threads)
                    client.set_timeout(timeout)
                    client.test_connection()  # Blocking call until server responds or RuntimeError occurs
                except RuntimeError:
                    # Client connection timed out, retry connection
                    client = None
                    if verbose:
                        print("Client connection timed out.")

            if client is None:
                exc = RuntimeError("Could not connect to CARLA server.")
                if err:
                    err.send(exc)
                    return
                else:
                    raise exc
            if verbose:
                print("Client ready")

            if target is None:
                return client
            else:
                try:
                    target(client)
                except Exception as exc:
                    # Send any exception through the error connection, such that it can be raised in the main process
                    if err:
                        err.send(exc)
                    # Raise it in child process as well to make debugging easier for users
                    raise exc
        finally:
            if err:
                # Always close error connection before returning
                err.send(False)
                err.close()

    @staticmethod
    def _test_target(client):
        # Test target to wait for the CARLA server to be ready for further client connections.
        client.test_connection()