import multiprocessing as mp
import os
import psutil
import subprocess
import time
import pkg_resources


class CarlaServer:
    CARLA_ROOT = os.environ.get('CARLA_ROOT')
    CARLA_VERSION = pkg_resources.get_distribution('carla').version

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