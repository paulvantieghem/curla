import utils
import time
import carla_env

CARLA_RESET_FREQ = 2000

def make_env(port):
    env = carla_env.CarlaEnv(seconds_per_episode=1, port=port)
    env.reset()
    return env

def main():

    # Initializations
    port = 50_000
    env = make_env(port)
    episode = 0
    done = False
    restart_server = False

    # Main loop
    for step in range(200):

        # Check if we need to restart the server
        if step % CARLA_RESET_FREQ == 0 and step > 0:
            restart_server = True

        if done:
            episode += 1
            obs = env.reset()
            done = False

        if restart_server:
            print(f'-------------------- Resetting CARLA server at step {step}')
            env.deactivate()
            del env
            port += 1
            env = make_env(port)
            restart_server = False

        action = env.action_space.sample()
        time.sleep(0.05)
        obs, reward, done, info = env.step(action)
        print(f"Step {step}, episode {episode}, reward {reward}, done {done}")
        
    env.deactivate()


if __name__ == '__main__':
    main()